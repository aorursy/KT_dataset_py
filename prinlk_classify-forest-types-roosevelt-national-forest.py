# Package imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,make_scorer



import seaborn as sns

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reads the .csv files into the train and test pandas data frames



train = pd.read_csv("../input/learn-together/train.csv") # training data set

test = pd.read_csv("../input/learn-together/test.csv") # testing data set
# Confirming the size of 'train' data frame - 56 columns with 15120 rows,

# Confirming the size of 'test' data frame - 55 columns with 565892 rows

train.shape,test.shape
# First few rows of the 'train' dataset

train.head()
# Confirming data types of the data frame

train.info()
train.describe().T
print(train['Soil_Type40'].value_counts())
# Let's look into if any column has missing values to handle in the process

train.isna().sum()
# Remove the Id column as it has no use in coorelation matrix

train = train.drop(["Id"], axis = 1)
# Let's check the correlation among non-binary type data and draw a heat map. So the approach is

# 1. Find all the binary type columns without typing names

# 2. Then find all non-binary type columns



all_columns = train.columns # All column names

binary_cols = [col_ for col_ in all_columns if (set(train[col_].unique()).issubset([0,1]))] # all binary columns

non_binary_cols = set(all_columns) - set(binary_cols) - set(['Cover_Type']) # all non binary columns excluding 'Cover Type'



# Let's get the correlation matrix for non_binary_cols

matrix = train[non_binary_cols].corr()



# Let's draw the heatmap for correlation matrix for non_binary_cols

plt.figure(figsize=(10, 7)) 

plt.title("Correlation Plot For Non-binary Fields")

mask = np.zeros_like(matrix)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"): 

    ax = sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=.5, mask=mask, cmap="viridis")

# Let's add wilderness area description for the dataset for clarity.

train_c = train.copy() # Create a copy to keep original dataframe unchanged



def wilderness(row):

    if(row['Wilderness_Area1'] == 1):

        return 'Rawah'

    elif(row['Wilderness_Area2'] == 1):

        return 'Neota'

    elif(row['Wilderness_Area3'] == 1):

        return 'Comanche Peak'

    elif(row['Wilderness_Area4'] == 1):

        return 'Cache la Poudre'

    else:

        return 'No-Wilderness'

    

# Let's add a new column for 'Wilderness' description

train_c['Wilderness'] = [wilderness(row_[1]) for row_ in train_c.iterrows()]



# Let's add a new column for 'Cover_Type' description

cover_dict ={1:'Spruce/Fir',2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}

train_c['Cover'] = train_c['Cover_Type'].map(cover_dict)
# The count distribution according to 'Wilderness' in the training dataset

p = sns.countplot(data=train_c,y = 'Wilderness')
# Let's draw kdeplots for few selected attributes (attribs) classified on 'Cover' and 'Wilderness'

attribs = ['Elevation','Aspect','Horizontal_Distance_To_Hydrology','Hillshade_Noon'] # Selected Attributes

for attr_ in attribs:

    g = sns.FacetGrid(train_c, col="Wilderness",hue="Cover",height=5)

    g.map(sns.kdeplot, attr_,shade=True)

    g.add_legend()
test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)



# Remove the columns Soil_Type7 and Soil_Type15 from the test set as they all 0

# cols_to_drop = ['Soil_Type7', 'Soil_Type15']

# test = test[test.columns.drop(cols_to_drop)]



train.head()
# train test split 20% for validation

X = train.drop(['Cover_Type'], axis=1)

y = train['Cover_Type']

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2,random_state=0)
# Import StandardScaler

# from sklearn.preprocessing import StandardScaler



# # Instantiate StandardScaler and use it to rescale X_train and X_val

# scaler = StandardScaler()

# rescaledX_train = scaler.fit_transform(X_train)

# rescaledX_val = scaler.fit_transform(X_val)
# Building the model 

model = ExtraTreesClassifier(n_estimators=1000,random_state=42)



model.fit(X_train,y_train)

predictions = model.predict(X_val)

accuracy_score(y_val, predictions)
# from sklearn.model_selection import GridSearchCV

# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'bootstrap': [True],

#     'max_depth': [80, 90, 100, 110],

#     'max_features': [2, 3],

#     'min_samples_leaf': [3, 4, 5],

#     'min_samples_split': [8, 10, 12],

#     'n_estimators': [500, 700]

# }



# model = RandomForestClassifier()



# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = model, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)



# # Fit the grid search to the data

# grid_search.fit(X_train,y_train)

# grid_search.best_params_



# best_grid = grid_search.best_estimator_

# grid_accuracy = evaluate(best_grid, X_val, y_val)

# def evaluate(model, test_features, test_labels):

#     predictions = model.predict(test_features)

#     errors = abs(predictions - test_labels)

#     mape = 100 * np.mean(errors / test_labels)

#     accuracy = 100 - mape

#     print('Model Performance')

#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

#     print('Accuracy = {:0.2f}%.'.format(accuracy))

    

#     return accuracy
# Drawing confusion matrix

cm = confusion_matrix(y_val, predictions)

plt.figure(figsize=(10,7))

sns.heatmap(cm, annot=True,fmt="2d",linewidths=.5,cmap="viridis")

plt.xlabel('Predicted')

plt.ylabel('Truth')
# # Using GridSearchCV for finding right hyperparameters

# # Building the model 

# model = RandomForestClassifier()



# parameters = {'n_estimators': [100, 200, 300]}



# # Type of scoring used to compare parameter combinations

# acc_scorer = make_scorer(accuracy_score)



# # Run the grid search

# grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer)

# # grid_obj = grid_obj.fit(X_train, y_train)

# grid_obj = grid_obj.fit(rescaledX_train, y_train)



# # Set the clf to the best combination of parameters

# model = grid_obj.best_estimator_



# # Fit the best algorithm to the data. 

# model.fit(rescaledX_train, y_train)



# # Prediction

# predictions = model.predict(rescaledX_val)

# accuracy_score(y_val, predictions)



# Use K-fold Cross Validation

# from sklearn.model_selection import cross_val_score,cross_val_predict

# from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC



# score = cross_val_score(model,X_train,y=y_train,cv=5).mean()

# print(score)
# Using XGBoost

# import xgboost as xgb



# xgb_model = xgb.XGBClassifier(objective="multi:softmax", random_state=42)

# xgb_model.fit(rescaledX_train,y_train)



# predictions = xgb_model.predict(rescaledX_val)

# accuracy_score(y_val, predictions)
# Evaluating the feature importance

# import eli5

# from eli5.sklearn import PermutationImportance



# perm = PermutationImportance(model, random_state=0).fit(X_val, y_val)

# eli5.show_weights(perm, feature_names = X_val.columns.tolist())
# Using GridSearchCV for finding right hyperparameters

test.head()
# scaler = StandardScaler()

# rescaled_test = scaler.fit_transform(test)





# Applying the model on the test set

# test_pred = model.predict(rescaled_test)

test_pred = model.predict(test)

# test_pred = cross_val_predict(model,X_train,y=y_train,cv=5) # model.predict(test)

# Save test predictions to file

output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('submission.csv', index=False)