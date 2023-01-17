import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import QuantileTransformer



from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.model_selection import cross_val_score



from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report, accuracy_score

from sklearn.pipeline import Pipeline



from mlxtend.classifier import StackingCVClassifier



from xgboost import XGBClassifier



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('../input/learn-together/train.csv')

test_df=pd.read_csv('../input/learn-together/test.csv')
train_df.head()
test_df.head()
print("shape training csv: %s" % str(train_df.shape)) 

print("shape test csv: %s" % str(test_df.shape)) 
train_df.columns
print(train_df.dtypes.value_counts())

print(test_df.dtypes.value_counts())
train_df.describe()
print(f"Missing Values in train: {train_df.isna().any().any()}")

print(f"Missing Values in test: {test_df.isna().any().any()}")
train_df = train_df.drop(["Id"], axis = 1)



test_ids = test_df["Id"]

test_df = test_df.drop(["Id"], axis = 1)
plt.figure(figsize=(10,5))

plt.title("Distribution of forest categories (Target Variable)")

sns.distplot(train_df["Cover_Type"])

plt.show()
sns.FacetGrid(train_df, hue="Cover_Type", size=10).map(plt.scatter, "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology").add_legend()
temp = train_df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]

plt.figure(figsize=(15,12))

pd.plotting.parallel_coordinates(temp,'Cover_Type', colormap=plt.get_cmap("Set1"))

plt.title("parallel plots of Hillshade with forest categories")

plt.xlabel("Hillshade")

plt.show()
print("percent of negative values (training): " + '%.3f' % ((train_df.loc[train_df['Vertical_Distance_To_Hydrology'] < 0].shape[0] / train_df.shape[0])*100))

print("percent of negative values (testing): " + '%.3f' % ((test_df.loc[test_df['Vertical_Distance_To_Hydrology'] < 0].shape[0]/ test_df.shape[0])*100))
plt.figure(figsize=(12,8))

sns.boxplot(train_df['Vertical_Distance_To_Hydrology'])

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(test_df['Vertical_Distance_To_Hydrology'])

plt.show()
cols=list(train_df.columns)
#list of columns other than Soil_Type and Wilderness_Area

other_columns=[]

for i in range(10):

    other_columns.append(cols[i])

print(other_columns)

print(f"Train Column Types: {set(train_df.dtypes)}")

print(f"Test Column Types: {set(test_df.dtypes)}")

for column in train_df.columns:

    print(column, train_df[column].nunique())
for column in test_df.columns:

    print(column, test_df[column].nunique())
print("- - - Train - - -")

print(train_df["Soil_Type7"].value_counts())

print(train_df["Soil_Type15"].value_counts())

print("\n")

print("- - - Test - - -")

print(test_df["Soil_Type7"].value_counts())

print(test_df["Soil_Type15"].value_counts())
train_df.drop(["Soil_Type7", "Soil_Type15"], axis = 1,inplace=True)

test_df.drop(["Soil_Type7", "Soil_Type15"], axis = 1,inplace=True)
train_minmax=train_df[other_columns]

test_minmax=test_df[other_columns]

mm_scaler = MinMaxScaler()

# my_train_minmax = mm_scaler.fit_transform(train_df[other_columns])

mm_scaler.fit(train_minmax)

train_trans=mm_scaler.transform(train_minmax)

test_trans=mm_scaler.transform(test_minmax)



temp_train=pd.DataFrame(train_trans)

temp_test=pd.DataFrame(test_trans)

temp_test.head()

for i in range(10):

    temp_train.rename(columns={i:other_columns[i]},inplace=True)

    temp_test.rename(columns={i:other_columns[i]},inplace=True)

temp_train.head()

train_df[other_columns]=temp_train[other_columns]

test_df[other_columns]=temp_test[other_columns]

train_df.head()
# def model_training(model,X_train,y_train):

#     scores =  cross_val_score(model, X_train, y_train,

#                               cv=5)

#     return scores.mean()
X=train_df.drop(['Cover_Type'], axis=1)

y=train_df['Cover_Type']

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.25,random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
# importing K-Nearest Neighbors Classifier function

from sklearn.neighbors import KNeighborsClassifier

knn_model=KNeighborsClassifier(n_jobs=-1)

# classifier learning the model

knn_model = knn_model.fit(X_train, y_train)
# Using 10 K-Fold CV on data, gives peroformance measures

accuracy  = cross_val_score(knn_model, X_train, y_train, cv = 10, scoring = 'accuracy')

f1_score = cross_val_score(knn_model, X_train, y_train, cv = 10, scoring = 'f1_macro')

# calculating mean of all 10 observation's accuracy and f1, 

# taking percent and rounding to two decimal places

acc_mean = np.round(accuracy.mean() * 100, 2)

f1_mean = np.round(f1_score.mean() * 100, 2)

print("The accuracy score of training set:", acc_mean,"%")

print("f1 score is", f1_mean)
# importing model for feature importance

# from sklearn.ensemble import ExtraTreesClassifier

# passing the model

# etc_model = ExtraTreesClassifier(n_jobs=-1,random_state = 42)

# training the model

# etc_model=etc_model.fit(X_train, y_train)

# etc_model=etc_model.fit(X, y)



# extracting feature importance from model and making a dataframe of it in descending order

# etc_feature_importances = pd.DataFrame(etc_model.feature_importances_, index = X.columns, columns=['ETC']).sort_values('ETC', ascending=False)

# show top 10 features

# etc_feature_importances.head(10)
sample_train = train_df[['Elevation','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Aspect','Wilderness_Area4',

            'Hillshade_Noon','Hillshade_3pm','Hillshade_9am','Slope','Soil_Type22','Soil_Type10','Soil_Type4','Soil_Type34','Soil_Type34','Wilderness_Area3','Soil_Type12',

            'Soil_Type2','Wilderness_Area1', 'Cover_Type']]
# feeding sample features to var 'X'

X = sample_train.iloc[:,:-1]

# feeding our target variable to var 'y'

y = sample_train['Cover_Type']
# etc_model = ExtraTreesClassifier(n_jobs=-1,random_state = 42)

# etc_model = etc_model.fit(X, y)

# accuracy  = cross_val_score(etc_model, X, y, cv = 10, scoring = 'accuracy')

# f1_score = cross_val_score(etc_model, X, y, cv = 10, scoring = 'f1_macro')



# calculating mean of all 10 observation's accuracy and f1, taking percent and rounding to two decimal places

# acc_mean = np.round(accuracy.mean() * 100, 2)

# f1_mean = np.round(f1_score.mean() * 100, 2)



# returns performance measure and time of the classifier 

# print("The accuracy score of this classifier on our training set is", acc_mean,"%")

# print("f1 score is", f1_mean,"%")
# Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 8)]

# Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# max_depth.append(None)

# Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

# bootstrap = [True, False]

# Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}

# print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

# rf_model = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

# rf_random.fit(X_train, y_train)
# rf_random.best_params_
# model = RandomForestClassifier(n_estimators=100,random_state = 42)

# model=RandomForestClassifier(n_estimators=885,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=110,bootstrap=False,random_state = 42)

# Train the model on training data

# model.fit(X_train, y_train)
# predictions = model.predict(X_val)

# accuracy_score(y_val, predictions)
model_rf=RandomForestClassifier(n_estimators=885,

                                   min_samples_split=2,

                                   min_samples_leaf=1,

                                   max_features='sqrt',

                                   max_depth=110,

                                   bootstrap=False,

                                  random_state=42)

# model_rf.fit(X,y)
model_xgb = OneVsRestClassifier(XGBClassifier(random_state=42))

# model_xgb.fit(X,y)
model_et = ExtraTreesClassifier(random_state=42)
# sclf = StackingCVClassifier(classifiers=[model_rf,model_xgb,model_et],

#                             use_probas=True,

#                             meta_classifier=model_rf,random_state=42)

# labels = ['Random Forest', 'XGBoost', 'ExtraTrees', 'MetaClassifier']
# for clf, label in zip([model_rf, model_xgb, model_et, model_rf], labels):

#     scores = cross_val_score(clf, X_train.values, y_train.values.ravel(),

#                              cv=5,scoring='accuracy')

#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# sclf.fit(X_train.values, y_train.values.ravel())
# val_pred = sclf.predict(X_val.values)

# acc = accuracy_score(y_val, val_pred)

# print(acc)
# model_sclf = StackingCVClassifier(classifiers=[model_rf,model_xgb,model_et],

#                                use_probas=True,

#                                meta_classifier=model_rf,random_state=42)

# model_sclf.fit(X.values, y.values.ravel())
sample_test = test_df[['Elevation','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Aspect','Wilderness_Area4',

            'Hillshade_Noon','Hillshade_3pm','Hillshade_9am','Slope','Soil_Type22','Soil_Type10','Soil_Type4','Soil_Type34','Soil_Type34','Wilderness_Area3','Soil_Type12',

            'Soil_Type2','Wilderness_Area1']]
# test_pred = model_rf.predict(test_df)

# test_pred = model_sclf.predict(sample_test.values)

# test_pred = etc_model.predict(sample_test.values)
# Save test predictions to file

# output = pd.DataFrame({'Id': test_ids,

#                        'Cover_Type': test_pred})

# output.to_csv('submission_sclf2.csv', index=False)