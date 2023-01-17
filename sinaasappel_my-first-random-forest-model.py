import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np



train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col = 'Id')

test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col = 'Id')
train.head()
train.describe().T
%matplotlib inline 

import matplotlib.pyplot as plt



train.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points'].hist(bins = 50, figsize=(20, 15))

plt.show()
train.loc[:,'Wilderness_Area1':'Wilderness_Area4'].hist(bins = 50, figsize=(20, 15))

plt.show()
train.loc[:,'Soil_Type1':'Soil_Type40'].hist(bins = 50, figsize=(20, 15))

plt.show()
from sklearn.model_selection import train_test_split



# Split into validation and training data, set to random_state 1

train_set, test_set = train_test_split(train, test_size = 0.20, random_state = 1)
## make training set

# Create target object and call it y

y_train = train_set.Cover_Type

X_train = train_set.drop('Cover_Type', axis = 1)



# make test set

y_test = test_set.Cover_Type

X_test = test_set.drop('Cover_Type', axis = 1)



# make final test set

X_final_test = test.copy()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

    

# make a function to extract columns

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y = None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
# here you can select your favorite numerical features

num_attribs = list(train_set.iloc[:,0:10].columns)



# features that should be removed, they were not present in train set

my_list = ['Soil_Type7', 'Soil_Type15']



# here you can select your favorite binary features

cat_attribs = list(train_set.iloc[:,10:54].columns)

cat_attribs = [e for e in cat_attribs if e not in (my_list)]
# make pipeline for numerical features

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),

                         ('std_scaler', StandardScaler(),

                        )])



cat_pipeline = Pipeline([('Selector', DataFrameSelector(cat_attribs))])



# combine both pipelines

from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list = [('num_pipeline', num_pipeline),

                                                 ('cat_pipeline', cat_pipeline)])
# run the pipeline to prepare the train data

X_train_prepared = full_pipeline.fit_transform(X_train)



# run the pipeline to prepare the test data

X_test_prepared = full_pipeline.transform(X_test)



# run the pipeline to prepare the final test data

X_final_test_prepared = full_pipeline.transform(X_final_test)
X_train_prepared.shape
X_test_prepared.shape
X_test_prepared.shape
from sklearn.ensemble import RandomForestClassifier

import warnings



# to remove warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# fit the model with default options

rf = RandomForestClassifier(random_state = 0)

rf.fit(X_train_prepared, y_train)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



scores_rf = cross_val_score(rf, X_train_prepared, y_train, cv=10, scoring='accuracy')



# Get the mean accuracy score

print("Average accuracy score random forest model (across experiments):")

print(scores_rf.mean())
# confusion matrix

from sklearn.metrics import confusion_matrix



y_test_predict = rf.predict(X_test_prepared)



# make a confusion matrix

conf_mx_test = confusion_matrix(y_test, y_test_predict)



# make a normalized confusion matrix

row_sums = conf_mx_test.sum(axis = 1, keepdims = True)

norm_conf_mx = conf_mx_test / row_sums
import seaborn as sns



f, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)



# names for labeling

alpha = ['Spruce/Fir', 'Lodgehole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas/Fir', 'Krummholz']



sns.heatmap(conf_mx_test, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[0])



sns.heatmap(norm_conf_mx, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[1])
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rf, random_state=1).fit(X_train_prepared, y_train)



eli5.show_weights(perm, feature_names = num_attribs + cat_attribs, top = 60)
#from sklearn.model_selection import GridSearchCV



#param_grid = [

#    {'n_estimators': [150, 200, 250, 300], 'max_features': [4, 8, 16, 32]},

#    {'bootstrap': [False], 'n_estimators':[150, 200, 250, 300], 'max_features':[4, 8, 16, 32]}

#]



#rf_final = RandomForestClassifier()



#grid_search = GridSearchCV(rf_final, param_grid, cv = 5, scoring = 'accuracy')



#grid_search.fit(X_train_prepared, y_train)
#grid_search.best_params_



#{'bootstrap': False, 'max_features': 16, 'n_estimators': 300}
# fit the model with optimized parameters

rf_final = RandomForestClassifier(bootstrap=False, n_estimators = 300, 

                                      max_features = 16, random_state = 0)



#making the model using cross validation

scores_rf = cross_val_score(rf_final, X_train_prepared, y_train, cv=10, scoring='accuracy')



# and get scores

print("Average accuracy score (across experiments):")

print(scores_rf.mean())



# 0.783

# 0.816
rf_final.fit(X_train_prepared, y_train)



# make predictions using our model

y_test_predict = rf_final.predict(X_test_prepared)
# evaluate the results

from sklearn.metrics import accuracy_score



accuracy_score(y_test_predict, y_test) # 0.850
# make a confusion matrix

conf_mx_test = confusion_matrix(y_test, y_test_predict)



# make a normalized confusion matrix

row_sums = conf_mx_test.sum(axis = 1, keepdims = True)

norm_conf_mx = conf_mx_test / row_sums



# visualize confusion matrices

f, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)



# names for labeling

alpha = ['Spruce/Fir', 'Lodgehole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas/Fir', 'Krummholz']



sns.heatmap(conf_mx_test, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[0])



sns.heatmap(norm_conf_mx, annot=True, xticklabels=alpha, yticklabels = alpha, cbar=False, ax=axes[1])
# make predictions using the model

predictions_test_final = rf_final.predict(X_final_test_prepared)



# Save test predictions to file

output = pd.DataFrame({'ID': test.index,

                       'Cover_Type': predictions_test_final})



output.to_csv('submission.csv', index=False)