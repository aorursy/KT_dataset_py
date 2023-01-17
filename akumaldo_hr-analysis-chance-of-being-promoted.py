# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



%matplotlib inline





pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



import warnings

warnings.filterwarnings(action="ignore")
train = pd.read_csv("../input/train_LZdllcl.csv")

test = pd.read_csv("../input/test_2umaH9m.csv")



train.set_index("employee_id", inplace = True)



test.set_index("employee_id", inplace = True)
train.head() #taking a look at the first entries
##Merging the train and test dataset in order to have more data to train our model.



train['source']='train' #creating a label for the training and testing set

test['source']='test'



data = pd.concat([train, test],ignore_index=True)

print (train.shape, test.shape, data.shape) #printing the shape
#checking for null values:

data.isnull().sum()
## gonna quickly change the names of KPI and Awards won, making it easier to read

#changing those to better names

data.rename(columns={"KPIs_met >80%":"KPI"},inplace=True)

data.rename(columns={"awards_won?":"awards_won"},inplace=True)
data.apply(lambda x: len(x.unique())) # let's take a look at how many unique values does it have
#Filter categorical variables

categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

#Exclude ID cols and source:

categorical_columns = [x for x in categorical_columns if x not in ['gender','source']]

#Print frequency of categories

for col in categorical_columns:

    print ('\nFrequency of Categories for varible %s'%col)

    print (data[col].value_counts())
data['gender'] = data['gender'].map( {'f': 0, 'm': 1} ).astype(int) #mapping gender
data["education"].fillna('Bachelor\'s',inplace=True) #filling in all the null values before mapping it with the most frequent value

data['education'] = data['education'].map( {'Below Secondary': 0, 'Bachelor\'s': 1, 'Master\'s & above': 2} ).astype(int) #mapping gender
data.isnull().sum() #Previous year rating is gonna be imput afterwards with pipelines, the mediam value
corr = data.drop(['source','recruitment_channel','department','region'],axis=1).astype(float).corr() #saving the correlation for later use

ax = sns.set(rc={'figure.figsize':(10,4)})

sns.heatmap(corr, annot=True).set_title('Pearsons Correlation Factors Heat Map', color='black', size='25')
sns.set(rc={'figure.figsize':(10,10)}) #setting the size of the figure to make it easier to read.

sns.countplot(y=data["age"]).set_title("Age distribution", fontsize=15) #plotting it horizontally to make it easier to read
data.loc[data['age'] <= 26, 'age'] = 0

data.loc[(data['age'] > 26) & (data['age'] <= 35), 'age'] = 1

data.loc[(data['age'] > 35) & (data['age'] <= 50), 'age'] = 2

data.loc[data['age'] > 50, 'age'] = 3
sns.distplot(data["age"]).set_title("Distribution of age after mapping")
sns.set(rc={'figure.figsize':(10,10)}) #setting the size of the figure to make it easier to read.

sns.countplot(y=data["department"]).set_title("Departament distribution", fontsize=15) #plotting it horizontally to make it easier to read
data_by_sex = data.pivot_table(columns=["department"], index=["gender"],values=["length_of_service"],

                               fill_value=0,margins=True,dropna=True)

data_by_sex.plot(kind="bar", stacked=False, figsize=(20,10), title="lenght of service by department and sex",fontsize=15)
from sklearn.model_selection import train_test_split #importing the relevant module

#recreating the training set and testing set with all the modifications that has been made.

## before anything, however, I am gonna drop region and recruitment channel, as it is only make training slower

data.drop(['recruitment_channel','region'], axis=1, inplace=True)

train = data[data['source'] == 'train'].copy()

test = data[data['source'] == 'test'].copy()



#splitting the data and creating the labels

train_set, test_set = train_test_split(train, test_size=0.2, random_state=42)



train_labels = train_set["is_promoted"].copy() #creating our Y_train

train_set.drop(['source','is_promoted'], axis=1, inplace=True)



test_set_labels = test_set["is_promoted"].copy() #creating our Y_test

test_set.drop(['source','is_promoted'], axis=1, inplace=True)





test.drop(['source','is_promoted'], axis=1, inplace=True)
print(train.shape, train_set.shape, test_set.shape, test.shape) #printing the shape of the newly created training and testing set
#let's use the Imputer to fill the NAN values with the median value

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder



#numerical values

num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])



#categorical values and numerical values, gonna use OneHotEnconder for the categorical.

num_attribs = train_set.select_dtypes(exclude=['object']) #selecting all the numerical data 

cat_attribs = train_set.select_dtypes(exclude=['int64','float64']) #selecting non numerical data



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, list(num_attribs)),

        ("cat", OrdinalEncoder(), list(cat_attribs)),

        #("cat", OneHotEncoder(), list(cat_attribs)), #this is a better option for datasets where the ordinal enconder would not capture the correlation

        

    ]) 

from sklearn.metrics import mean_squared_error

from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.metrics import f1_score

import time #implementing in this function the time spent on training the model

from sklearn.model_selection import cross_val_score



#Generic function for making a classification model and accessing performance:

def classification_model(model, X_train, y_train, predict_only = False):

    #Fit the model:

    time_start = time.perf_counter() #start counting the time

    

    if not predict_only: #optional parameter of our function, whether we want our model to be trained, or if we had trained it previously and want to see predictions only

        model.fit(X_train,y_train)



    #predicting using the model that has been trained above

    train_predictions = model.predict(X_train)

    

    #measuring each different score

    accuracy = accuracy_score(y_train, train_predictions)

    precision = precision_score(y_train, train_predictions)

    recall = recall_score(y_train, train_predictions)

    f1 = f1_score(y_train, train_predictions)

    

    print("-Model results, in percentage-")

    print("Accuracy: %.2f%%" % (accuracy * 100))

    print("Precision: %.2f%%" % (precision * 100))

    print("Recall: %.2f%%" % (recall * 100))

    print("F1 score: %.2f%%" % (f1 * 100))

    cr_val_precision = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')

    cr_val_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    cr_val_recall = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')

    cr_val_roc_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    cr_val_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')

    

    time_end = time.perf_counter() #end of counting the time

    

    total_time = time_end-time_start #total time spent during training and cross_validation

    print("Cross Validation Score(scoring: precision): %.2f%%" % (np.mean(cr_val_precision) * 100))

    print("Cross Validation Score(scoring: accuracy): %.2f%%" % (np.mean(cr_val_accuracy) * 100))

    print("Cross Validation Score(scoring: recall): %.2f%%" % (np.mean(cr_val_recall) * 100))

    print("Cross Validation Score(scoring: roc_auc): %.2f%%" % (np.mean(cr_val_roc_auc) * 100))

    print("Cross Validation Score(scoring: f1): %.2f%%" % (np.mean(cr_val_f1) * 100))

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))
train_prepared = full_pipeline.fit_transform(train_set)
#ok, nice, now that we have our data prepared, let's start testing some models

from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver="liblinear")

classification_model(log_reg, train_prepared,train_labels)
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)



classification_model(sgd_clf, train_prepared, train_labels)
from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(n_estimators=200)

classification_model(rand_forest,train_prepared, train_labels) 
def plot_feature_importances(model):

    # Plot feature importance

    feature_importance = model.feature_importances_

    # make importances relative to max importance

    plt.figure(figsize=(20, 20))

    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, train_set.columns[sorted_idx], fontsize=15)

    plt.xlabel('Relative Importance', fontsize=30)

    plt.title('Variable Importance')
plot_feature_importances(rand_forest) #PLOTING FEATURES IMPORTANCES RELATIVE TO RANDOM FOREST
from sklearn.ensemble import GradientBoostingClassifier



gbc_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.09)

classification_model(gbc_model,train_prepared, train_labels)
plot_feature_importances(gbc_model)#PLOTING FEATURES IMPORTANCES RELATIVE TO GRADIENT BOOSTING
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



ada_clf = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=1), n_estimators=200,

    algorithm="SAMME.R", learning_rate=0.1, random_state=42)



classification_model(ada_clf,train_prepared, train_labels)
plot_feature_importances(ada_clf)#PLOTING FEATURES IMPORTANCES RELATIVE TO ADA BOOSTING
train_prepared_some_features = num_pipeline.fit_transform(train_set[['avg_training_score','length_of_service',

                                                                      'no_of_trainings','previous_year_rating',

                                                                      'KPI','awards_won','gender','education'

                                                                     ]])
import xgboost as xgb



xgb_model = xgb.XGBClassifier(learning_rate=0.09, n_estimators=1000, max_depth=4, min_child_weight=15, 

                      gamma=0.4,nthread=4, subsample=0.8, colsample_bytree=0.8, 

                        objective= 'binary:logistic',scale_pos_weight=4,seed=29)



classification_model(xgb_model,train_prepared_some_features,train_labels) #training with only some features selected
classification_model(xgb_model, train_prepared, train_labels) #training with all features
import lightgbm as lgb



lgb_model = lgb.LGBMClassifier(learning_rate=0.3, n_estimators=200, max_depth=4, min_child_weight=15, 

                      gamma=0.4,nthread=4, subsample=0.8, colsample_bytree=0.8, 

                        scale_pos_weight=4,seed=29) # same parameters as I used for XGBoost
classification_model(lgb_model, train_prepared, train_labels)
from sklearn.model_selection import GridSearchCV

params = {

  'min_child_weight':[6,10,15],

  #'max_depth': range(3,10,2),

  'n_estimators':[600,900,1000],

  'scale_pos_weight':[1,2,3,4],

    'learning_rate': [0.005,0.05,0.09,0.1]

  #'colsample_bytree':[0.7,0.8], 

  #'subsample':[0.7,0.8],

  #'gamma':[0,0.2.0.4]

    

}

grid_search = GridSearchCV(estimator = lgb_model,

param_grid = params, scoring='recall',n_jobs=4,iid=False, verbose=10, cv=5)
grid_search.fit(train_prepared,train_labels)
grid_search.best_estimator_
grid_search.best_params_
final_model_lgb = grid_search.best_estimator_ #creating a variable with our best model
classification_model(final_model_lgb, train_prepared, train_labels, predict_only=True)
test_val_prepared = full_pipeline.fit_transform(test_set)

classification_model(final_model_lgb, test_val_prepared, test_set_labels, predict_only=True)
classification_model(xgb_model, test_val_prepared, test_set_labels, predict_only=True)