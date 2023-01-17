# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Download the data from repository for EDA

df1=pd.read_csv('/kaggle/input/titanic/train.csv')



df3=pd.read_csv('/kaggle/input/titanic/test.csv')

# Get frist hand knowledge about the data

#profile = df1.profile_report(title='Pandas Profiling Report')

#profile.to_file(output_file="train_pandas_profiling.html")

#profile 
df1.head(5)


    ## Columns with numerical values are PassengerId,Survived(0 or 1),Pclass(1,2 or 3),Age(.42 to 80 yrs),SibSp(0 to 8),Parch,Fare(0 to 512) 

df1.columns.values
df1.info()
df1.describe()
df1[['Cabin']].groupby('Cabin', as_index= False).sum().sort_values(by= 'Cabin', ascending= False)
#Adding a new Column 'has_cabin'

df1['has_cabin']=df1['Cabin'].apply(lambda x:0 if type(x)==float else 1)

df1= df1.drop('Cabin', axis=1)

df1.head()
#Adding a new column as 'has_family'by combining SibSp and Parch

df1['has_family']= df1['SibSp']+df1['Parch']+1

df1['has_family']=df1['has_family'].apply(lambda x:0 if x==1 else 1)

df1=df1.drop(['SibSp', 'Parch'], axis=1)

df1.head()
#Lets treat the missing values in Age column by imputing with mean value

mean_1=df1['Age'].mean()

mean_1
df1['Age']= df1['Age'].fillna(mean_1)
#Adding a new column with age group

def age_group(i):

    if (i <= 1):

        return 'Infant'

    elif (i>1) & (i <=3):

        return 'Toddler'

    elif (i>3) & (i<=12):

        return 'Kid'

    elif (i>12) & (i<=18):

        return'Teen'

    elif (i>18) & (i<= 60):

        return 'Adult'

    else:

        return'Old'

df1['age_group']=df1.apply(lambda x: age_group(x['Age']),axis=1)

df1=df1.drop('Age', axis=1)

df1.head()
df1.info()
#df1 = df1.dropna(subset=['Age','Embarked'])

#df1 = df1.dropna(subset=['Embarked'])

df1.groupby('Embarked').count()
# as Emabrked S has maximum values, lets impute Null values with S

df1['Embarked']=df1['Embarked'].fillna('S')
# Lets generalise Fare column

def fare_group(i):

    if i <= 8:

        return 'group1'

    elif (i>8) & (i <=24):

        return 'group2'

    elif (i>24) & (i <=50):

        return 'group3'

    elif (i>50) & (i<=100):

        return 'group4'

    else:

        return 'group5'

df1['fare_group']= df1.apply(lambda x:fare_group(x['Fare']), axis=1)

df1=df1.drop('Fare', axis=1)

df1.info()

    
"""""total = df1.isnull().sum().sort_values(ascending=False)

percent = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Missing Percent'])

missing_data['Missing Percent'] = missing_data['Missing Percent'].apply(lambda x: x * 100)

missing_data.loc[missing_data['Missing Percent'] > 0][:10]"""""

#The above data shows, there are about 6 object type variables in df1, need to assign numbers prior to our anlaysis

from sklearn import preprocessing

df1_catagorical= df1.select_dtypes(include=['object'])

df1_catagorical

# assigning numbers to object columns entries-apply label encoder to df1_catagorical

le = preprocessing.LabelEncoder()

df1_catagorical = df1_catagorical.apply(le.fit_transform)

df1_catagorical.head()

#dropping the catagorical columns and adding the new Catagorical columns

df1= df1.drop(df1_catagorical.columns, axis =1)

df1=pd.concat([df1, df1_catagorical], axis=1)

df1.info()
df1.sample(5)
df1=df1.drop('Name', axis=1)

df1=df1.drop('Ticket', axis=1)
X= df1.drop('Survived',axis=1)

y= df1['Survived'].astype('category')
#Lets check the Testing data

df3.info()
#Impute missing Age values with mean

mean_a=df3['Age'].mean()

mean_a
df3['Age']=df3['Age'].fillna(mean_a)

df3.info()

#lets add a column with Cabin info

df3['has_cabin']=df3['Cabin'].apply(lambda x:0 if type(x)==float else 1)

df3= df3.drop('Cabin', axis=1)

df3.head()
#Adding a new column as 'has_family'by combining SibSp and Parch

df3['has_family']= df3['SibSp']+df3['Parch']+1

df3['has_family']=df3['has_family'].apply(lambda x:0 if x==1 else 1)

df3=df3.drop(['SibSp', 'Parch'], axis=1)

df3.head()
#dding a new column for age group

df3['age_group']=df3.apply(lambda x: age_group(x['Age']),axis=1)

df3=df3.drop('Age', axis=1)

df3.head()
df3.info()
#Lets choose 'Fare' column first to impute the value

df3[df3['Fare'].isnull()] 
mean_b=df3['Fare'].mean()

mean_b
#impute missing Fare values with mean_b

df3['Fare']=df3['Fare'].fillna(mean_b)
df3.info()
df3['fare_group']= df3.apply(lambda x:fare_group(x['Fare']), axis=1)

df3=df3.drop('Fare', axis=1)

df3.info()
#df3 has 6 object type coulmns , lets assighn numbers wrt labels

df3_catagorical= df3.select_dtypes(include=['object'])

df3_catagorical


df3_catagorical = df3_catagorical.apply(le.fit_transform)

df3_catagorical.head()
#dropping the catagorical columns and adding the new Catagorical columns

df3= df3.drop(df1_catagorical.columns, axis =1)

df3=pd.concat([df3, df3_catagorical], axis=1)

df3.info()
df3=df3.drop('Name', axis=1)

df3=df3.drop('Ticket', axis=1)
df3
X_test_final=df3
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=.3, random_state=50)

dt= RandomForestClassifier()

dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

y_pred

y_pred.size
#converting y_pred to data frame

y_pred= pd.DataFrame(y_pred)

y_pred
X_test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
# Printing confusion matrix

confusion_matrix(y_test, y_pred)
# Printing accuracy_score

accuracy_score(y_test, y_pred)
# GridSearchCV to find optimal tuning gdata

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
param_grid = {

    'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]

}



n_folds = 5



# Instantiate the grid search model

rfc = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 

                          cv = n_folds, verbose = 1)



# Fit the grid search to the data

grid_search.fit(X_train,y_train)
# printing the optimal accuracy score and hyperparameters

print("best accuracy", grid_search.best_score_)

print(grid_search.best_estimator_)
# model with mdified optimal hyperparameters

clf_gini = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=10, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=50, min_samples_split=50,

                       min_weight_fraction_leaf=0.0, n_estimators=10,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

clf_gini.fit(X_train, y_train)
#Lets do the predictions after tuning the Hyper parameters

y_pred_new=clf_gini.predict(X_test)
# Printing accuracy_score prior to hyper parameter Tuning

accuracy_score(y_test, y_pred)
# Printing accuracy_score after Hyper parameter tuning

accuracy_score(y_test, y_pred_new)
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred_new))
# Printing confusion matrix Prior to Hyper parameter Tuning

confusion_matrix(y_test, y_pred)
# Printing confusion matrix post Hyper parameter Tuning

confusion_matrix(y_test, y_pred_new)
import sklearn

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics



import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

import gc # for deleting unused variables

%matplotlib inline



import os

import warnings

warnings.filterwarnings('ignore')



model= XGBClassifier()

model.fit(X_train,y_train)
# make predictions for test data

# use predict_proba since we need probabilities to compute auc

y_pred = model.predict_proba(X_test)

y_pred[:10]
# evaluate predictions

roc = metrics.roc_auc_score(y_test, y_pred[:, 1])

print("AUC: %.2f%%" % (roc * 100.0))
# hyperparameter tuning with XGBoost



# creating a KFold object 

folds = 3



# specify range of hyperparameters

param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]}          





# specify model

xgb_model = XGBClassifier(classifier_max_depth=2, n_estimators=100)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring= 'roc_auc', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)
# fit the model

model_cv.fit(X_train, y_train)
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# convert parameters to int for plotting on x-axis

cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')

#cv_results['param_max_depth'] = cv_results['param_max_depth'].astype('float')

cv_results.head()
# # plotting

plt.figure(figsize=(16,6))



param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]} 





for n, subsample in enumerate(param_grid['subsample']):

    



    # subplot 1/n

    plt.subplot(1,len(param_grid['subsample']), n+1)

    df = cv_results[cv_results['param_subsample']==subsample]



    plt.plot(df["param_learning_rate"], df["mean_test_score"])

    plt.plot(df["param_learning_rate"], df["mean_train_score"])

    plt.xlabel('learning_rate')

    plt.ylabel('AUC')

    plt.title("subsample={0}".format(subsample))

    plt.ylim([0.60, 1])

    plt.legend(['test score', 'train score'], loc='upper left')

    plt.xscale('log')
params = {'learning_rate': 0.2,

          'max_depth': 2, 

          'n_estimators':100,

          'subsample':0.9,

         'objective':'binary:logistic'}



# fit model on training data

model = XGBClassifier(params = params)

model.fit(X_train, y_train)
# predict

y_pred_boost = model.predict_proba(X_test)

y_pred_boost[:10]
auc = sklearn.metrics.roc_auc_score(y_test, y_pred[:, 1])

auc
#converting y_pred_boost to data frame

y_pred_boost= pd.DataFrame(y_pred)
#lets convert y_pred_new to data frame



y_pred_new=pd.DataFrame(y_pred_new)

y_pred_final=model.predict(X_test_final)

y_pred_final=pd.DataFrame(y_pred_final)

y_pred_final.info()
X_test_final.info()
# adding Test & Predicted data into a single dataframe

output=X_test_final.join(y_pred_final)

output

output.columns


output.columns.values[8] = 'Survived'

output = output.reset_index()

output.head()
predictions=output[['PassengerId','Survived']]

predictions
predictions.to_csv('predictions.csv', index= False)
feature_importances = pd.DataFrame(model.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances # Used initial model for imprtance feature calculations
output[['Sex', 'Survived']].groupby('Sex', as_index= False).sum().sort_values(by= 'Survived', ascending= False)
