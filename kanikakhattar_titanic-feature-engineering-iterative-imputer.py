# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
passengerId = test_data.PassengerId
!pip install fancyimpute
data.info()
data.shape
round(100*(data.isnull().sum()/len(data.index)),2)
round(100*(test_data.isnull().sum()/len(test_data.index)),2)
#checking Cabin
data[['Cabin']].head(20)
#Lets split the column into Cabin Letter and Cabin number. As there are large number of missing values, instead of imputing we will replace the missing values with letter 'M'
data[['cabin_letter', 'cabin_number']] = data['Cabin'].str.extract('([A-Z])(\d*)', expand = True)
test_data[['cabin_letter', 'cabin_number']] = test_data['Cabin'].str.extract('([A-Z])(\d*)', expand = True)

data.drop(['Cabin','cabin_number'],axis=1,inplace=True)
test_data.drop(['Cabin','cabin_number'],axis=1,inplace=True)
data['cabin_letter'].value_counts()
pd.crosstab(data.Survived,data.cabin_letter,normalize=True)
data['cabin_letter'] = data['cabin_letter'].fillna('M')
test_data['cabin_letter'] = data['cabin_letter'].fillna('M')

round(100*(data.isnull().sum()/len(data.index)),2)
round(100*(test_data.isnull().sum()/len(test_data.index)),2)
#checking Embarked
data['Embarked'].value_counts(dropna=False) 
test_data['Embarked'].value_counts(dropna=False) 
data.loc[data['Embarked'].isnull()]
data.groupby(by=['Embarked','Pclass','cabin_letter'])['Fare'].describe()
data['Embarked'] = data['Embarked'].fillna('C')
round(100*(data.isnull().sum()/len(data.index)),2)
#data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
round(100*(test_data.isnull().sum()/len(test_data.index)),2)
#Function to check total counts, skewness and importance of a categorical column based on conversion rate. 
def check_count_conversion_rate(data,X,target):
    #checking counts of col
    col_counts = pd.DataFrame(data[X].value_counts()).reset_index()
    col_counts.columns = [X,'Counts']
    col_counts['Total%'] = col_counts['Counts']/len(data.index)
    #checking conversion rate by col
    groupby_col = pd.DataFrame(data.groupby(X)[target].mean()).reset_index()

    col_counts_percentage = col_counts.merge(groupby_col,how='inner',on=X)
    return col_counts_percentage

# Checking Sex column 
pd.crosstab(data.Survived,data.Sex,normalize = True)
check_count_conversion_rate(data,'Sex','Survived')
#Checking distribution of PClass 
sns.countplot("Survived",data=data,hue='Pclass')
check_count_conversion_rate(data,'Pclass','Survived')
#Checking Age
data.loc[data['Survived']==0]['Age'].hist(color='green')
data.loc[data['Survived']==1]['Age'].hist(color='blue')

sns.boxplot(y='Age',x='Survived',data=data)
#Checking Age
data.loc[data['Survived']==0]['Fare'].hist(color='green',bins=50)
data.loc[data['Survived']==1]['Fare'].hist(color='blue',bins=50)

sns.boxplot(x='Survived',y='Fare',data=data)
sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=data)
data['Parch'].value_counts(dropna=False)
check_count_conversion_rate(data,'Parch','Survived')
test_data['Parch'].value_counts(dropna=False)
sns.heatmap(data.corr(),annot=True)
data.head()
#Extracting titles from name
data['Title'] = [i.split('.')[0] for i in data['Name']]
data['Title'] = [i.split(',')[1] for i in data['Title']]

test_data['Title'] = [i.split('.')[0] for i in test_data['Name']]
test_data['Title'] = [i.split(',')[1] for i in test_data['Title']]

data['Title'].value_counts(dropna = False)
data['Title'].unique()
data.Title = data.Title.apply(lambda x:x.strip())
test_data.Title = test_data.Title.apply(lambda x:x.strip())
data.Title.replace(('Ms','Mlle','Mme'),'Miss',inplace=True)
data.Title.replace(('Rev','Don','Dr','Major','Lady','Sir','Col','Capt','the Countess','Jonkheer'),'Royalty',inplace=True)

test_data.Title.replace(('Ms','Mlle','Mme'),'Miss',inplace=True)
test_data.Title.replace(('Rev','Don','Dr','Major','Lady','Sir','Col','Capt','the Countess','Jonkheer','Dona'),'Royalty',inplace=True)

data.Title.value_counts(dropna=False)
test_data.Title.value_counts(dropna=False)
check_count_conversion_rate(data,'Title','Survived')
data.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
test_data.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
#Imputing age and Fare
data.head()
test_data.head()
#Checking Age - We will impute Age by using Iterative Imputer
#Creating dummies
data.Sex.replace(['female','male'],[0,1],inplace=True)
test_data.Sex.replace(['female','male'],[0,1],inplace=True)
data.head()
# For pd.get_dummies: the data should be object data type
# Create a subset of categorical data
data_cat = data[['Pclass','SibSp','Embarked','cabin_letter','Title']]
test_data_cat = test_data[['Pclass','SibSp','Embarked','cabin_letter','Title']]
data_cat.head()
data_cat.info()
#There are still categorical columns which are of type int. Convert them to object data datatype
for i in ['Pclass','SibSp','Embarked','cabin_letter','Title']:
  data_cat[i] = data_cat[i].astype(str)
  test_data_cat[i] = test_data_cat[i].astype(str)
test_data_cat.info()
#Now we can create dummies
data_cat_dummy1 = pd.get_dummies(data_cat[['Pclass','SibSp','Embarked','Title']],drop_first=True)
data_cat_dummy2 = pd.get_dummies(data_cat['cabin_letter'])

test_data_cat_dummy1 = pd.get_dummies(test_data_cat[['Pclass','SibSp','Embarked','Title']],drop_first=True)
test_data_cat_dummy2 = pd.get_dummies(test_data_cat['cabin_letter'])
#Dropping value 'M' (Missing) as this was the value which we replace the NaN 
data_cat_dummy2.drop(['M'],axis=1,inplace=True)
test_data_cat_dummy2.drop(['M'],axis=1,inplace=True)
data.drop(['Pclass','SibSp','Embarked','cabin_letter','Title'],axis=1,inplace=True)
test_data.drop(['Pclass','SibSp','Embarked','cabin_letter','Title'],axis=1,inplace=True)
data.head()
data = pd.concat([data,data_cat_dummy1,data_cat_dummy2],axis=1)
test_data = pd.concat([test_data,test_data_cat_dummy1,test_data_cat_dummy2],axis=1)

print(data.shape)
print(test_data.shape)

from fancyimpute import IterativeImputer
#Using Iterative Imputer to impute missing value of Age

#First let us store the column names
data_ii  = data.drop(['Survived'],axis=1)
data_cols = data_ii.columns

ii = IterativeImputer()
data_clean = pd.DataFrame(ii.fit_transform(data.drop(['Survived'],axis=1)))
test_data_clean = pd.DataFrame(ii.transform(test_data))
# the output is the numpy array.
# ii looks for all the columns which have missing values and then imputes them
# need to cross check the age imputation as ii can assign values from -inf to +inf
data_clean.columns=data_cols
test_data_clean.columns = data_cols
data_clean.head()
data.shape
#round(100*(data_clean.isnull().sum()/len(data_clean)),2)
round(100*(test_data_clean.isnull().sum()/len(test_data_clean)),2)
#cross checking age
sns.boxplot(y=test_data_clean['Age'])
sns.boxplot(y=test_data['Age'])
#to convert continuous variables to categorical using pd.cut function
#data_clean['ageGroup'] = pd.cut(data_clean['Age'],bins = [0,16,32,48,64,300],labels=[0,1,2,3,4])
#test_data_clean['ageGroup'] = pd.cut(test_data_clean['Age'],bins = [0,16,32,48,64,300],labels=[0,1,2,3,4])


#data_clean.loc[data_clean['ageGroup'].isnull()]
#data_clean.drop(['Age'],axis=1,inplace=True)
#test_data_clean.drop(['Age'],axis=1,inplace=True)
data_clean.columns
#checking for outlier
sns.boxplot(y=data['Fare'])
# There is only one data point that lies far apart. It is better to delete that one data point and keep the others as it is.
# the other outliers are actually depicting pattern in the data. It is better not to cap it. Also the outliers are more in number.
# Any change in the outlier will change the pattern and meaning of the data.
data_clean.drop(data_clean.index[data_clean['Fare']>300],inplace=True)
data.drop(data.index[data['Fare']>300],inplace=True)
data_clean.shape
data.shape
#creating dummies for AgeGroup now
#data_cat_dummy = pd.get_dummies(data_clean['ageGroup'],drop_first=True)
#test_data_clean_cat_dummy = pd.get_dummies(test_data_clean['ageGroup'],drop_first=True)

#data_clean = pd.concat([data_clean,data_cat_dummy],axis=1)
#test_data_clean = pd.concat([test_data_clean,test_data_clean_cat_dummy],axis=1)
data_clean.isnull().sum()
#diving data in train and test
X = data_clean
y = data['Survived']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100,stratify=y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#checking data imbalance
100 * y_train.value_counts()/len(y_train)
# Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
print("Accuracy: {}".format(metrics.accuracy_score(y_test,y_pred)))
print("Recall: {}".format(metrics.recall_score(y_test,y_pred)))

# Logistic Regression model with class_weight
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(class_weight='balanced')
#logreg = LogisticRegression(class_weight={0:38.164251,1:61.835749})
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
print("Accuracy: {}".format(metrics.accuracy_score(y_test,y_pred)))
print("Recall: {}".format(metrics.recall_score(y_test,y_pred)))

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)
print("Accuracy: {}".format(metrics.accuracy_score(y_test,y_pred)))
print("Recall: {}".format(metrics.recall_score(y_test,y_pred)))


#hyper parameter tuning
# Create the parameter grid based on the results of random search 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits=3, random_state=2)
param_grid = {
    'max_depth': [4,5,10],
    'min_samples_leaf': [5, 6,7],
    'min_samples_split': [5, 10, 16],
    'n_estimators': [100,500,700], 
    'max_features': [5,10,15]
}
# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = cv, n_jobs = -1,verbose = 1)

grid_search.fit(X_train, y_train)
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
#using best parameters after tuning
rf_model = RandomForestClassifier(class_weight='balanced',
                                  criterion='gini',
                                  min_samples_leaf =5,
                                  min_samples_split= 5,
                                  n_estimators=100,
                                  max_features=10,
                                  max_depth=4)
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
print("Accuracy: {}".format(metrics.accuracy_score(y_test,y_pred)))
print("Recall: {}".format(metrics.recall_score(y_test,y_pred)))
#Cheking feature importance
pd.concat([pd.DataFrame(X.columns,columns = ['variables']),
           pd.DataFrame(rf_model.feature_importances_,columns = ['importance'])],
          axis=1).sort_values(by='importance',ascending=False)

test_data_clean.isnull().sum()
#scaling test data
test_data_scaled = scaler.transform(test_data_clean)
test_pred = rf_model.predict(test_data_scaled)
submission = pd.DataFrame({
        "PassengerId": passengerId,
        "Survived": test_pred
    })
submission['Survived'] = submission['Survived'].astype('int')
#submission.to_csv('gender_submission.csv', index=False)
submission.head()
submission.to_csv('gender_submission_results1.csv', index=False)
#Score in Kaggle: 0.7846
