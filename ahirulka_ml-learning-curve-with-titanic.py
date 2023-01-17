# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np #NumPy is the fundamental package for scientific computing with Python.
import pandas as pd #library written for the Python for data manipulation and analysis. In particular, 
                    #it offers data structures and operations for manipulating numerical tables and time series .
import matplotlib.pyplot as plt #plotting library  and numerical mathematics extends NumPy
import seaborn as sns  #Python data visualization library based on matplotlib.

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
Train_Master = pd.read_csv('../input/train.csv')
Test_Master = pd.read_csv('../input/test.csv')
print('*** About Training Data ---> \n')
print('Dimensions: {} Rows and {} Columns \n'.format(Train_Master.shape[0],Train_Master.shape[1]))
print('Column names are -> \n{}\n'.format(Train_Master.columns.values))
Train_Master.info()
print('\n\n\n')
print('*** About Test Data ---> \n')
print('Dimensions: {} Rows and {} Columns \n'.format(Test_Master.shape[0],Test_Master.shape[1]))
print('Column names are -> \n{}\n'.format(Test_Master.columns.values))
Test_Master.info()
Train_Master.describe()
Train_Master.isnull().sum(axis=0)
#Lets drop column "Cabin" from training set.
Train_Master.drop('Cabin', axis=1, inplace=True)
#Lets fill null values of Age by mean of column values
Train_Master['Age'] = Train_Master['Age'].fillna(np.mean(Train_Master['Age']))
# Still we have two null values in column Embarked.
print(Train_Master['Embarked'].describe())
print(Train_Master['Embarked'].mode())
# Replacing null value with most ocurred value seems proper solution.
Train_Master['Embarked'] = Train_Master['Embarked'].fillna('S')
Train_Master.isnull().sum(axis=0)  # Re-validate Null value count
Train_Master.columns.values
Features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
y = Train_Master.Survived
X = Train_Master[Features]
y.head()
X.head()
X.describe()
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state =1,test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
ML_101_DTC = DecisionTreeClassifier(random_state=1)
ML_101_DTC.fit(train_X, train_y)
pred_ML_101_DTC = ML_101_DTC.predict(val_X)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(val_y,pred_ML_101_DTC))
print(classification_report(val_y, pred_ML_101_DTC))
print(accuracy_score(val_y, pred_ML_101_DTC))
from sklearn.ensemble import RandomForestClassifier
ML_101_RFC = RandomForestClassifier(random_state=1)
ML_101_RFC.fit(train_X, train_y)
pred_ML_101_RFC = ML_101_RFC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_101_RFC))
print(classification_report(val_y, pred_ML_101_RFC))
print(accuracy_score(val_y, pred_ML_101_RFC))
#For test data test_X will be Test_Master[Features]
#But Null values will give error for predict function.

print(Test_Master[Features].isnull().sum())
# Fix Age as we did for Training Set and Fare to be filled with median of Fare
Test_Master['Age'] = Test_Master['Age'].fillna(np.mean(Test_Master['Age']))
Test_Master['Fare'] = Test_Master['Fare'].fillna(Test_Master['Fare'].median())

pred_ML_101_sub = ML_101_RFC.predict(Test_Master[Features])


#test_ids = Test_Master.loc[:, 'PassengerId']
#my_submission = pd.DataFrame(data={'PassengerId':test_ids, 'Survived':pred_ML_101_sub})

#print(my_submission['Survived'].value_counts())

#Export Results to CSV
# my_submission.to_csv('submission.csv', index = False)   ---Exporting csv from Next Section
Train_Master = pd.read_csv('../input/train.csv')
Test_Master = pd.read_csv('../input/test.csv')
Train_Master.sample(5)
print(Train_Master.info())
print('*-*'*25)
print(Train_Master.describe())
fig, ax  = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(12)
sns.heatmap(data=Train_Master.isnull(),cbar=False,yticklabels=False,cmap='cividis', ax=ax[0])
ax[0].set_title('Data Missingness for Training Data',fontsize=16)
sns.heatmap(data=Test_Master.isnull(),cbar=False,yticklabels=False, cmap='cividis',ax=ax[1])
ax[1].set_title('Data Missingness for Test Data',fontsize=16)
plt.show()
# Age
print('Total No of rows in training set: {} \n'.format(len(Train_Master)))
print('Total No of  rows  in  test  set: {}\n \n '.format(len(Test_Master)))

per_nan_age_train = ((Train_Master['Age'].isnull().sum())*100)/(len(Train_Master))
per_nan_age_test  = ((Test_Master['Age'].isnull().sum())*100)/(len(Test_Master))

print('% of missing rows in training set: {0:0.2f} \n'.format(per_nan_age_train))
print('% of missing  rows  in  test  set: {0:0.2f} \n'.format(per_nan_age_test))
Titanic_Master = Train_Master.append(Test_Master)
age_data = Train_Master['Age'].dropna().append(Test_Master['Age'].dropna())
sns.distplot(age_data,bins = 50,color='blue')
plt.show()
Train_Master.corr()
sns.heatmap(Train_Master.corr(), cmap='rainbow', annot=True,cbar=False,linewidths=1, linecolor='black')
plt.show()
sns.boxplot(x='Pclass',y='Age',data=Train_Master)
sns.scatterplot(x='Fare',y='Age',data=Train_Master,hue='Pclass',palette='rainbow')
plt.show()
fig, ax  = plt.subplots(1,4)
fig.set_figheight(8)
fig.set_figwidth(20)
sns.scatterplot(y='Age',x='Pclass',data=Train_Master,ax=ax[0])
sns.scatterplot(y='Age',x='Fare',data=Train_Master,ax=ax[1])
sns.scatterplot(y='Age',x='SibSp',data=Train_Master,ax=ax[2])
sns.scatterplot(y='Age',x='Parch',data=Train_Master,ax=ax[3])
plt.show()
Titanic_Master.groupby(['Pclass'])['Age'].median()
Titanic_Master[Titanic_Master.Pclass==1]['Age'].median()
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return Titanic_Master[Titanic_Master.Pclass==1]['Age'].median()
        elif Pclass ==2:
            return Titanic_Master[Titanic_Master.Pclass==2]['Age'].median()
        elif Pclass ==3:
            return Titanic_Master[Titanic_Master.Pclass==3]['Age'].median()
    else:
        return Age
Train_Master['Age'] = Train_Master[['Age','Pclass']].apply(impute_age,axis=1)
Test_Master['Age'] = Test_Master[['Age','Pclass']].apply(impute_age,axis=1)
print(Train_Master['Embarked'].isnull().sum())
sns.countplot(data=Train_Master, x='Embarked')
plt.show()
# Fill value of most occurred value

Train_Master['Embarked'] = Train_Master['Embarked'].fillna('S')
Test_Master[Test_Master['Fare'].isnull()]
print('Min. of Fare: {}'.format(Titanic_Master.Fare.min()))
print('Max. of Fare: {}'.format(Titanic_Master.Fare.max()))
print('Average Fare: {}'.format(Titanic_Master.Fare.mean()))
Train_Master[Train_Master.Fare==0]
sns.boxplot(x='Pclass', y='Fare', data=Titanic_Master)
print('Mean of fare for Class-3:',end ='   ')
print(Titanic_Master[Titanic_Master.Pclass==3].Fare.mean())
print('Median of fare for whole dataset:',end ='   ')
print(Titanic_Master.Fare.median())
print('Mean Fare values after removing outliers:',end='   ')
print(Titanic_Master[Titanic_Master.Fare<500]['Fare'].mean())
Test_Master['Fare'] = Test_Master['Fare'].fillna(Titanic_Master.Fare.median())
# Age
print('Total No of rows in training set: {} \n'.format(len(Train_Master)))
print('Total No of  rows  in  test  set: {}\n \n '.format(len(Test_Master)))
print('-'*100)
per_nan_cabin_train = ((Train_Master['Cabin'].isnull().sum())*100)/(len(Train_Master))
per_nan_cabin_test  = ((Test_Master['Cabin'].isnull().sum())*100)/(len(Test_Master))

print('% of missing rows in training set: {0:0.2f} \n'.format(per_nan_cabin_train))
print('% of missing  rows  in  test  set: {0:0.2f} \n'.format(per_nan_cabin_test))
Train_Master.drop('Cabin', axis=1, inplace=True)
Test_Master.drop('Cabin', axis=1, inplace=True)
fig, ax  = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(12)
sns.heatmap(data=Train_Master.isnull(),cbar=False,yticklabels=False,cmap='cividis', ax=ax[0])
ax[0].set_title('Data Missingness for Training Data',fontsize=16)
sns.heatmap(data=Test_Master.isnull(),cbar=False,yticklabels=False, cmap='cividis',ax=ax[1])
ax[1].set_title('Data Missingness for Test Data',fontsize=16)
plt.show()
fig, ax = plt.subplots(1,2)
fig.set_figwidth(12)

sns.countplot(x='Pclass',data=Train_Master,ax=ax[0])
sns.countplot(x='Pclass',data=Train_Master, hue='Survived',palette=('red','green'),ax=ax[1])
ax[0].set_title('Passenger Count by Pclass')
ax[1].set_title('Survival Distribution of by Pclass')

sns.factorplot(x='Pclass',y='Survived',hue='Sex',col='Embarked',data=Train_Master)
plt.show()
fig, ax = plt.subplots(1,2)
fig.set_figwidth(12)
sns.countplot(x='Sex',data=Train_Master, hue='Survived',palette=('red','green'),ax=ax[1])
ax[1].set_title('Survival Distribution of Male and Female')

sns.countplot(x='Sex',data=Train_Master,ax=ax[0])
ax[0].set_title('Total Number of Male and Female of Titanic')

plt.show()
fig, ax = plt.subplots(1,2)
fig.set_figwidth(12)
sns.countplot(x='Embarked', hue='Survived',data=Train_Master,palette=('red','green'),ax=ax[1])
ax[0].set_title('Total No Embarked Passengers from given Location')
sns.countplot(x='Embarked',data=Train_Master,ax=ax[0])
ax[1].set_title('Survival Distribution as per Embarked Status')
plt.show()
Train_Master = pd.get_dummies(Train_Master, columns=['Sex','Pclass', 'Embarked'], drop_first=True)
Train_Master.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

test_ids = Test_Master.loc[:, 'PassengerId']
Test_Master = pd.get_dummies(Test_Master, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
Test_Master.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
Train_Master.head()
Train_Master['FamilySize'] = Train_Master['SibSp'] + Train_Master['Parch']
Test_Master['FamilySize'] = Test_Master['SibSp'] + Test_Master['Parch']
Train_Master.drop(columns=['SibSp','Parch'], axis=1, inplace =True)
Test_Master.drop(columns=['SibSp','Parch'], axis=1, inplace =True)
y = Train_Master.Survived
X = Train_Master.drop('Survived', axis=1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state =1,test_size=0.30)
ML_102_DTC = DecisionTreeClassifier(random_state=1)
ML_102_DTC.fit(train_X, train_y)
pred_ML_102_DTC = ML_102_DTC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_102_DTC))
print(classification_report(val_y, pred_ML_102_DTC))
print(accuracy_score(val_y, pred_ML_102_DTC))
ML_102_RFC = RandomForestClassifier(random_state=1,n_estimators =300)
ML_102_RFC.fit(train_X, train_y)
pred_ML_102_RFC = ML_102_RFC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_102_RFC))
print(classification_report(val_y, pred_ML_102_RFC))
print(accuracy_score(val_y, pred_ML_102_RFC))
from sklearn.neighbors import KNeighborsClassifier
ML_102_KNN = KNeighborsClassifier(n_neighbors = 3)
ML_102_KNN.fit(train_X, train_y)
pred_ML_102_KNN = ML_102_KNN.predict(val_X)

print(confusion_matrix(val_y,pred_ML_102_KNN))
print(classification_report(val_y, pred_ML_102_KNN))
print(accuracy_score(val_y, pred_ML_102_KNN))
ML_102_SVC = SVC(probability=True,random_state=1)
ML_102_SVC.fit(train_X, train_y)
pred_ML_102_SVC = ML_102_SVC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_102_SVC))
print(classification_report(val_y, pred_ML_102_SVC))
print(accuracy_score(val_y, pred_ML_102_SVC))
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000,10000],'gamma':[1,0.1,0.01,0.001,0.0001,0.00001]}
grid = GridSearchCV(SVC(), param_grid,verbose=3)
grid.fit(train_X,train_y)
grid.best_params_
grid.best_estimator_
pred_ML_102_SVC_grid = grid.predict(val_X)
print(confusion_matrix(val_y,pred_ML_102_SVC_grid))
print(classification_report(val_y, pred_ML_102_SVC_grid))
print(accuracy_score(val_y, pred_ML_102_SVC_grid))
## Submitting Results of RFC....

#pred_ML_102_sub = grid.predict(Test_Master)

#my_submission = pd.DataFrame(data={'PassengerId':test_ids, 'Survived':pred_ML_102_sub})

#print(my_submission['Survived'].value_counts())

#Export Results to CSV
#my_submission.to_csv('submission.csv', index = False)
#Import Data
Train_Master = pd.read_csv('../input/train.csv')
Test_Master = pd.read_csv('../input/test.csv')

#Creating temperory dataset by appending Test set to Training Set
#https://pandas.pydata.org/pandas-docs/stable/merging.html
Titanic_Master = Train_Master.append(Test_Master)

#Impute Age - helper function
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return Titanic_Master[Titanic_Master.Pclass==1]['Age'].median()
        elif Pclass ==2:
            return Titanic_Master[Titanic_Master.Pclass==2]['Age'].median()
        elif Pclass ==3:
            return Titanic_Master[Titanic_Master.Pclass==3]['Age'].median()
    else:
        return Age
#Imputing Age
#Train_Master['Age'] = Train_Master[['Age','Pclass']].apply(impute_age,axis=1)
#Test_Master['Age'] = Test_Master[['Age','Pclass']].apply(impute_age,axis=1)
# commenting code for Age imputation, since later stages it was found that Age can be well estimated from Title of Person.
#Impute Embark
Train_Master['Embarked'] = Train_Master['Embarked'].fillna('S')
#Impute Fare
Test_Master['Fare'] = Test_Master['Fare'].fillna(Titanic_Master.Fare.median())
fig, ax  = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(12)
sns.heatmap(data=Train_Master.isnull(),cbar=False,yticklabels=False,cmap='cividis', ax=ax[0])
ax[0].set_title('Data Missingness for Training Data',fontsize=16)
sns.heatmap(data=Test_Master.isnull(),cbar=False,yticklabels=False, cmap='cividis',ax=ax[1])
ax[1].set_title('Data Missingness for Test Data',fontsize=16)
plt.show()
Train_Master['Name'].sample(3)
#How to split name?
'Strange, Dr. Steven (Doctor Strage)'.split(sep=',')[1].split(sep='.')[0]
#However Train_Master['Name'] is series. Working text split for Series
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.split.html

#Train_Master['Name'].str.split(pat=",",expand=True)[1].str.split(pat='.',expand=True)[0]
#Lets store title in new column called 'Title'

Train_Master['Title'] =Train_Master['Name'].str.split(pat=",",expand=True)[1].str.split(pat='.',expand=True)[0].str.strip()
Test_Master['Title'] =Test_Master['Name'].str.split(pat=",",expand=True)[1].str.split(pat='.',expand=True)[0].str.strip()
Titanic_Master['Title'] =Titanic_Master['Name'].str.split(pat=",",expand=True)[1].str.split(pat='.',expand=True)[0].str.strip()
Titanic_Master.head()
#Objective is to find out if there is relation between Title vs "Social Status & Age of Person".
fig, ax  = plt.subplots(2,1)
fig.set_figheight(8)
fig.set_figwidth(15)
fig.tight_layout()
sns.countplot(x='Title',data=Titanic_Master,ax=ax[0])
sns.boxplot(x='Title',y ='Age', data=Titanic_Master,ax=ax[1])
plt.show()
print(Titanic_Master.Title.value_counts())
print(Titanic_Master.Title.unique())
#Lets create mapping for the titles 
Title_dict = {
  'Mr': 'Mr',
  'Mrs': 'Mrs',
  'Miss': 'Miss',
  'Master': 'Master',
  'Don': 'Others_M',
  'Rev': 'Rev',
  'Dr': 'Dr',
  'Mme': 'Others_F',
  'Ms': 'Others_F',
  'Major': 'Others_M',
  'Lady': 'Others_F',
  'Sir': 'Others_M',
  'Mlle': 'Others_F',
  'Col': 'Others_M',
  'Capt': 'Others_M',
  'the Countess': 'Others_F',
  'Jonkheer': 'Others_M',
  'Dona': 'Others_F',
}
#Updating Titles as per defined mapping
Train_Master['Title'] = Train_Master['Title'].map(Title_dict)
Test_Master['Title'] = Test_Master['Title'].map(Title_dict)
Titanic_Master['Title'] = Titanic_Master['Title'].map(Title_dict)
fig, ax  = plt.subplots(2,1)
fig.set_figheight(8)
fig.set_figwidth(12)
fig.tight_layout()
sns.countplot(x='Title',data=Titanic_Master,ax=ax[0])
sns.boxplot(x='Title',y ='Age', data=Titanic_Master,ax=ax[1])
plt.show()
#Train_Master = pd.get_dummies(Train_Master, columns=['Title'], drop_first=True)
#Test_Master = pd.get_dummies(Test_Master, columns=['Title'], drop_first=True)
#Train_Master.head()
# Skipping this part since we may need 'Title' to derive other features like 'Age'
Titanic_Master.groupby(['Title'])['Age'].median()
titles = ['Dr', 'Mr', 'Master','Miss','Mrs','Others_F','Others_M','Rev']
titles
for title in titles: #['Dr', 'Master','Miss','Mrs','Others_F','Others_M']
    Test_Master.loc[(Test_Master['Title'] == title) & (Test_Master['Age'].isnull()), 'Age'] = Titanic_Master[Titanic_Master['Title'] == title]['Age'].median()
    Train_Master.loc[(Train_Master['Title'] == title) & (Train_Master['Age'].isnull()) , 'Age'] = Titanic_Master[Titanic_Master['Title'] == title]['Age'].median()
print('---'*25)
# Distribution of Age
sns.distplot(Titanic_Master.Age.dropna(), bins=50)
#Till this point we have gained great insight over data, we can logically decide bins.
# Min_age in our data = 0.92
# Max_age in our data = 80
 
# We have also seen there is relation between <code>Age</code> and <code>Title</code>
#Titanic_Master.dropna().groupby(['Title'])['Age'].describe()
#------------------------------------------------------------------------------------------
Titanic_Master.dropna()['Age'].describe()
# mapping Age to bin 
#0  = <24
#1  = >24 and  < 36
#2  = >36 and <47
#3  = >47
Train_Master.loc[ Train_Master['Age'] <= 24, 'Age'] = 0
Train_Master.loc[(Train_Master['Age'] > 24) & (Train_Master['Age'] <= 36), 'Age'] = 1
Train_Master.loc[(Train_Master['Age'] > 36) & (Train_Master['Age'] <= 47), 'Age'] = 2
Train_Master.loc[ Train_Master['Age'] > 47 , 'Age'] = 3
Train_Master['Age'] = Train_Master['Age'].astype(int)

Test_Master.loc[ Test_Master['Age'] <= 24, 'Age'] = 0
Test_Master.loc[(Test_Master['Age'] > 24) & (Test_Master['Age'] <= 36), 'Age'] = 1
Test_Master.loc[(Test_Master['Age'] > 36) & (Test_Master['Age'] <= 47), 'Age'] = 2
Test_Master.loc[ Test_Master['Age'] > 47 , 'Age'] = 3
Test_Master['Age'] = Test_Master['Age'].astype(int)

Train_Master = pd.get_dummies(Train_Master, columns=['Title'])
Test_Master = pd.get_dummies(Test_Master, columns=['Title'])
Train_Master = pd.get_dummies(Train_Master, columns=['Age'])
Test_Master = pd.get_dummies(Test_Master, columns=['Age'])  #, drop_first=True
Test_Master.head()
Titanic_Master = Train_Master.append(Test_Master)
Titanic_Master.reset_index(inplace=True, drop=True)
Titanic_Master.Fare.isnull().any()

median = np.median(Titanic_Master.Fare)
upper_quartile = np.percentile(Titanic_Master.Fare, 75)
lower_quartile = np.percentile(Titanic_Master.Fare, 25)

iqr = upper_quartile - lower_quartile
upper_whisker = Titanic_Master['Fare'][Titanic_Master['Fare']<=upper_quartile+1.5*iqr].max()
lower_whisker = Titanic_Master['Fare'][Titanic_Master['Fare']>=lower_quartile-1.5*iqr].min()

fig, ax = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(16)
sns.boxplot(y='Fare',data=Titanic_Master, ax=ax[0], color='green')
sns.distplot(Titanic_Master.Fare, bins=100,ax=ax[1])
plt.sca(ax[1])
plt.xticks(np.arange(min(Titanic_Master.Fare), max(Titanic_Master.Fare), 50))
plt.sca(ax[0])
plt.yticks(np.arange(min(Titanic_Master.Fare), max(Titanic_Master.Fare), 50))
fig.tight_layout()
plt.axhline(y=upper_quartile,linewidth=2, color='b', linestyle='--')
plt.axhline(y=upper_whisker,linewidth=2, color='b', linestyle='--')
ax[0].annotate('Upper Whisker', xy=(0,upper_whisker), xytext=(0,upper_whisker))
plt.show()
fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(8)
sns.boxplot(y='Fare',data=Titanic_Master[Titanic_Master.Fare<upper_whisker], ax=ax, color='green')
plt.axhline(y=53,linewidth=2, color='b', linestyle='--')
plt.show()
Titanic_Master[Titanic_Master.Fare<=53]['Fare'].describe()
print('Total No. of  Passengers , where Fare of individual is more than 50 is {}'.format(Titanic_Master[Titanic_Master.Fare>50]['Fare'].count()))
print('Total Fare paid by Passengers , where Fare of individual is more than 50 is {}'.format(Titanic_Master[Titanic_Master.Fare>50]['Fare'].sum()))
print('\n')
print('Total No. of  Passengers , where Fare of individual is less  than 50 is {}'.format(Titanic_Master[Titanic_Master.Fare<=50]['Fare'].count()))
print('Total Fare paid by Passengers , where Fare of individual is less than 50 is {}'.format(Titanic_Master[Titanic_Master.Fare<=50]['Fare'].sum()))
Train_Master.loc[ Train_Master['Fare'] == 0.0 , 'Fare'] = 0
Train_Master.loc[(Train_Master['Fare'] > 0.0) & (Train_Master['Fare'] <= 7.85), 'Fare'] = 1
Train_Master.loc[(Train_Master['Fare'] > 7.85) & (Train_Master['Fare'] <= 12.2), 'Fare'] = 2
Train_Master.loc[(Train_Master['Fare'] > 12.2) & (Train_Master['Fare'] <= 24), 'Fare'] = 3
Train_Master.loc[(Train_Master['Fare'] > 24) & (Train_Master['Fare'] <= 53), 'Fare'] = 4
Train_Master.loc[ Train_Master['Fare'] > 53, 'Fare'] = 5
Train_Master['Fare'] = Train_Master['Fare'].astype(int)

Test_Master.loc[ Test_Master['Fare'] == 0.0 , 'Fare'] = 0
Test_Master.loc[(Test_Master['Fare'] > 0.0) & (Test_Master['Fare'] <= 7.85), 'Fare'] = 1
Test_Master.loc[(Test_Master['Fare'] > 7.85) & (Test_Master['Fare'] <= 12.2), 'Fare'] = 2
Test_Master.loc[(Test_Master['Fare'] > 12.2) & (Test_Master['Fare'] <= 24), 'Fare'] = 3
Test_Master.loc[(Test_Master['Fare'] > 24) & (Test_Master['Fare'] <= 53), 'Fare'] = 4
Test_Master.loc[ Test_Master['Fare'] > 53, 'Fare'] = 5
Test_Master['Fare'] = Test_Master['Fare'].astype(int)
Train_Master = pd.get_dummies(Train_Master, columns=['Fare'], drop_first=True)
Test_Master = pd.get_dummies(Test_Master, columns=['Fare'], drop_first=True)
Test_Master.head()
Titanic_Master = Train_Master.append(Test_Master)
Titanic_Master.reset_index(inplace=True, drop=True)
Titanic_Master[['Name','Ticket', 'Pclass', 'Cabin']].head()
Cabin_t = []
Pclass_t = []

# Split ticket for both type of ticket numbers as mentioned earlier.
# We will create Cabin_t for text extracted from Ticket No
# And Pclass_t for first digit of ticket number.
for i in range(len(Titanic_Master.Ticket.str.split(" ", n = 1))):
        if len(Titanic_Master.Ticket.str.split(" ", n = 1)[i]) == 1:
            Pclass_t.append(Titanic_Master.Ticket.str.slice(0,1)[i])
            #Cabin_t.append(None)
            Cabin_t.append('U')
        elif len(Titanic_Master.Ticket.str.split(" ", n = 1)[i]) == 2:
            Pclass_t.append(Titanic_Master.Ticket.str.split(" ", n = 1)[i][1][0])
            #Cabin_t.append(Titanic_Master.Ticket.str.split(" ", n = 1)[i][0][0])
            Cabin_t.append(Titanic_Master.Ticket.str.split(" ", n = 1)[i][0].replace( '.' , '' ).replace( '/' , '' ))
Titanic_Master['Cabin_t'] = Cabin_t
Titanic_Master['Pclass_t'] = Pclass_t
Titanic_Master['Cabin_t']= Titanic_Master['Cabin_t'].str.replace('\d+', '')
Titanic_Master[['Ticket', 'Cabin','Cabin_t','Pclass', 'Pclass_t']].sample(10)
#Titanic_Master[(~Titanic_Master['Cabin'].isna()) & (~Titanic_Master['Cabin_t'].isna())][['Cabin','Ticket','Cabin_t','Pclass','Embarked']]
Titanic_Master.Cabin_t.unique()
Titanic_Master['Pclass'] = Titanic_Master['Pclass'].astype(str)
Titanic_Master['Pclass_t'] = Titanic_Master['Pclass_t'].astype(str)
fig, ax = plt.subplots(2,2)
fig.tight_layout()
fig.set_figheight(8)
fig.set_figwidth(12)
sns.countplot('Pclass_t',data=Titanic_Master[Titanic_Master.Pclass == Titanic_Master.Pclass_t], ax=ax[0,0])
ax[0,0].set_title('Pclas == Pclass_t')

sns.scatterplot('Pclass', 'Pclass_t',data=Titanic_Master[Titanic_Master.Pclass == Titanic_Master.Pclass_t], ax=ax[0,1])
sns.scatterplot('Pclass', 'Pclass_t', data=Titanic_Master[Titanic_Master.Pclass != Titanic_Master.Pclass_t],ax=ax[1,1])
sns.countplot('Pclass_t',data=Titanic_Master[Titanic_Master.Pclass != Titanic_Master.Pclass_t], ax=ax[1,0])
ax[1,0].set_title('Pclas != Pclass_t')
plt.show()
Train_Master['FamilySize'] = Train_Master['SibSp'] + Train_Master['Parch'] + 1
Test_Master['FamilySize'] = Test_Master['SibSp'] + Test_Master['Parch'] + 1
Titanic_Master['FamilySize'] = Titanic_Master['SibSp'] + Titanic_Master['Parch'] + 1
sns.countplot(x='FamilySize', data=Titanic_Master)
plt.show()
# 1     : Travelling Alone
# 2,3,4 : Small Family
# >5    : Large Family
#{'1': 1, '2':'2', '3':'2', '4':'2', '5':'3', '6':'3', '7':'3', '8':'3', '11':'3'}
Titanic_Master['FamilySize'] = Titanic_Master['FamilySize'].map({1:'1',2:'2',3:'2',4:'2',5:'3',6:'3',7:'3',8:'3',11:'3'})
Train_Master['FamilySize'] = Train_Master['FamilySize'].map({1:'1',2:'2',3:'2',4:'2',5:'3',6:'3',7:'3',8:'3',11:'3'})
Test_Master['FamilySize'] = Test_Master['FamilySize'].map({1:'1',2:'2',3:'2',4:'2',5:'3',6:'3',7:'3',8:'3',11:'3'})
Train_Master = pd.get_dummies(Train_Master, columns=['FamilySize']) #drop_first=True
Test_Master = pd.get_dummies(Test_Master, columns=['FamilySize'])
Train_Master = pd.get_dummies(Train_Master, columns=['Sex'], drop_first=True)
Test_Master = pd.get_dummies(Test_Master, columns=['Sex'], drop_first=True)

Train_Master = pd.get_dummies(Train_Master, columns=['Pclass'], drop_first=True)
Test_Master = pd.get_dummies(Test_Master, columns=['Pclass'], drop_first=True)

Train_Master = pd.get_dummies(Train_Master, columns=['Embarked'], drop_first=True)
Test_Master = pd.get_dummies(Test_Master, columns=['Embarked'], drop_first=True)
Train_Master['Cabin'].fillna('U', inplace=True)
Test_Master['Cabin'].fillna('U', inplace=True)

Train_Master['Cabin'] = Train_Master['Cabin'].str[0]
Test_Master['Cabin'] = Test_Master['Cabin'].str[0]

Train_Master = pd.get_dummies(Train_Master, columns=['Cabin']) #, drop_first=True
Test_Master = pd.get_dummies(Test_Master, columns=['Cabin'])

Train_Master.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket'], inplace=True)
Test_Master.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket'],inplace=True)
#Train_Master.head()
#Train_Master.head()
Train_Master.drop(columns=['Title_Rev', 'FamilySize_3', 'Cabin_U','Age_0'], inplace=True)
Test_Master.drop(columns=['Title_Rev', 'FamilySize_3', 'Cabin_U','Age_0'],inplace=True)
Train_Master.head()
fig, ax = plt.subplots()
fig.tight_layout()
fig.set_figheight(20)
fig.set_figwidth(20)
sns.heatmap(Train_Master.corr(), cbar=False,annot=True, ax=ax,linewidths=1, linecolor='black')
plt.show()
Corr = Train_Master.corr().iloc[0].to_frame().reset_index().rename(columns={"index": "Feature", "Survived": "Corr_Coeff"})
fig, ax = plt.subplots()
fig.tight_layout()
fig.set_figheight(6)
fig.set_figwidth(20)
g= sns.barplot('Feature','Corr_Coeff', data=Corr,ax=ax,palette='rainbow')
g.set_xticklabels(labels=xlabels, rotation=45)
plt.show()
y = Train_Master.Survived
X = Train_Master.drop('Survived', axis=1)

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state =1,test_size=0.30)

ML_103_RFC = RandomForestClassifier(random_state=1,n_estimators =200,max_depth=10)
ML_103_RFC.fit(train_X, train_y)
pred_ML_103_RFC = ML_103_RFC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_103_RFC))
print(classification_report(val_y, pred_ML_103_RFC))
print(accuracy_score(val_y, pred_ML_103_RFC))

Test_Master['Cabin_T'] = 0
pred_ML_103_sub = ML_103_RFC.predict(Test_Master)
my_submission = pd.DataFrame(data={'PassengerId':test_ids, 'Survived':pred_ML_103_sub})

print(my_submission['Survived'].value_counts())

#Export Results to CSV
my_submission.to_csv('submission.csv', index = False)
ML_103_SVC = SVC(probability=True,random_state=1)
ML_103_SVC.fit(train_X, train_y)
pred_ML_103_SVC = ML_103_SVC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_103_SVC))
print(classification_report(val_y, pred_ML_103_SVC))
print(accuracy_score(val_y, pred_ML_103_SVC))

from sklearn.grid_search import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid,verbose=3)
grid.fit(train_X,train_y)

pred_ML_103_SVC_grid = grid.predict(val_X)
print(confusion_matrix(val_y,pred_ML_103_SVC_grid))
print(classification_report(val_y, pred_ML_103_SVC_grid))
print(accuracy_score(val_y, pred_ML_103_SVC_grid))

##===========================================================
from sklearn.ensemble import GradientBoostingClassifier
ML_103_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.95, max_features=15, max_depth = 2, random_state = 0)
ML_103_GBC.fit(train_X, train_y)
pred_ML_103_GBC = ML_103_GBC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_103_GBC))
print(classification_report(val_y, pred_ML_103_GBC))
print(accuracy_score(val_y, pred_ML_103_GBC))
##==================================================
print('BaggingClassifier---')
from sklearn.ensemble import BaggingClassifier
ML_103_BC = BaggingClassifier(max_samples= 0.25, n_estimators= 300, max_features=10)
ML_103_BC.fit(train_X, train_y)
pred_ML_103_BC = ML_103_BC.predict(val_X)

print(confusion_matrix(val_y,pred_ML_103_BC))
print(classification_report(val_y, pred_ML_103_BC))
print(accuracy_score(val_y, pred_ML_103_BC))
print('*-*'*25)
##==================================================

print('Submit Results for Best working model.')
pred_ML_103_sub = ML_103_RFC.predict(Test_Master)
my_submission = pd.DataFrame(data={'PassengerId':test_ids, 'Survived':pred_ML_103_sub})

print(my_submission['Survived'].value_counts())

#Export Results to CSV
my_submission.to_csv('submission.csv', index = False)
