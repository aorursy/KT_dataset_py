import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt #for visualisation

import seaborn as sns   #for visualisation

sns.set(style="white") #setting background of vizualisation as white 

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")
%%time

#using pandas to read the train and test file 

train= pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.shape)

print(test.shape)
train.shape, train.columns
train.head() #to understand the type of values in these columns
#Lets now understand our target variable that out 891 how many survived or died in this tragedy

f,ax=plt.subplots(1,2,figsize=(18,8))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Died vs Survived percentage')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Died vs Survived Count')

ax[1].set_xlabel('')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex Percentage')

sns.countplot(x='Survived',hue='Sex',data=train,ax=ax[1])

ax[1].set_title('Survived vs Sex')

plt.legend(loc = 'top left')
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Class Percentage')

sns.countplot(x='Survived',hue='Pclass',data=train,ax=ax[1])

ax[1].set_title('Survived vs Class')

plt.legend(loc = 'top left')
sns.factorplot('Pclass','Survived',hue='Sex',data=train)#This clearly summarises our hypothesis
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])#lets check using violin plot

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))



sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
#Lets plot survival vs rest of the variable 

variable = ['Embarked', 'Parch', 'SibSp']



fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 8))

plt.subplots_adjust(right=1.5, top=1.25)



for i, v in enumerate(variable, 1):    

    plt.subplot(1, 3, i)

    sns.countplot(x=v, hue='Survived', data=train)

    

    plt.xlabel('{}'.format(v), size=10, labelpad=10)

    plt.ylabel('Passenger Count', size=10, labelpad=10)    

    plt.tick_params(axis='x', labelsize=10)

    plt.tick_params(axis='y', labelsize=10)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 10})

    plt.title('Count of Survival in {} Feature'.format(v), size=10, y=1.05)



plt.show()
%%time

titanic_df = pd.concat([train, test])

print(titanic_df.shape) #lets check how concatenation worked

titanic_df.head()   #lets check the first 5 rows now

titanic_df.dtypes #Lets check the type of data we are dealing with...
titanic_df.describe() #summarizes all numerical columns
#lets check data has how many missing values

titanic_df.isnull().sum()

titanic_df['Title'] = titanic_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

titanic_df.head()
titanic_df['Title'].value_counts()
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',

           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',

           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

titanic_df.replace({'Title': mapping}, inplace=True)

titanic_df['Title'].value_counts()#left with only 6 titles now
# impute missing Age values using median of Title groups

title_ages = dict(titanic_df.groupby('Title')['Age'].median())





titanic_df['age_med'] = titanic_df['Title'].apply(lambda x: title_ages[x])



# replace all missing ages with the value in this column

titanic_df['Age'].fillna(titanic_df['age_med'], inplace=True, )

del titanic_df['age_med']

sns.distplot(titanic_df['Age'])
sns.barplot(x='Title', y='Age', data=titanic_df, estimator=np.median, ci=None, palette='Blues_d')

plt.xticks(rotation=45)

plt.show()
titanic_df.isnull().sum()
%%time

#from sklearn.preprocessing import Imputer

#impute = Imputer(missing_values='NaN', strategy='mean', axis=0) #SKlearn Mean imputer

#impute.fit(titanic_df['Age'])

#titanic_df= impute.transform(titanic_df['age'])



#from fancyimpute import KNN

#imputer = KNN(k=2)

#titanic_df['Age'] = imputer.fit_transform(titanic_df['Age'])
sns.heatmap(titanic_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 

#titanic_df.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
%%time

# impute missing Fare values using median of Pclass groups

class_fares = dict(titanic_df.groupby('Pclass')['Fare'].median())



# create a column of the average fares

titanic_df['fare_med'] = titanic_df['Pclass'].apply(lambda x: class_fares[x])



# replace all missing fares with the value in this column

titanic_df['Fare'].fillna(titanic_df['fare_med'], inplace=True, )

del titanic_df['fare_med']

#lets check data has how many missing values

titanic_df.isnull().sum()
titanic_df['Embarked'].value_counts()
#As Embarked is a categorical variable only 2 missing values so we will simply use backfill    

titanic_df['Embarked'].fillna(method='backfill', inplace=True)

titanic_df['Embarked'].isnull().sum()
#we have number of parents and no.of siblings so we can calculate the size of the family 

titanic_df['Family Size'] = titanic_df['Parch'] + titanic_df['SibSp']+1

titanic_df['IsAlone'] = 1 #initialize to yes/1 is alone

titanic_df['IsAlone'].loc[titanic_df['Family Size'] > 1] = 0

titanic_df.columns

#We will now drop column which have mostly Null and unimportant values. 

drop = ['Cabin', 'Ticket','Name','PassengerId']

titanic_df.drop(drop, axis=1, inplace = True)

titanic_df.head()
titanic_df['Sex'].replace(['male','female'],[0,1],inplace=True)

titanic_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

titanic_df['Title'].replace(['Mr','Mrs','Miss','Master','Dr','Rev'],[0,1,2,3,4,5],inplace=True)



#you can also use get dummies or label encoder

titanic_df.head()
#Import all neccessary SKlearn libraries for KNN classifier

from sklearn.preprocessing import MinMaxScaler#to make sure all variables are on the same scale

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import f1_score

from sklearn import metrics
y_train=titanic_df['Survived'].iloc[:891]#training target variable

x_train=titanic_df.drop('Survived', axis=1)#dropping target variable

#y_train.tail()

x_train.tail()
scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x_train)

x_scaled = pd.DataFrame(x_scaled, columns = x_train.columns)

x_scaled.tail()#All variables are now in same scale of 0 to 1 
from sklearn.model_selection import train_test_split

train_scaled= x_scaled.iloc[:891] #train dataset

#train_scaled.shape

test_scaled=x_scaled.iloc[891:]#final dataset

#test_scaled.shape

#Creating training and test dataset from from training dataset

train_x,test_x,train_y,test_y = train_test_split(train_scaled,y_train, random_state = 56, stratify=y_train)
%%time

# Creating instance of KNN

clf = KNN(n_neighbors = 10)



# Fitting the model

clf.fit(train_x, train_y)



# Predicting over the Train Set and calculating accuracy score

test_predict = clf.predict(test_x)

print('The accuracy of the KNN is',metrics.accuracy_score(test_predict,test_y))

final_predict1=clf.predict(test_scaled)

print(final_predict1)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

#Running logistics Regression model

lr = LogisticRegression()

lr.fit(train_x,train_y)

#making predicition on test set created from train_scaled data

valid1=lr.predict(test_x)

print(lr.score(test_x, test_y))

pred1=lr.predict(test_scaled)

pred1[:10]



#Running KNN model

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(train_x,train_y)

#making predicition on test set created from train_scaled data

valid2=knn.predict(test_x)

print(knn.score(test_x, test_y))

pred2=knn.predict(test_scaled)

pred2[:10]
#Running Decision tree model

dt = DecisionTreeClassifier(max_depth=5)

dt.fit(train_x,train_y)

#making predicition on test set created from train_scaled data

valid3=dt.predict(test_x)

print(dt.score(test_x, test_y))

pred3=dt.predict(test_scaled)

pred3[:10]
from sklearn.metrics import accuracy_score

from statistics import mode

final_pred = np.array([])

for i in range(0,len(test)):

    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))



final_pred=final_pred.astype(int)

final_pred.shape
#Creating a New train dataframe

train_prediction = {

              'LR': valid1,

              'knn': valid2,

              'DT': valid3

              }

train_predictions = pd.DataFrame(train_prediction)

train_predictions.shape, test_y.shape
#Creating a New test dataframe

test_prediction = {

              'LR': pred1,

              'knn': pred2,

              'DT': pred3

              }

test_predictions = pd.DataFrame(test_prediction)

test_predictions.head()
# Stacking Model

model = LogisticRegression()

model.fit(train_predictions, test_y)

final_pred=model.predict(test_predictions)

final_pred=final_pred.astype(int)

final_pred[:10],final_pred.dtype
from sklearn.model_selection import KFold

train_pred = np.empty((0,0) , int)

skfold = KFold(10, random_state = 101)

  

#For every permutation of KFold

for i,j in skfold.split(train_x, train_y):

    x_train, x_test = train_x.iloc[i], train_x.iloc[j]

    y_train, y_test = train_y.iloc[i], train_y.iloc[j]



#Everything else remains same as regular stacking



#Running logistics Regression model

lr = LogisticRegression()

lr.fit(x_train,y_train)

#making predicition on test set created from train_scaled data

valid1=lr.predict(x_test)

print(lr.score(x_test, y_test))

pred1=lr.predict(test_scaled)





#Running KNN model

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)

#making predicition on test set created from train_scaled data

valid2=knn.predict(x_test)

print(knn.score(x_test, y_test))

pred2=knn.predict(test_scaled)





#Running Decision tree model

dt = DecisionTreeClassifier(max_depth=5)

dt.fit(x_train,y_train)

#making predicition on test set created from train_scaled data

valid3=dt.predict(x_test)

print(dt.score(x_test, y_test))

pred3=dt.predict(test_scaled)





#Creating a New train dataframe

train_prediction = {

              'LR': valid1,

              'knn': valid2,

              'DT': valid3

              }

train_predictions = pd.DataFrame(train_prediction)







#Creating a New test dataframe

test_prediction = {

              'LR': pred1,

              'knn': pred2,

              'DT': pred3

              }

test_predictions = pd.DataFrame(test_prediction)







# Stacking Model

model = KNeighborsClassifier(n_neighbors=10)

model.fit(train_predictions, y_test)

final_pred=model.predict(test_predictions)

final_pred=final_pred.astype(int)

final_pred[:10],final_pred.dtype
#Importing random forest classifier 

from sklearn.ensemble import RandomForestClassifier

train_accuracy = []

validation_accuracy = []

for depth in range(1,10):

    rfc_model = RandomForestClassifier(max_depth=depth, random_state=10)

    rfc_model.fit(train_x, train_y)

    train_accuracy.append(rfc_model.score(train_x, train_y))

    validation_accuracy.append(rfc_model.score(test_x, test_y))
table = pd.DataFrame({'max_depth':range(1,10), 'train_acc':train_accuracy, 'test_acc':validation_accuracy})

table
plt.figure(figsize=(12,6))

plt.plot(table['max_depth'], table['train_acc'], marker='o')

plt.plot(table['max_depth'], table['test_acc'], marker='o')

plt.xlabel('Depth of tree')

plt.ylabel('performance')

plt.legend()
#creating a random forest instance

RFC = RandomForestClassifier(random_state=10,max_depth=7, max_leaf_nodes=25,n_estimators=12)

RFC.fit(train_x,train_y)

pred4=dt.predict(test_scaled)

final_pred=pred4.astype(int)

final_pred[:10],final_pred.dtype
#Importing GBDT Classifier 

from sklearn.ensemble import GradientBoostingClassifier

gbc= GradientBoostingClassifier(random_state=101)
#putting some random values for hyperp optimisation

parameter_grid = {

    'max_depth' : [4,5,6,7,8],

    'n_estimators': [100,150,200, 250],

    'min_samples_split': [50,100,150,200]

    }
from sklearn.model_selection import GridSearchCV

gridsearch = GridSearchCV(estimator=gbc, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=5)
gridsearch.fit(train_x, train_y)

gridsearch.best_params_
#creating an Gradient boosting instance

gbc= GradientBoostingClassifier(random_state=101, n_estimators=150,min_samples_split=100, max_depth=6)

gbc.fit(train_x,train_y)

print(gbc.score(test_x, test_y))

pred5=gbc.predict(test_scaled)

final_pred=pred5.astype(int)
 #Output the predictions into a csv

#output= pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':final_pred})

#output.to_csv('gender_submission.csv', index=False)