#importing required libraries

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Loading data 

train = pd.read_csv('../input/black-friday/train.csv')

test = pd.read_csv('../input/black-friday/test.csv')
train.head()
train.tail()
train.info()
train.describe()
train.columns
train.shape
#checking that null values in Product_Category_3 and Product_Category_2
train.Product_Category_3.value_counts(dropna=False).head()
train.Product_Category_2.value_counts(dropna=False).head()
test.head()

test.tail()
test.info()
test.describe()

test.Product_Category_3.value_counts(dropna=False).head()
test.Product_Category_2.value_counts(dropna=False).head()
#calculating missing values in terms of percentage

PC_2 = (173638/550068)*100

PC_3 = (383247/550068)*100

print ('Missing values in Product_Category_2 of train dataset is {0}%'.format(PC_2))

print ('Missing values in Product_Category_3 of train dataset is {0}%'.format(PC_3))

pc_2 = (72344/233599)*100

pc_3 = (162562/233599)*100

print ('Missing values in Product_Category_2 of test dataset is {0}%'.format(pc_2))

print ('Missing values in Product_Category_3 of test dataset is {0}%'.format(pc_3))
# working on missing data or null values

# removing Product_Category_3 from both data sets as more than 50% data is missing from the column

train.drop(columns='Product_Category_3',axis=1,inplace=True)

test.drop(columns='Product_Category_3',axis=1,inplace=True)
test.head()
train.head()
# Now filling the null values in Product_Category_2

sns.countplot(x='Product_Category_2', data=train)
train['Product_Category_2'].min()
train['Product_Category_2'].max()
test['Product_Category_2'].min()
test['Product_Category_2'].max()
# Filling null values randomly 
def fillNaN_with_random(data):

    a = data.values

    m = np.isnan(a) 

    

    a[m] = np.random.randint(2, 18, size=m.sum())

    return data
fillNaN_with_random(train['Product_Category_2'])
fillNaN_with_random(test['Product_Category_2'])
train.Product_Category_2.value_counts(dropna=False).head()
test.Product_Category_2.value_counts(dropna=False).head()
assert train.notnull().all().all()  # this line will through error if there will be any null value in data
assert test.notnull().all().all()
# checking unique entries in columns with object data type

for col_name in ['Gender', 'Age', 'City_Category','Stay_In_Current_City_Years']:

    print(sorted(train[col_name].unique()))
train.dtypes
# saving the names of all the columns having obect data type in obj_cols

obj_cols=train.select_dtypes(include=['object']).columns

print(obj_cols)
# applying label encoder to convert object data types into int data type

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train[obj_cols] = train[obj_cols].apply(lambda col: le.fit_transform(col))

train[obj_cols].head()
test[obj_cols] = test[obj_cols].apply(lambda col: le.fit_transform(col))

test[obj_cols].head()
#Visualizing Data
plt.figure(figsize = (10, 10))

sns.heatmap( train.corr(),annot=True)

plt.title('Correlation Of Train Data')
plt.figure(figsize = (10, 10))

sns.heatmap( test.corr(),annot=True)

plt.title('Correlation Of Test Data')
fig = plt.figure(figsize=(15,4))

train['Purchase'] = train.Purchase.apply(lambda amount : amount-(amount%1000))

sns.countplot(x='Purchase', data=train)
# checking for outliers in data 

sns.boxplot(x='Purchase',data=train)
sns.boxplot(x='Age',data=train)
sns.boxplot(x='Product_Category_1',data=train)
sns.boxplot(x='Product_Category_2',y='Purchase',data=train)
sns.boxplot(x='Product_Category_1',y='Purchase',data=train)
sns.boxplot(x='Age',y='Purchase',hue='Gender',data=train)
sns.boxplot(x='City_Category',y='Purchase',data=train)
train.shape
# there are outliers in three columns, removing outliers from the data 

outliers = train[['Age','Product_Category_1','Purchase']]

outliers.head()

Q1 = outliers.quantile(0.25)

Q3 = outliers.quantile(0.75)

IQR = Q3 - Q1

print(IQR) #prints IQR for each column

lb = Q1 -( 1.5 * IQR)

print('lower bound is \n',lb)

up = Q3 + (1.5 * IQR)

print('uper bound is \n',up)
print(outliers < lb ) |(outliers > up )
outliers.shape
train= train [~((outliers < lb ) |(outliers > up )).any(axis=1) ]
train.head()
train.shape
outliersTest = test[['Age','Product_Category_1']]

outliers.head()

Q1 = outliersTest.quantile(0.25)

Q3 = outliersTest.quantile(0.75)

IQR = Q3 - Q1

print(IQR) 
print(outliersTest <  Q1 -( 1.5 * IQR)) |(outliersTest >  Q3 + (1.5 * IQR) )
test= test [~((outliersTest < Q1 -( 1.5 * IQR) ) |(outliersTest > Q3 + (1.5 * IQR) )).any(axis=1) ]
test.shape
sns.boxplot(x='Product_Category_1',data=train)
sns.boxplot(x='Age',data=train)
sns.boxplot(x='Purchase',data=train)
train.to_csv("clean_train.csv",index=False, encoding='utf8')

test.to_csv("clean_test.csv",index=False, encoding='utf8')
testData = pd.read_csv('clean_test.csv')

testData.head()
ytrain = train ['Purchase']

Xtrain = train.drop(['Purchase'],axis=1)

Xtest = testData

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(Xtrain,ytrain)
prediction = tree.predict(Xtest)
prediction
print(" Accuaracy is {0}%".format(tree.score(Xtest,prediction)*100))
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 42)
regressor.fit(Xtrain,ytrain)
reg_pred = regressor.predict(Xtest)

reg_pred = [round(x) for x in reg_pred]
reg_pred 
ytrain
print("Accuracy is {0} %".format(regressor.score(Xtest,reg_pred)*100))
# making new dataframe

submission = pd.DataFrame()
#copying test data to this new data frame

submission = Xtest

submission.head()
#Assign all predictions of test Data to this new Data frame

submission['Purchase']=reg_pred
submission.head()
submission.shape
submission.to_csv("BlackFridayResults.csv",index=False, encoding='utf8')