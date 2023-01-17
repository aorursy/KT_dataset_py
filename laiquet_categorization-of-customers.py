import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import linear_model

import warnings
#Disabling warnings

warnings.simplefilter("ignore")
#Importing data

data = pd.read_csv('../input/BlackFriday.csv')
#Data shape & Description

print(data.shape)

print(data.describe())
#Missing values

data.isna().sum()
#Filling missings data of respective columns with their respective mean values

data['Product_Category_2'] = data['Product_Category_2'].fillna(round(data['Product_Category_2'].mean()))

data['Product_Category_3'] = data['Product_Category_3'].fillna(round(data['Product_Category_3'].mean()))
#Checking proportion of male & female customers

pl.figure(figsize =(10,3))

data.groupby('Gender').User_ID.count().plot('barh')

pl.ylabel('Gender', fontsize=12)

pl.xlabel('Total Count of Customers', fontsize=12)

pl.title('Proportion of male & female customers', fontsize=12)

plt.show()



plt.pie(data["Gender"].value_counts().values, labels=["Males","Females"], autopct="%1.0f%%", wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.title("Proportion of male & female customers")

plt.show()
#Comparing total purchase amounts of single & married individuals

pl.figure(figsize =(10,4))

data.groupby('Marital_Status').Purchase.sum().plot('barh')

pl.ylabel('Marital Status {0:Single ; 1:Married}', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of Single & Married individuals', fontsize=12)

plt.show()



plt.pie(data["Marital_Status"].value_counts().values, labels=["Single","Married"], autopct="%1.0f%%", wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.title("Proportion of Single & Married individuals purchases")

plt.show()
#Comparing total purchase amounts of different age individuals

pl.figure(figsize =(10,4))

data.groupby('Age').Purchase.sum().plot('barh')

pl.ylabel('Age Buckets', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of different age individuals', fontsize=12)

plt.show()
#Comparing total purchase amounts w.r.t individuals occupation

pl.figure(figsize =(15,7))

data.groupby('Occupation').Purchase.sum().plot('barh')

pl.ylabel('Occupation Category', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of different Occupation individuals', fontsize=12)

plt.show()
#Comparing total purchase amounts w.r.t individuals residence

pl.figure(figsize =(10,3))

data.groupby('City_Category').Purchase.sum().plot('barh')

pl.ylabel('City Category', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of different City Categories', fontsize=12)

plt.show()



plt.pie(data["City_Category"].value_counts().values, labels=["B","C","A"], autopct="%1.0f%%", wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.title("Proportion of purchases city category wise")

plt.show()
#Comparing total purchase amounts of items in product category 1

pl.figure(figsize =(15,7))

data.groupby('Product_Category_1').Purchase.sum().plot('barh')

pl.ylabel('Product Category 1: Items', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of Product Category 1 Items', fontsize=15)

plt.show()
#Comparing total purchase amounts of items in product category 2

pl.figure(figsize =(15,7))

data.groupby('Product_Category_2').Purchase.sum().plot('barh')

pl.ylabel('Product Category 2: Items', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of Product Category 2 Items', fontsize=15)

plt.show()
#Comparing total purchase amounts of items in product category 3

pl.figure(figsize =(15,7))

data.groupby('Product_Category_3').Purchase.sum().plot('barh')

pl.ylabel('Product Category 3: Items', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts of Product Category 3 Items', fontsize=15)

plt.show()
#Comparing total purchase amounts w.r.t individuals city & their residing years

pl.figure(figsize =(15,7))

data.groupby(['City_Category','Stay_In_Current_City_Years']).Purchase.sum().plot('barh')

pl.ylabel('City & Staying years', fontsize=12)

pl.xlabel('Total Purchase Amounts', fontsize=12)

pl.title('Total Purchase Amounts w.r.t City & Staying years', fontsize=15)

plt.show()
#Total Purchases of Specific Product category 1 items among male & female customers

males_spp = np.array(data[data['Gender']=='M'].groupby(['Product_Category_1']).Purchase.count())

females_spp = np.array(data[data['Gender']=='F'].groupby(['Product_Category_1']).Purchase.count())



pl.figure(figsize =(20,10))

N = 18

ind = np.arange(N) 

width = 0.35       

plt.bar(ind, males_spp, width, label='Male Customers')

plt.bar(ind + width, females_spp, width, label='Female Customers')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 1 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 1 items among male & female customers', fontsize=15)



plt.xticks(ind + width / 2, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 2 items among male & female customers

males_spp = np.array(data[data['Gender']=='M'].groupby(['Product_Category_2']).Purchase.count())

females_spp = np.array(data[data['Gender']=='F'].groupby(['Product_Category_2']).Purchase.count())



pl.figure(figsize =(20,10))

N = 17

ind = np.arange(N) 

width = 0.35       

plt.bar(ind, males_spp, width, label='Male Customers')

plt.bar(ind + width, females_spp, width, label='Female Customers')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 2 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 2 items among male & female customers', fontsize=15)



plt.xticks(ind + width / 2, ('2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 3 items among male & female customers

males_spp = np.array(data[data['Gender']=='M'].groupby(['Product_Category_3']).Purchase.count())

females_spp = np.array(data[data['Gender']=='F'].groupby(['Product_Category_3']).Purchase.count())



pl.figure(figsize =(20,10))

N = 15

ind = np.arange(N) 

width = 0.35       

plt.bar(ind, males_spp, width, label='Male Customers')

plt.bar(ind + width, females_spp, width, label='Female Customers')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 3 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 3 items among male & female customers', fontsize=15)



plt.xticks(ind + width / 2, ('3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 1 items among individuals of different ages

age017 = np.array(data[data['Age']=='0-17'].groupby(['Product_Category_1']).Purchase.count())

age1825 = np.array(data[data['Age']=='18-25'].groupby(['Product_Category_1']).Purchase.count())

age2635 = np.array(data[data['Age']=='26-35'].groupby(['Product_Category_1']).Purchase.count())

age3645 = np.array(data[data['Age']=='36-45'].groupby(['Product_Category_1']).Purchase.count())

age4650 = np.array(data[data['Age']=='46-50'].groupby(['Product_Category_1']).Purchase.count())

age5155 = np.array(data[data['Age']=='51-55'].groupby(['Product_Category_1']).Purchase.count())

age55p = np.array(data[data['Age']=='55+'].groupby(['Product_Category_1']).Purchase.count())



pl.figure(figsize =(20,15))

N = 18

ind = np.arange(N) 

width = 0.10       

plt.bar(ind, age017, width, label='0 - 17 years')

plt.bar(ind + width, age1825, width, label='18 - 25 years')

plt.bar(ind + width+ width, age2635, width, label='26 - 35 years')

plt.bar(ind + width+ width+ width, age3645, width, label='36 - 45 years')

plt.bar(ind + width+ width+ width+ width, age4650, width, label='46 - 50 years')

plt.bar(ind + width+ width+ width+ width+ width, age5155, width, label='51 - 55 years')

plt.bar(ind + width+ width+ width+ width+ width+ width, age55p, width, label='55+ years')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 1 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 1 items among individuals of different ages', fontsize=15)



plt.xticks(ind + width / 7, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 2 items among individuals of different ages

age017 = np.array(data[data['Age']=='0-17'].groupby(['Product_Category_2']).Purchase.count())

age1825 = np.array(data[data['Age']=='18-25'].groupby(['Product_Category_2']).Purchase.count())

age2635 = np.array(data[data['Age']=='26-35'].groupby(['Product_Category_2']).Purchase.count())

age3645 = np.array(data[data['Age']=='36-45'].groupby(['Product_Category_2']).Purchase.count())

age4650 = np.array(data[data['Age']=='46-50'].groupby(['Product_Category_2']).Purchase.count())

age5155 = np.array(data[data['Age']=='51-55'].groupby(['Product_Category_2']).Purchase.count())

age55p = np.array(data[data['Age']=='55+'].groupby(['Product_Category_2']).Purchase.count())



pl.figure(figsize =(20,15))

N = 17

ind = np.arange(N) 

width = 0.10       

plt.bar(ind, age017, width, label='0 - 17 years')

plt.bar(ind + width, age1825, width, label='18 - 25 years')

plt.bar(ind + width+ width, age2635, width, label='26 - 35 years')

plt.bar(ind + width+ width+ width, age3645, width, label='36 - 45 years')

plt.bar(ind + width+ width+ width+ width, age4650, width, label='46 - 50 years')

plt.bar(ind + width+ width+ width+ width+ width, age5155, width, label='51 - 55 years')

plt.bar(ind + width+ width+ width+ width+ width+ width, age55p, width, label='55+ years')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 2 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 2 items among individuals of different ages', fontsize=15)



plt.xticks(ind + width / 7, ('2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 3 items among individuals of different ages

age017 = np.array(data[data['Age']=='0-17'].groupby(['Product_Category_3']).Purchase.count())

age1825 = np.array(data[data['Age']=='18-25'].groupby(['Product_Category_3']).Purchase.count())

age2635 = np.array(data[data['Age']=='26-35'].groupby(['Product_Category_3']).Purchase.count())

age3645 = np.array(data[data['Age']=='36-45'].groupby(['Product_Category_3']).Purchase.count())

age4650 = np.array(data[data['Age']=='46-50'].groupby(['Product_Category_3']).Purchase.count())

age5155 = np.array(data[data['Age']=='51-55'].groupby(['Product_Category_3']).Purchase.count())

age55p = np.array(data[data['Age']=='55+'].groupby(['Product_Category_3']).Purchase.count())



pl.figure(figsize =(20,15))

N = 15

ind = np.arange(N) 

width = 0.10       

plt.bar(ind, age017, width, label='0 - 17 years')

plt.bar(ind + width, age1825, width, label='18 - 25 years')

plt.bar(ind + width+ width, age2635, width, label='26 - 35 years')

plt.bar(ind + width+ width+ width, age3645, width, label='36 - 45 years')

plt.bar(ind + width+ width+ width+ width, age4650, width, label='46 - 50 years')

plt.bar(ind + width+ width+ width+ width+ width, age5155, width, label='51 - 55 years')

plt.bar(ind + width+ width+ width+ width+ width+ width, age55p, width, label='55+ years')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 3 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 3 items among individuals of different ages', fontsize=15)



plt.xticks(ind + width / 7, ('3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 1 items among married/Single males/females

sm = np.array(data[np.logical_and(data['Gender']=='M', data['Marital_Status']==0)].groupby(['Product_Category_1']).Purchase.count())

mm = np.array(data[np.logical_and(data['Gender']=='M', data['Marital_Status']==1)].groupby(['Product_Category_1']).Purchase.count())

sf = np.array(data[np.logical_and(data['Gender']=='F', data['Marital_Status']==0)].groupby(['Product_Category_1']).Purchase.count())

mf = np.array(data[np.logical_and(data['Gender']=='F', data['Marital_Status']==1)].groupby(['Product_Category_1']).Purchase.count())



pl.figure(figsize =(20,15))

N = 18

ind = np.arange(N) 

width = 0.15       

plt.bar(ind, sm, width, label='Single Males')

plt.bar(ind + width, mm, width, label='Married Males')

plt.bar(ind + width+ width, sf, width, label='Single Females')

plt.bar(ind + width+ width+ width, mf, width, label='Married Females')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 1 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 1 items among married/Single males/females', fontsize=15)



plt.xticks(ind + width / 7, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 2 items among married/Single males/females

sm = np.array(data[np.logical_and(data['Gender']=='M', data['Marital_Status']==0)].groupby(['Product_Category_2']).Purchase.count())

mm = np.array(data[np.logical_and(data['Gender']=='M', data['Marital_Status']==1)].groupby(['Product_Category_2']).Purchase.count())

sf = np.array(data[np.logical_and(data['Gender']=='F', data['Marital_Status']==0)].groupby(['Product_Category_2']).Purchase.count())

mf = np.array(data[np.logical_and(data['Gender']=='F', data['Marital_Status']==1)].groupby(['Product_Category_2']).Purchase.count())



pl.figure(figsize =(20,15))

N = 17

ind = np.arange(N) 

width = 0.15       

plt.bar(ind, sm, width, label='Single Males')

plt.bar(ind + width, mm, width, label='Married Males')

plt.bar(ind + width+ width, sf, width, label='Single Females')

plt.bar(ind + width+ width+ width, mf, width, label='Married Females')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 2 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 2 items among married/Single males/females', fontsize=15)



plt.xticks(ind + width / 7, ( '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Total Purchases of Specific Product category 3 items among married/Single males/females

sm = np.array(data[np.logical_and(data['Gender']=='M', data['Marital_Status']==0)].groupby(['Product_Category_3']).Purchase.count())

mm = np.array(data[np.logical_and(data['Gender']=='M', data['Marital_Status']==1)].groupby(['Product_Category_3']).Purchase.count())

sf = np.array(data[np.logical_and(data['Gender']=='F', data['Marital_Status']==0)].groupby(['Product_Category_3']).Purchase.count())

mf = np.array(data[np.logical_and(data['Gender']=='F', data['Marital_Status']==1)].groupby(['Product_Category_3']).Purchase.count())



pl.figure(figsize =(20,15))

N = 15

ind = np.arange(N) 

width = 0.15       

plt.bar(ind, sm, width, label='Single Males')

plt.bar(ind + width, mm, width, label='Married Males')

plt.bar(ind + width+ width, sf, width, label='Single Females')

plt.bar(ind + width+ width+ width, mf, width, label='Married Females')



plt.ylabel('Total Purchases', fontsize=15)

plt.xlabel('Product Category 3 Items', fontsize=15)

plt.title('Total Purchases of Specific Product category 3 items among married/Single males/females', fontsize=15)



plt.xticks(ind + width / 7, ('3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'))

plt.legend(loc='best')

plt.show()
#Data Transformation

encode = LabelEncoder()

encode.fit(['0-17','18-25','26-35','36-45','46-50','51-55', '55+'])

data['Age'] = encode.transform(data['Age'])



encode.fit(['M','F'])

data['Gender'] = encode.transform(data['Gender'])



encode.fit(['A','B','C'])

data['City_Category'] = encode.transform(data['City_Category'])



encode.fit(['0','1','2', '3', '4+'])

data['Stay_In_Current_City_Years'] = encode.transform(data['Stay_In_Current_City_Years'])
#Dropping unrequired columns

data.drop(columns=['User_ID', 'Product_ID'], inplace=True)
#Correlation matrix & Heatmap - Finding correlation

pl.figure(figsize =(10,10))

corrmat = data.corr()

sns.heatmap(corrmat, annot=True, fmt='.3f', vmin=0, vmax=1, square=True);

plt.show()
#Categorization of customers in 4 groups based on their items purchased ammount 

#Category-0: 0-6000

#Category-1: 6000-9333

#Category-2: 9333-13000

#Category-3: 13000-24000



data['category'] = 999

data.loc[np.logical_and(data['Purchase']>0, data['Purchase']<=6000), 'category'] = 0

data.loc[np.logical_and(data['Purchase']>6000, data['Purchase']<=9334), 'category'] = 1

data.loc[np.logical_and(data['Purchase']>9334, data['Purchase']<=13000), 'category'] = 2

data.loc[np.logical_and(data['Purchase']>13000, data['Purchase']<=24000), 'category'] = 3

data.head(10)
#Labels and featureSet columns

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['category']]

target = 'category'



X = data[columns]

y = data[target]
#Splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



print("Training FeatureSet:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Testing FeatureSet:", X_test.shape)

print("Testing Labels:", y_test.shape)
#Initializing the model with some parameters.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy:",round(metrics.accuracy_score(y_test, predictions))*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y_test))*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual', 'actual', 'actual'], ['0','1','2','3']],

    columns = [['predicted', 'predicted', 'predicted', 'predicted'], ['0', '1', '2', '3']])

print(df)