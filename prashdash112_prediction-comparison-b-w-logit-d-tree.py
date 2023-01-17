import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import pprint

%matplotlib inline 
df=pd.read_csv(r'../input/adult-income-dataset/adult.csv')

pprint.pprint(df.head())

print('\n\n')

pprint.pprint(df.shape)
sns.heatmap(df.isnull())
cat_df=df.select_dtypes('object')

cat_df.head()
print('unique elements of education column: \n')

pprint.pprint(df.education.unique())

print('\n\n')

print('unique elements of marital-status column: \n')

pprint.pprint(df['marital-status'].unique())

print('\n\n')

print('unique elements of occupation: \n')

pprint.pprint(df['occupation'].unique())

print('\n\n')

print('unique elements of workclass: \n')

pprint.pprint(df['workclass'].unique())

print('\n\n')

print('unique elements of relationship: \n')

pprint.pprint(df['relationship'].unique())

print('\n\n')

print('unique elements of race: \n')

pprint.pprint(df['race'].unique())

print('\n\n')

print('unique elements of gender: \n')

pprint.pprint(df['gender'].unique())

print('\n\n')

print('unique elements of native-country: \n')

pprint.pprint(df['native-country'].unique())

print('\n\n')

print('unique elements of income: \n')

pprint.pprint(df['income'].unique())

print('\n\n')

arr1=[]

for item in cat_df['workclass']:

    if (item == '?'):

        arr1.append(item)

print('Length of missing vals in workclass column:')

print(len(arr1))

print('\n')

arr2=[]

for item in cat_df['occupation']:

    if (item == '?'):

        arr2.append(item)

print('Length of missing vals in occupation column:')

print(len(arr2))
null_data=((2809+2799)/(48842-(2809+2799)))*100

print(null_data)
x=df.select_dtypes(object)
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()

cat_df=oe.fit_transform(cat_df)
cat_df
cat_df1=pd.DataFrame(data=cat_df,columns=x.columns)

cat_df1
num_df1=df.select_dtypes(int)

num_df1
final_df=pd.concat([num_df1,cat_df1],axis=1)

final_df
from sklearn.model_selection import train_test_split 
X=final_df.drop('income',axis=1)

y=final_df['income']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver='lbfgs',max_iter=200)
logmodel.fit(X_train,y_train)
prediction=logmodel.predict(X_test)

prediction
pred=pd.DataFrame(data=prediction,columns=['prediction'])

pred
from sklearn.metrics import classification_report
result=classification_report(pred,y_test)

print(result)
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=7)
tree.fit(X_train,y_train)
predictions=tree.predict(X_test)

print(predictions)
pred2=pd.DataFrame(data=predictions,columns=['predictions'])

pred2['predictions']
def num(n):

    if(n < 0.5):

        return 0

    else:

        return 1
x=pred2['predictions'].apply(num)

x.unique()
result2=classification_report(x,y_test)

print(result2)