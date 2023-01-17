# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import re

import gc 

from tqdm import tqdm

from datetime import date     #calculating age

from datetime import datetime #converting string to date

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV , train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score , f1_score , make_scorer

from sklearn.preprocessing import StandardScaler,OneHotEncoder , LabelEncoder ,normalize

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/train.csv")

df2 = pd.read_csv("../input/test_bqCt9Pv.csv")

#df1 = dff.drop('loan_default' , axis=1)

#df = pd.concat([df1,df2], axis=0 , sort=True)
df1.head()
#df.drop(['UniqueID'] , inplace=True , axis=1)
df1.describe()
df1.isnull().sum()
df1['Employment.Type'].value_counts()
df1 = df1.fillna(df1.mode().iloc[0])

df2 = df2.fillna(df2.mode().iloc[0])
df1.shape
#Plotting Histogram    

df1.hist(bins=50 , figsize=(20,20))

plt.show()
# Checking on Categorical data

for col in tqdm(df1.columns):

    if df1[col].dtype == 'object':

        print(col , ":", df1[col].nunique())
# CODE FOR AGE CONVERSION: https://www.geeksforgeeks.org/python-program-to-calculate-age-in-year/

def calcAge(born):

    born = datetime.strptime(born , '%d-%m-%y')

    today= date.today()

    age = today.year- born.year - ((today.month,today.day) < (born.month,born.day))

    return age
#1

df1.drop(['DisbursalDate' , 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH'], inplace=True , axis=1)

df2.drop(['DisbursalDate' , 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH'], inplace=True , axis=1)

#2

df1['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=['Not Scored: More than 50 active Accounts found', 

                                                         'Not Scored: No Activity seen on the customer (Inactive)',

                                                         'Not Scored: No Updates available in last 36 months',

                                                         'Not Enough Info available on the customer','Not Scored: Only a Guarantor',

                                                         'Not Scored: Sufficient History Not Available',

                                                         'Not Scored: Not Enough Info available on the customer'], 

                                                   value= 'Not Scored', inplace = True)

df2['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=['Not Scored: More than 50 active Accounts found', 

                                                         'Not Scored: No Activity seen on the customer (Inactive)',

                                                         'Not Scored: No Updates available in last 36 months',

                                                         'Not Enough Info available on the customer','Not Scored: Only a Guarantor',

                                                         'Not Scored: Sufficient History Not Available',

                                                         'Not Scored: Not Enough Info available on the customer'], 

                                                   value= 'Not Scored', inplace = True)



#3 

dob = df1['Date.of.Birth']

df1['Age'] = dob.map(calcAge)

df1.drop(['Date.of.Birth'] , axis=1 , inplace=True)



dob = df2['Date.of.Birth']

df2['Age'] = dob.map(calcAge)

df2.drop(['Date.of.Birth'] , axis=1 , inplace=True)
df1.shape , df2.shape
def removeOutlier(df, cols):

    indexes=[]

    for col in tqdm(cols):

        if (df[col].dtypes !='object'):

            Q1 = df[col].quantile(q=0.001)

            Q3 = df[col].quantile(q=0.999)        

            for i in (df.index):

                if ((df.loc[i,col]< Q1/5) or (df.loc[i,col] > 5*Q3)):

                    df = df.drop(index=i)

                    indexes.append(i)

    return df, indexes
cols_with_outliers=['disbursed_amount', 'asset_cost', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT','PRI.SANCTIONED.AMOUNT',

       'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS','SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',

       'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',

       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'NO.OF_INQUIRIES','Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION']



new_df1, indexex = removeOutlier(df1 , cols_with_outliers)

new_df2, indexex2 = removeOutlier(df2 , cols_with_outliers)
new_df1.shape,new_df2.shape
cat_data= df1.select_dtypes(include='object').columns

#6

df = pd.get_dummies(new_df1[cat_data])

dff = pd.get_dummies(new_df2[cat_data])

print("The shape of dummy variables : " , df.shape)

print("The shape of dummy variables : " , dff.shape)
num_data = list(new_df1._get_numeric_data().columns)

num_data.remove('loan_default')

scaler = StandardScaler()

scaler.fit(new_df1[num_data])

normalized = scaler.transform(new_df1[num_data])

normalized2 = scaler.transform(new_df2[num_data])

normalized = pd.DataFrame(normalized , columns=num_data)

normalized2 = pd.DataFrame(normalized2 , columns=num_data)

print("The shape of normalised numerical data : " , normalized.shape)

print("The shape of normalised numerical data : " , normalized2.shape)
final_df = pd.concat([normalized , df], sort=True , axis=1)

test_df = pd.concat([normalized2 , dff], sort=True , axis=1)

final_df= final_df.dropna(axis=0)

test_df = test_df.dropna(axis=0)

print("The shape of the final data:" , final_df.shape)

print("The shape of the final data:" , test_df.shape)
# Finding Correlation between features

def correlation(df ,column):

    column = column.iloc[df.index]

    dff= df.join(column)

    corr_max = dff.corr()  #create correlation matrix

    top_15 = corr_max.nlargest(20 , 'loan_default')['loan_default'].index # select top 15 correlate features

    corr = np.corrcoef(dff[top_15].values.T)    

    return corr , top_15



#Plot correlation map

def plot_heatmap(corr,top_15):

    plt.figure(figsize=(15,10))

    sns.heatmap(corr, cbar=True , annot=True , fmt='.2f', yticklabels=top_15.values , xticklabels=top_15.values)

    plt.title('CORRELATION MATRIX')

    plt.show()

    
corr , top_15 = correlation(final_df, df1['loan_default'])

print("The top 15 feature correlated to target variable", top_15)
# def LogReg(X, y):

#     kwargs ={

#         'class_weight':None,

#         'fit_intercept':True,

#         'max_iter':1000

#     }

    

#     param_grid =[{ 'C': np.random.rand(5) , 

#                  'solver': ['liblinear'],

#                  'penalty':['l1']

#                  },

#                 { 'C': np.random.rand(5) , 

#                  'solver': ['lbfgs'],

#                  'penalty':['l2']

#                  }]

#     scoring = {'accuracy': make_scorer(accuracy_score) , 

#                'f1': make_scorer(f1_score)}

    

#     X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.3 , random_state=0)

#     lr = LogisticRegression(**kwargs)

#     model =GridSearchCV(lr , param_grid=param_grid , scoring=scoring , cv=5 , refit=False , verbose=10)

#     model.fit(X_train,y_train)

#     print(model.best_params_)

#     print(model.score(X_test , y_test))
X = final_df

y = new_df1['loan_default'].iloc[final_df.index]

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.3 , random_state=0)

c= [0.001 ,0.0001,0.01,1,10]
scores=[]

for i in (c):

    lr = LogisticRegression(penalty='l1' , solver='liblinear' , max_iter=1000 , C=i)

    top15= list(top_15)

    top15.remove('loan_default')

    lr.fit(X_train[top15],y_train)

    scores.append(lr.score(X_test[top15],y_test))

print("The score for C= {} is {}".format(scores.index(max(scores)),max(scores)))
for i in (c):

    lr = LogisticRegression(penalty='l2' , solver='lbfgs' , max_iter=1000 , C=i)

    top15= list(top_15)

    top15.remove('loan_default')

    lr.fit(X_train[top15],y_train)

    scores.append(lr.score(X_test[top15],y_test))

print("The score for C= {} is {}".format(scores.index(max(scores)),max(scores)))
model = LogisticRegression(penalty='l2' , solver='lbfgs' , C=1 ,max_iter=1000 )

model.fit(X,y)

predictions = model.predict(test_df)
predictions