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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/Churn_Modelling.csv')

df.head()
# Here we identify the Target variable and Predictor variable

Target_var=df['Exited']

Pred_var=df.drop(['Exited'],axis=1)
#Function to Identify the catagory of various column.

def variabletype(data):

    colname=data.columns

    coltype=data.dtypes

    variabletype=[]

    for i in data:

        if (data[i].nunique()>11) and (data[i].dtype=='int64' or data[i].dtype=='float64'):

            variabletype.append('Continuous')

        else:

            variabletype.append('Class')

    #variabletype

    dict={'ColumnName':colname,

         'Column_dtype':coltype,

          'Variable_Type':variabletype}

    return pd.DataFrame(dict)

df1=variabletype(df)

df1
cont=df[['RowNumber','CustomerId','CreditScore','Age','Balance','EstimatedSalary']]

cata=df[['Surname','Geography','Gender','Tenure','NumOfProducts','HasCrCard','IsActiveMember']]
cont.describe()
sns.boxplot(x=cont['Age'])
sns.distplot(cont.Age, bins=100)
sns.boxplot(x=cont['RowNumber'])
sns.distplot(cont['RowNumber'], bins=100)
sns.boxplot(x=cont['CustomerId'])
sns.distplot(cont['CustomerId'],bins=100)
sns.boxplot(cont['CreditScore'])
sns.distplot(cont['CreditScore'],bins=100)
sns.boxplot(cont['Balance'])
sns.distplot(cont['Balance'])
cata=df[['Surname','Geography','Gender','Tenure','NumOfProducts','HasCrCard','IsActiveMember']]

sns.countplot(x='Geography',hue='Gender',data=df)

df['Geography'].value_counts()
sns.countplot(x='Tenure',data=df)

df['Tenure'].value_counts()
sns.countplot(x='NumOfProducts',data=df)

df['NumOfProducts'].value_counts()
sns.countplot(x='HasCrCard',hue='Gender',data=df)

df['HasCrCard'].value_counts()
sns.countplot(x='IsActiveMember',data=df)

df['IsActiveMember'].value_counts()
sns.countplot(x='Exited',data=df)

df['Exited'].value_counts()
#'CreditScore','Age','Balance','EstimatedSalary'

sns.pairplot(x_vars=['CreditScore','Age','Balance','EstimatedSalary'],

             y_vars=['CreditScore','Age','Balance','EstimatedSalary'],

             data=df)

#sns.scatterplot(x='CreditScore',y='Exited',data=df)
df2=df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(df2,annot=True)
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)