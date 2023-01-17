# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading the data



df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')



df.head()
df.shape
df.info()
# changing null values in the salary column to zero



df['salary'].fillna(0,inplace=True)



df.info()
#Dropping the variable sl_no since it has no impact on the dependent variable



df.drop('sl_no',axis=1,inplace=True)
#Checking the distribution of the data



df.describe()
#Checking for duplicate rows



df.loc[df.duplicated()]
#plotting the distribution plot



df_num = df.select_dtypes(include=[np.number])



col_num = list(df_num.columns)



c = len(col_num)

m = 1

n = 0



plt.figure(figsize=(20,30))



for i in col_num:

  if m in range(1,c+1):

    plt.subplot(8,4,m)

    sns.distplot(df_num[df_num.columns[n]])

    m=m+1

    n=n+1



plt.show()
#Plotting the pairplot



sns.heatmap(df.corr(),linewidth=0.5,cmap='YlGnBu',annot=True)

plt.show()
df.info()
df.ssc_b.replace('Others','sscb_other',inplace=True)

df.hsc_b.replace('Others','hscb_other',inplace=True)

df.degree_t.replace('Others','deg_other',inplace=True)
df.head()
# Function for creating dummy variables for categorical variables



def dummy(x,df):

    temp = pd.get_dummies(df[x],drop_first = True)

    df =pd.concat([df,temp],axis=1)

    df.drop(x,axis=1,inplace=True)

    return df



#Getting dummy variables for the categorical variables in df

df = dummy('status',df)

df = dummy('specialisation',df)

df = dummy('workex',df)

df = dummy('degree_t',df)

df = dummy('hsc_s',df)

df = dummy('hsc_b',df)

df = dummy('ssc_b',df)

df = dummy('gender',df)
df.head()
df.rename(columns={'Yes':'Workex','M':'Male'},inplace=True)

df.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(1001)



df_train,df_test = train_test_split(df,test_size=0.2,random_state=100)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

df_train[col_num] = scaler.fit_transform(df_train[col_num])



y_train = df_train['mba_p']

X_train = df_train[['ssc_p','hsc_p']]
import statsmodels.api as sm



#Adding constant to X_train since by default statsmodel fits a regression line passing through the origin.

X_train = sm.add_constant(X_train)



#Fitting linear model



lm = sm.OLS(y_train,X_train).fit()



#printing paramaters



print(lm.params)



#Printing the summary



print(lm.summary())
y_train2 = df_train['mba_p']

X_train2 = df_train[['ssc_p','degree_p']]
#Adding constant

X_train2 = sm.add_constant(X_train2)



#fitting linear model



lm2 = sm.OLS(y_train2,X_train2).fit()
#Printing model parameters



print(lm2.params)



#printing model summary

print(lm2.summary())
y_train3 = df_train['mba_p']

X_train3 = df_train[['hsc_p','degree_p']]
#Adding constant to X_tarin3



X_train3 = sm.add_constant(X_train3)



#fitting the linear model



lm3 = sm.OLS(y_train3,X_train3).fit()
print(lm3.params)

print(lm3.summary())
y_train4 = df_train['mba_p']

X_train4 = df_train[['ssc_p','hsc_p','degree_p']]
#Adding constant to X_train4



X_train4= sm.add_constant(X_train4)



#Fitting the linear model



lm4 = sm.OLS(y_train4,X_train4).fit()
#Printing coefficients and statistical summary

print(lm4.params)



print(lm4.summary())
#Predicting on the train data



y_train_pred = lm4.predict(X_train4)
# Plotting the histogram of the error terms



fig = plt.figure()

sns.distplot((y_train4 - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                

plt.xlabel('Errors', fontsize = 18)   

plt.show()
df_test.head()
#Transforming the numerical varianles of test data

df_test[col_num] = scaler.transform(df_test[col_num])
#Extracting X_test and y_test from the df_test  



X_test = df_test[['ssc_p','hsc_p','degree_p']]

y_test = df_test['mba_p']

#Adding constant



X_test = sm.add_constant(X_test)



#Predicting on th emodel



y_pred = lm4.predict(X_test)
#Evaluating R2 score on the predictions



from sklearn.metrics import r2_score



print(r2_score(y_test,y_pred))
# Plotting y_test and y_pred to understand the spread.



fig = plt.figure()

sns.scatterplot(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)   

plt.show()
#Putting y_test and y_pred to a dataframe.



compare_pred = pd.DataFrame(columns=['y_test','y_pred'])

compare_pred['y_test'] = y_test

compare_pred['y_pred'] = y_pred



compare_pred.head(10)