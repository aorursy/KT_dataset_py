
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
print(df_train.dtypes)

#for missing sum wrt to each columns
df_train.isnull().sum()
df_train.shape
#calculating mean mode median
col=df_train.columns
l=[]
for c in col:
    if((df_train.dtypes[c]=='float64' or df_train.dtypes[c]=='int64')  ) :#selecting only float nd int type col
        df=pd.DataFrame(df_train[c])
        df.fillna(0)
        l=df.values.T.tolist()
        print("mean for column"+str(c)+str(df.sum()/614))
        print("median for column"+str(df.median(axis=0,skipna=True )))
        print("mode for column"+str(df.mode())  )
        
#here i know that col no 6 , 7 nd 8 have their type int or float so this only this columns  null values can be replaced by their mean mode or median
#===>selecting column with atleast one null and int or float value 
#df will be datatset of those columns

col=df_train.columns
df=pd.DataFrame()
i=0
for c in col:
    if((df_train.dtypes[c]=='float64' or df_train.dtypes[c]=='int64') and (df_train.isnull().sum()[c]>0 ) ):#selecting only float nd int type col with null values 
        if(i==0):
            df=pd.DataFrame(df_train[c])
            i=i+1
        else:
            df[c]=df_train[c]
#replacing null values wih mean ,mode or median
#using imputer for histogram
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
mean=imp.fit(df).transform(df)#col list where null  reaplced with mean 
imp=Imputer(missing_values='NaN',strategy='median',axis=0)
median=imp.fit(df).transform(df)#col list where null  reaplced with median
imp=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
mode=imp.fit(df).transform(df)#col list where null  reaplced with mode
#plotting histograme of mean mode median of 'LoanAmount'
import matplotlib.pyplot as plt
plt.subplot(311)#for first column its histo with mean 
plt.hist(mean[:,0],bins=100,alpha=0.9,color='r',histtype='bar')
plt.hist(mode[:,0],bins=100,alpha=0.5,color='b',histtype='bar')
plt.hist(median[:,0],bins=100,alpha=0.8,color='k',histtype='bar')
plt.subplot(312)#for second column its histo with mean mode median
plt.hist(mean[:,1],bins=35,alpha=0.9,color='r',histtype='bar')
plt.hist(mode[:,1],bins=35,alpha=0.5,color='b',histtype='bar')
plt.hist(median[:,1],bins=35,alpha=0.1,color='g',histtype='bar')
plt.subplot(313)#for third column its histo with mean mode median
plt.hist(mean[:,2],bins=30,alpha=0.9,color='r',histtype='bar')
plt.hist(mode[:,2],bins=30,alpha=0.5,color='y',histtype='bar')
plt.hist(median[:,2],bins=30,alpha=0.1,color='g',histtype='bar')
plt.show()

#boxplot for int nd float dattatype column
df_train.boxplot(column=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'])


cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
df_train[cols].describe()
#printing oulier count wrt to (int or float) datat type columns
cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
for col in cols:
    q1=df_train[col].describe()[4]#25%
    q3=df_train[col].describe()[6]#75%
    inter_range=q3-q1
    A=q1-1.5*inter_range #lower outlier range
    B=q3+q1-1.5*inter_range#upper outlier range
    print("no of ouliers in column-"+str(col))
    print(str(df_train[(df_train[col]<A)+(df_train[col]>B)][col].count()))
#creating a series of  ApplicantIncome column 
col_series = df_train['ApplicantIncome']
q1=df_train['ApplicantIncome'].describe()[4]#25%
q2=df_train['ApplicantIncome'].describe()[5]#50%
q3=df_train['ApplicantIncome'].describe()[6]#75%
x1=col_series<q1
x2=(col_series>q1)*(col_series<q2)
x3=(col_series<q3)*(col_series>q2)
x4=col_series>q3
col_series[x1]='Lower Class'
col_series[x2]='Lower Middle Class'
col_series[x3]='Upper Middle Class'
col_series[x4]='Upper Class'
#finally replacing ApplicantIncome column with series col_series with  (Lower Class|Lower Middle Class|Upper Middle Class|Upper Class) 
df_train['ApplicantIncome']=col_series
df_train

