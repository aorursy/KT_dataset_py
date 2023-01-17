# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

dataset=pd.read_csv('../input/train_AV3.csv')
#dataset_test=pd.read_csv('../input/test.csv')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
dataset

dataset.isnull().sum()
#by analysing the dataset we get to know that from those who having NaN value only two columns(LoanAmount and Loan_Amount_Term )  are changable with mean,mode and median.
#we will Impute the values of these two columns with mean,median and mode of the columns.
x=dataset.iloc[:, 8:10].values
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
y_mean=imputer.fit(x).transform(x)
imputer1 = Imputer(missing_values='NaN',strategy='median',axis=0)
y_median=imputer1.fit(x).transform(x)
imputer2 = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
y_mode=imputer2.fit(x).transform(x)
# here we creating a imputer object for replacing NaN  values with mean,mode and median



y_median

#here we got the replaced columns which we will use in creating graphs b/w mean,median,mode
# we are ploting the replaced-col for the "LoanAmount" col of main dataset
plt.hist(y_mean[:, 0],bins=80,color='r',histtype='bar')
plt.hist(y_mode[:, 0],bins=80,color='y',histtype='bar')
plt.hist(y_median[:, 0],bins=80,alpha=0.7,color='k',histtype='bar')
plt.title('LoanAmount Visualisation')
plt.show()
# we are ploting the replaced-col for the "Loan_Amount_Term" col of main dataset
plt.hist(y_mean[:, 1],bins=30,color='r',histtype='bar')
plt.hist(y_mode[:, 1],bins=30,alpha=0.5,color='k',histtype='bar')
plt.hist(y_median[:, 1],bins=30,alpha=0.5,color='y',histtype='bar')
plt.title('Loan_Amount_Term Visualisation')
plt.show()
dataset.boxplot(column=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'])
#plotted the boxplot for
column=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
dataset[column].describe()
#created a function for outlier dection 
def outlier_detector(column):
    for i in column:
        q1=dataset[i].describe()[4]
        q3=dataset[i].describe()[6]
        iqr=q3-q1
        a=q1-(1.5*iqr)
        b=q3+1.5*iqr
        x= dataset[i]
        outlier=(x>b)+(x<a)
        print("no. of ouliers in column '"+str(i)+"': "+str(dataset[outlier][i].count()))
# showing outlier values in columns having int values
outlier_detector(column)
x = dataset['ApplicantIncome']
#collecting the particular column 
q1=dataset['ApplicantIncome'].describe()[4]
q3=dataset['ApplicantIncome'].describe()[6]
q2=dataset['ApplicantIncome'].describe()[5]
y1=(x<q1)
y2=(x>q1)*(x<q2)
y3=(x>q2)*(x<q3)
y4=(x>q3)
x[y1] = "Lower Class"
x[y2] = "Lower Middle Class"
x[y3] = "Middle Class"
x[y4] = "Upper Class"
x
#updated column

dataset['ApplicantIncome'] = x
#updating the 'ApplicantIncome' column
dataset

