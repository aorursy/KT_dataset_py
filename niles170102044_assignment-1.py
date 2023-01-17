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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train=pd.read_csv('../input/train.csv')
#Reading Data
train.head()
train.dtypes
#Structure Of Dataset
train.describe()
#A summary
train.shape
#Dimensions of dataset
train['ApplicantIncome'].median()
train['ApplicantIncome'].mode()
#Computing mean,median, mode
#Can be done simultaneously with the agg function
train.isnull().sum()
#Knowing the amount of missing values in each coloumn
train.dropna(how='any').shape
#Shape after dropping the row with any missing value
import matplotlib.pyplot as plt
Income=train.ApplicantIncome
bins= 50
plt.hist(Income,bins,histtype='bar')
plt.show()
#Histogram showing ApplicantIncome
depend=train.Dependents
bins= 5
plt.hist(depend,bins,histtype='bar')
plt.show()
#Histogram showing no. of dependents
train.Property_Area.value_counts().plot(kind='bar')
#Graph showing where people hold their property
train.Education.value_counts().plot(kind='bar')
#Education level of people taking loans
train.Self_Employed.value_counts().plot(kind='bar')
#Employment type of people taking loans
train.Gender.value_counts().plot(kind='bar')
#Gender of people taking loans
train.LoanAmount.plot(kind='hist')
#loan amount histogram
plt.bar(train.Self_Employed, train.ApplicantIncome)
plt.show()
#Income of self employed and non self employed people
lower_class= train.ApplicantIncome<=2877.50
lower_middle_class=((train.ApplicantIncome>2877.50)&(train.ApplicantIncome<=3812.50))
upper_middle_class=((train.ApplicantIncome>3812.50)&(train.ApplicantIncome<=5795.0))
upper_class=(train.ApplicantIncome>5795.0)
#Classification according to Applicantincome
train[upper_class].head()
train.ApplicantIncome.describe()
import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
#Detecting outliers
outliers_iqr(train.ApplicantIncome)


plt.boxplot(train['ApplicantIncome'])
trainmed=train.fillna(train.median())
#Imputation of missing values
plt.boxplot(trainmed['LoanAmount'])
train2=pd.concat([train["Gender"]=="Male",train["Loan_Status"]=='Y'],axis=1)
#Male having accepted loan requests
sum(train2["Gender"] & train2["Loan_Status"])
train2=pd.concat([train["Gender"]=="Female",train["Loan_Status"]=='Y'],axis=1)
sum(train2["Gender"] & train2["Loan_Status"])
#Female having accepted loan requests
train.Gender.value_counts()
