import pandas as pd
import numpy as np
train=pd.read_csv("../input/train_AV3.csv")
test=pd.read_csv("../input/test_AV3.csv")
train.shape
test.shape
train.head()
test.head()
train.isnull().sum()
train.describe()
import matplotlib.pyplot as plt
train.plot()
plot.show()
train["ApplicantIncome"].plot()
test.isnull().sum()
#Checking Null Values
train.dtypes
train["Loan_Status"].replace("Y",1,inplace=True)
train["Loan_Status"].replace("N",0,inplace=True)
train.head()
train.describe()
train_drop=train.dropna()
train_drop.shape
train_drop.describe()
train.describe()
train_mean=train.fillna(train.mean())
#imputing values via mean
train.isnull().sum()
train_median=train.fillna(train.median())
#Imputing values via Median
plt.boxplot(train_median["LoanAmount"])
c=pd.concat([train["Gender"]=="Male",train["Loan_Status"]==1],axis=1)
sum(c["Gender"] & c["Loan_Status"])
c.sum()
train_income=pd.qcut(train["ApplicantIncome"],4,["Lower","L_Middle","U_Middle","Upper"])
#Classifying applicant income
train_income.head()
train2 = pd.concat([train,train_income], axis=1)
#making a new dataframe containing class of income
train2.head()
d=train_income=="Lower"
sum(d & train["Loan_Status"])
sum(d)
train_income.value_counts()
train2.describe()
train.describe()
c=list(train.columns.values)
c.append("class")
train2.columns=c
train2.head()
plt.scatter(train2["class"],train2["Loan_Status"])
#shows that every class of income have loan status approved and unaaproved
train2[(train2["class"]=="Upper") & (train2["Loan_Status"]==0)]
#trying relating income class and loan status
plt.scatter(train2["LoanAmount"],train2["Loan_Status"])
#no direct relation bw loan amount and loan status
train2["Loan_Status"].value_counts()
train2["Property_Area"].value_counts()
sum((train2["Property_Area"]=="Rural") & (train2["Loan_Status"]==0))
#No direct relation between rural propeerties and loan status no
train2["Credit_History"].value_counts(dropna=False)
sum((train2["Credit_History"]==0) & (train2["Loan_Status"]==0))
#A relation is seen here as out of 89 person with 0 Credit History 82 were not approved the loan
sum((train2["Credit_History"]==1) & (train2["Loan_Status"]==1))
#from above two data it is seen that Credit History has a major impact on Loan Stautus (378/422=90%)
y1=[sum((train2["Credit_History"]==0) & (train2["Loan_Status"]==1)),sum((train2["Credit_History"]==1) & (train2["Loan_Status"]==1))]
y2=[sum((train2["Credit_History"]==0) & (train2["Loan_Status"]==0)),sum((train2["Credit_History"]==1) & (train2["Loan_Status"]==0))]
p1=plt.bar([0,1],y1,width=0.5)
p2=plt.bar([0,1],y2,width=0.5,bottom=y1,color='#d62728')
plt.xlabel("Credit History")
plt.ylabel("No of Approved vs Unapproved loans")
plt.legend((p1, p2), ('Approved', 'Unapproved'))
plt.show()
#High Dependence on Credit History can be seen
plt.boxplot(train_mean["ApplicantIncome"])
train2.describe()
iqr=train["ApplicantIncome"].quantile(q=0.75)-train["ApplicantIncome"].quantile(q=0.25)
range1=iqr*1.5
train[(train["ApplicantIncome"]>(train["ApplicantIncome"].quantile(q=0.75)+range1)) | (train["ApplicantIncome"]<(train["ApplicantIncome"].quantile(q=0.25)-range1))]
#outlier based on applicant Income
iqr2=train_mean["LoanAmount"].quantile(q=0.75)-train["LoanAmount"].quantile(q=0.25)
range2=iqr2*1.5
train_mean[(train_mean["LoanAmount"]>(train_mean["LoanAmount"].quantile(q=0.75)+range1)) | (train["LoanAmount"]<(train["LoanAmount"].quantile(q=0.25)-range1))]
#no outlier based on Loan Amount
import sklearn
from sklearn.cluster import DBSCAN
e=list(train2.describe().columns)
e
e
data=train_mean[e]
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data)
c=pd.DataFrame(x_scaled)
model_mean2=DBSCAN(eps=0.1,min_samples=9).fit(c)
sum(model_mean2.labels_==-1)
outliers_mean=train2[model_mean2.labels_==-1]
outliers_mean
sum((outliers_mean["Credit_History"]==0) & (outliers_mean["Loan_Status"]==1))
#Shows that those loan status and credit history were not matching are hypothesis were outliers(7 out of 7)
sum((outliers_mean["Credit_History"]==1) & (outliers_mean["Loan_Status"]==0))
#here also outliers occupy considerable percentage (31 out of 97)
