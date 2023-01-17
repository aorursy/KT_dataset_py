import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import classification_report

lba = pd.read_csv('../input/column_2C_weka.csv')
lba.head()
#310 observations
lba.info()
#Checking missing values
lba.isnull().sum()
#converting class column - Abnormal =0 and Normal=1
def convert_class_numeric(col):
    if col == 'Abnormal':
        return 0
    else:
        return 1
lba['class'] = lba['class'].apply(lambda x:convert_class_numeric(x))
lba.head()

lba.tail()
lba['class'].value_counts()
sns.countplot(x='class',data=lba ,palette='Paired')
#This gives a snapshot of correlation between varaibles.
sns.pairplot(data=lba)
#One of the check for Logistic regression is independent variables should be independent of each other
#As I mentioned earlier, after my research i found out that, PI(Pelvic Incidence)= pelvic_tilt(PT) + sacral_slope(SS)
#Due to this reason,I have dropped the pelvic_tilt numeric and sacral_slope attributes of teh data set.
sns.scatterplot(x='pelvic_incidence',y=lba['pelvic_tilt numeric']+lba['sacral_slope'],data=lba)
# Based on research, i found that Pelvic Incidence (PI) and Pelvic radius(PR) are two techniques used to figure spine abnormalities
#For PR techinque needs attributes like hip angle(HA), pelvic angle(PA) .With this it seems like we can disregard, Pelvic radius column and 
#also PR is used in calculating the lumbar_lordosis_angle. 
#This leaves us with degree of degree_spondylolisthesis attribute, it seems like it is being used in classification of spine as Abnormal or Normal
#There is a correlation between lumbar_lordosis_angle and sacral slope.
sns.scatterplot(x='lumbar_lordosis_angle',y='sacral_slope',data=lba)
sns.scatterplot(x='pelvic_incidence',y='class',data=lba)
sns.scatterplot(x='degree_spondylolisthesis',y='class',data=lba)

lba.columns
#Trainign the model with pelvic_incidence and class

X1=lba[['pelvic_incidence']]
y=lba['class']

#Tranining the model1
X1_train, X1_test, y_train, y_test = train_test_split(X1,y, test_size=0.30, random_state=101)
print(X1_train.head())
print(X1_train.shape)
print(y_train.head())
print(y_test.shape)
logmodel = LogisticRegression()
logmodel.fit(X1_train,y_train)
#Predictions
predictions1 = logmodel.predict(X1_test)
logit_model=sm.Logit(y,X1)
result1=logit_model.fit()
print(result1.summary2())
#Evaluation
print(classification_report(y_test,predictions1))
X2=lba[['degree_spondylolisthesis']]
y=lba['class']
X2_train, X2_test, y_train2, y_test2 = train_test_split(X2,y, test_size=0.30, random_state=101)
#Creating instance of LogRegression and fittign the model
logmodel2 = LogisticRegression()
logmodel2.fit(X2_train,y_train2)
predictions2 = logmodel2.predict(X2_test)
#Evaluations
logit_model2=sm.Logit(y,X2)
result2=logit_model2.fit()
print(result2.summary2())
print(classification_report(y_test2,predictions2))
X3=lba[['lumbar_lordosis_angle','pelvic_radius', 'degree_spondylolisthesis']]
y=lba['class']
X3_train, X3_test, y_train3, y_test3 = train_test_split(X3,y, test_size=0.30, random_state=101)
#Creating instance of LogRegression and fittign the model
logmodel3 = LogisticRegression()
logmodel3.fit(X3_train,y_train3)
predictions3 = logmodel3.predict(X3_test)
#Evaluations
logit_model3=sm.Logit(y,X3)
result3=logit_model3.fit()
print(result3.summary2())
print(classification_report(y_test3,predictions3))
X4=lba[['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle',
       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']]
y=lba['class']
X4_train, X4_test, y_train4, y_test4 = train_test_split(X4,y, test_size=0.30, random_state=101)
#Creating instance of LogRegression and fittign the model
logmodel4 = LogisticRegression()
logmodel4.fit(X4_train,y_train4)
predictions4 = logmodel4.predict(X4_test)
#Evaluations
logit_model4=sm.Logit(y,X4)
result4=logit_model4.fit()
print(result4.summary2())
print(classification_report(y_test4,predictions4))
