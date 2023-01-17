import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()
data.columns
data = data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
data.columns
data['MaritalStatus'].unique()
data.loc[data['Attrition']=='No','Attrition'] = 0
data.loc[data['Attrition']=='Yes','Attrition'] = 1
data.head()
data['Business_Travel_Rarely']=0
data['Business_Travel_Frequently']=0
data['Business_Non-Travel']=0

data.loc[data['BusinessTravel']=='Travel_Rarely','Business_Travel_Rarely'] = 1
data.loc[data['BusinessTravel']=='Travel_Frequently','Business_Travel_Frequently'] = 1
data.loc[data['BusinessTravel']=='Non-Travel','Business_Non-Travel'] = 1
data['Life Sciences']=0
data['Medical']=0
data['Marketing']=0
data['Technical Degree']=0
data['Education Human Resources']=0
data['Education_Other']=0

data.loc[data['EducationField']=='Life Sciences','Life Sciences'] = 1
data.loc[data['EducationField']=='Medical','Medical'] = 1
data.loc[data['EducationField']=='Other','Education_Other'] = 1
data.loc[data['EducationField']=='Technical Degree','Technical Degree'] = 1
data.loc[data['EducationField']=='Human Resources','Education Human Resources'] = 1
data.loc[data['EducationField']=='Marketing','Marketing'] = 1
data['Sales']=0
data['R&D']=0
data['Dept_Human Resources'] =0

data.loc[data['Department']=='Sales','Sales'] = 1
data.loc[data['Department']=='Research & Development','R&D'] = 1
data.loc[data['Department']=='Human Resources','Dept_Human Resources'] = 1
data.loc[data['Gender']=='Male','Gender'] = 1
data.loc[data['Gender']=='Female','Gender'] = 0
data['Research Scientist']=0
data['Laboratory Technician']=0
data['Sales Executive']=0
data['Manufacturing Director']=0
data['Healthcare Representative']=0
data['Sales Representative']=0
data['Research Director']=0
data['Manager'] = 0
data['Job_Human_Resources'] = 0

data.loc[data['JobRole']=='Research Scientist','Research Scientist'] = 1
data.loc[data['JobRole']=='Laboratory Technician','Laboratory Technician'] = 1
data.loc[data['JobRole']=='Sales Executive','Sales Executive'] = 1
data.loc[data['JobRole']=='Sales Representative','Sales Representative'] = 1
data.loc[data['JobRole']=='Manufacturing Director','Manufacturing Director'] = 1
data.loc[data['JobRole']=='Healthcare Representative','Healthcare Representative'] = 1
data.loc[data['JobRole']=='Research Director','Research Director'] = 1
data.loc[data['JobRole']=='Manager','Manager'] = 1
data.loc[data['JobRole']=='Human Resources','Job_Human_Resources'] = 1
data.head()
data['Marital_single']=0
data['Marital_married']=0
data['Marital_divorced']=0

data.loc[data['MaritalStatus']=='Married','Marital_married'] = 1
data.loc[data['MaritalStatus']=='Single','Marital_single'] = 1
data.loc[data['MaritalStatus']=='Divorced','Marital_divorced'] = 1
data.loc[data['OverTime']=='No','OverTime'] = 0
data.loc[data['OverTime']=='Yes','OverTime'] = 1
data.head()
data.columns
data = data.drop(['BusinessTravel','EducationField',
                        'Department','JobRole','MaritalStatus'],axis=1)
data.head()
data.dtypes
data['Attrition'] = data['Attrition'].astype('int')
data['Gender'] = data['Gender'].astype('int')
data['OverTime'] = data['OverTime'].astype('int')
data.corr()
from sklearn.cross_validation import train_test_split
#from random import seed

#seed(20)
train_x = data.drop(['Attrition'],axis=1)
train_y = data['Attrition']

X,test_x,Y,test_y = train_test_split(train_x, train_y, test_size=0.3,random_state=20)
len(test_x)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X,Y)
from sklearn.metrics import accuracy_score

pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y, normalize=True, sample_weight=None)
accuracy
from sklearn.metrics import classification_report

print(classification_report(test_y, pred_y))
from sklearn.feature_selection import RFE

rfe = RFE(clf,40)
rfe = rfe.fit(train_x,train_y)
print(rfe.support_)
print(rfe.ranking_)
X =rfe.transform(X)
test_x = rfe.transform(test_x)
X.shape
from sklearn.metrics import accuracy_score

clf.fit(X,Y)
pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y, normalize=True, sample_weight=None)
accuracy
from sklearn.metrics import classification_report

print(classification_report(test_y, pred_y))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf.fit(X,Y)
from sklearn.metrics import accuracy_score

pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y, normalize=True, sample_weight=None)
accuracy
from sklearn.metrics import classification_report

print(classification_report(test_y, pred_y))

