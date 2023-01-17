# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_excel('/kaggle/input/the-interview-attendance-problem/Interview.xlsx')
data.head(3)
data.columns
data.isnull().sum()
cols=['Date', 'Client name', 'Industry', 'Location', 'Position_Type',

       'SkillSet_Name', 'Interview_type', 'Candidate_ID', 'Gender',

       'Current_Location', 'Company_Location', 'Interview_Venue',

       'Candidate_Hometown', 'Permission_Obtained', 'No_Unscheduled_Meeting',

       'Follow-Up_Call_OK', 'Alternate_Number_Given', 'CV_JD_Ready',

       'Venue_Clear', 'Call_Letter_Received', 'Expected_Attendance',

       'Observed_Attendance', 'Marital_Status']

data1=data[cols]

data1.columns
data1.shape
data1.dtypes
data1[data1['Gender'].isnull()]
data1=data1.dropna(thresh=10)
def levels(df):

    return (pd.DataFrame({'dtype':df.dtypes, 

                         'levels':df.nunique(), 

                         'levels':[df[x].unique() for x in df.columns],

                         'null_values':df.isna().sum(),

                         'unique':df.nunique()}))
levels(data1)
data1['Permission_Obtained'].value_counts()
data1['Permission_Obtained'][data1['Permission_Obtained']=='Yet to confirm']='No'

data1['Permission_Obtained'][data1['Permission_Obtained']=='Not yet']='No'

data1['Permission_Obtained'][data1['Permission_Obtained'].isnull()]='Na'
data1['No_Unscheduled_Meeting'].value_counts()
data1['No_Unscheduled_Meeting'][data1['No_Unscheduled_Meeting'].isnull()]='Na'

data1['No_Unscheduled_Meeting'][(data1['No_Unscheduled_Meeting']!='Yes')&

                               (data1['No_Unscheduled_Meeting']!='No')&

                               (data1['No_Unscheduled_Meeting']!='Na')]='NS'
data1['Follow-Up_Call_OK'].value_counts()
data1['Follow-Up_Call_OK'][data1['Follow-Up_Call_OK']=='No Dont']='No'

data1['Follow-Up_Call_OK'][data1['Follow-Up_Call_OK'].isnull()]='Na'
data1['Alternate_Number_Given'].value_counts()
data1['Alternate_Number_Given'][data1['Alternate_Number_Given']=='No I have only thi number']='No'

data1['Alternate_Number_Given'][data1['Alternate_Number_Given']=='na']='Na'

data1['Alternate_Number_Given'][data1['Alternate_Number_Given'].isnull()]='Na'
data1['CV_JD_Ready'].value_counts()
data1['CV_JD_Ready'][data1['CV_JD_Ready'].isnull()]='Na'

data1['CV_JD_Ready'][data1['CV_JD_Ready']=='na']='Na'

data1['CV_JD_Ready'][(data1['CV_JD_Ready']!='Yes')&

                               (data1['No_Unscheduled_Meeting']!='No')&

                               (data1['No_Unscheduled_Meeting']!='Na')]='No'
data1['CV_JD_Ready'].value_counts()
print(data1['Venue_Clear'].value_counts())

print(data1['Call_Letter_Received'].value_counts())
data1['Venue_Clear'].replace('na','Na',inplace=True)

data1['Venue_Clear'].replace('No- I need to check','No',inplace=True)

data1['Venue_Clear'][data1['Venue_Clear'].isnull()]='Na'
data1['Call_Letter_Received'].replace(['Not Sure','Not sure',

                                      'Need To Check','Not yet',

                                      'Havent Checked','Yet to Check'],'NS',inplace=True)
data1['Call_Letter_Received'][data1['Call_Letter_Received'].isnull()]='Na'

data1['Call_Letter_Received'].replace(['na'],'Na',inplace=True)
print(data1['Venue_Clear'].value_counts())

print(data1['Call_Letter_Received'].value_counts())
data1['Expected_Attendance'].value_counts()
data1['Expected_Attendance'][data1['Expected_Attendance'].isnull()]='Uncertain'
data1['Observed_Attendance'].replace('Yes ','Yes',inplace=True)

data1['Observed_Attendance'].replace('No ','No',inplace=True)
data1['Observed_Attendance'].value_counts()
data1['Client name'].value_counts()
cols=['Client name','Industry','Location']

print(data1[cols][data1['Client name']=='Aon hewitt Gurgaon'].head(2))

print(data1[cols][data1['Client name']=='Aon Hewitt'].head(2))

print(data1[cols][data1['Client name']=='Standard Chartered Bank'].head(2))

print(data1[cols][data1['Client name']=='Standard Chartered Bank Chennai'].head(2))
data1['Client name'][data1['Client name']=='Aon hewitt Gurgaon']='Aon Hewitt'

data1['Industry'].replace(['IT Products and Services','IT Services'],'IT',inplace=True)

data1['Location'].replace('Gurgaonr','Gurgaon',inplace=True)
data1['Interview_type'][data1['Interview_type']!='Walkin']='Scheduled'
data1['SkillSet_Name'].value_counts().head(50)
data1['Skill']=pd.Series()

data1['Skill'][data1['SkillSet_Name'].str.contains("eveloper")]='Developer'

data1['Skill'][data1['SkillSet_Name'].str.contains("SAP")]='SAP'

data1['Skill'][data1['SkillSet_Name'].str.contains("naly")]='Analyst'

data1['Skill'][data1['SkillSet_Name'].str.contains("RA")]='RA'

data1['Skill'][data1['SkillSet_Name'].str.contains("roduc")]='Product'

data1['Skill'][data1['SkillSet_Name'].str.contains("SCC")]='SCCM'

data1['Skill'][data1['SkillSet_Name'].str.contains("Scc")]='SCCM'

data1['Skill'][data1['SkillSet_Name'].str.contains("sccm")]='SCCM'

data1['Skill'][data1['SkillSet_Name'].str.contains("abili")]='LL'

data1['Skill'][data1['SkillSet_Name'].str.contains("L & L")]='LL'

data1['Skill'][data1['SkillSet_Name'].str.contains("abli")]='LL'

data1['Skill'][data1['SkillSet_Name'].str.contains("Publi")]='Publishing'

data1['Skill'][data1['SkillSet_Name'].str.contains('esti')]='Testing'

data1['Skill'][data1['SkillSet_Name'].str.contains("ednet")]='Mednet'

data1['Skill'][data1['SkillSet_Name'].str.contains("Java")]='Java'

data1['Skill'][data1['SkillSet_Name'].str.contains("JAVA")]='JAVA'

data1['Skill'][data1['SkillSet_Name'].str.contains("Oracle")]='Oracle'

data1['Skill'][data1['SkillSet_Name'].str.contains("Fres")]='Fresher'

data1['Skill'][data1['SkillSet_Name'].str.contains("pera")]='Operations'

data1['Skill'][data1['SkillSet_Name'].str.contains("KYC")]='KYC'

data1['Skill'][data1['SkillSet_Name'].str.contains("Routi")]='Routine'

data1['Skill'][data1['SkillSet_Name'].str.contains("SAS")]='SAS'

data1['Skill'][data1['SkillSet_Name'].str.contains("anag")]='Manager'

data1['Skill'][data1['SkillSet_Name'].str.contains("iosim")]='Biosimilarity'

data1['Skill'][data1['SkillSet_Name'].str.contains("ot ")]='Dot Net'

data1['Skill'][data1['SkillSet_Name'].str.contains("Hadoop")]='Hadoop'

data1['Skill'][data1['SkillSet_Name'].str.contains("egula")]='Regulatory'

data1['Skill'][data1['SkillSet_Name'].str.contains("EMEA")]='EMEA'

data1['Skill'][data1['SkillSet_Name'].str.contains("COTS")]='COTS'

data1['Skill'][data1['SkillSet_Name'].str.contains("ETL")]='ETL'

data1['Skill'][data1['SkillSet_Name'].str.contains("abel")]='Labelling'

data1['Skill'][data1['Skill'].isnull()]='Other'
data1[data1['Industry'].str.contains('harm')].tail(50)
data2 = data1.drop(['Date','SkillSet_Name'],axis=1)

data2 = data2.set_index('Candidate_ID')
levels(data2)
data2['Observed_Attendance'].replace("Yes",1,inplace=True)

data2['Observed_Attendance'].replace("No",0,inplace=True)
cols = data2.columns

for col in cols:

    data2[col]=data2[col].astype('category')
x = data2.copy().drop("Observed_Attendance",axis=1)

y = data2["Observed_Attendance"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 200)
from sklearn.preprocessing import LabelEncoder

cols = ['Skill','Candidate_Hometown','Client name','Location','Current_Location']

import bisect

for col in cols:

    le = LabelEncoder()

    x_train[col] = le.fit_transform(x_train[col])

    x_test[col] = x_test[col].map(lambda s: 'other' if s not in le.classes_ else s)

    le_classes = le.classes_.tolist()

    bisect.insort_left(le_classes, 'other')

    le.classes_ = le_classes

    x_test[col] = le.transform(x_test[col])
oth = set(x.columns)-set(cols)

x_train = pd.get_dummies(x_train,columns=oth,drop_first=True)

x_test = pd.get_dummies(x_test,columns=oth,drop_first=True)
x_train_cols = x_train.columns

x_test_cols = x_test.columns



common_cols = x_train_cols.intersection(x_test_cols)

train_not_test = x_train_cols.difference(x_test_cols)



train_not_test
x_train.shape,x_test.shape
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=200, max_iter=2000,solver='sag')

LR.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, LR.predict(x_test))

print("Accuracy on train is:",accuracy_score(y_train,LR.predict(x_train)))

print("Accuracy on test is:",accuracy_score(y_test,LR.predict(x_test)))

from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(y_train, LR.predict_proba(x_train)[:,1])

roc_auc = auc(fpr, tpr)



import matplotlib.pyplot as plt

%matplotlib notebook

# plt.figure()

plt.plot(fpr,tpr,color='orange', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc="lower right")
from sklearn.neighbors import KNeighborsClassifier # kNN classifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier(algorithm = "brute")

params = {"n_neighbors": [3,5,7],"metric": ["euclidean", "cityblock"],'weights':['uniform','distance']}

grid = GridSearchCV(knn,param_grid=params,scoring="accuracy",cv=5)

grid.fit(x_train,y_train)
grid.cv_results_
knn_best = grid.best_estimator_
cm = confusion_matrix(y_test, knn_best.predict(x_test))

print("Accuracy on train is:",accuracy_score(y_train,knn_best.predict(x_train)))

print("Accuracy on test is:",accuracy_score(y_test,knn_best.predict(x_test)))
knn_best
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc

dtc.fit(x_train, y_train)
cm = confusion_matrix(y_test, dtc.predict(x_test))

print("Accuracy on train is:",accuracy_score(y_train,dtc.predict(x_train)))

print("Accuracy on test is:",accuracy_score(y_test,dtc.predict(x_test)))
parameters={'max_depth':range(1,10)}

dtc_grid = GridSearchCV(DecisionTreeClassifier(),param_grid=parameters,n_jobs=-1,cv=10)

dtc_grid.fit(x_train,y_train)

print(dtc_grid.best_score_)

print(dtc_grid.best_params_)
dtc_grid.cv_results_
dtc2=DecisionTreeClassifier(max_depth=1)

dtc2.fit(x_train,y_train)

cm = confusion_matrix(y_test, dtc.predict(x_test))

print("Accuracy on train is:",accuracy_score(y_train,dtc2.predict(x_train)))

print("Accuracy on test is:",accuracy_score(y_test,dtc2.predict(x_test)))
from xgboost import XGBClassifier

xgbc = XGBClassifier()

print(xgbc)

xgbc1=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.009,

       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,

       n_estimators=100, n_jobs=1, nthread=None,

       objective='binary:logistic', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,

       subsample=1, verbosity=1)
xgbc.fit(x_train,y_train)

xgbc1.fit(x_train,y_train)
cm = confusion_matrix(y_test, xgbc.predict(x_test))

print("Accuracy on train is:",accuracy_score(y_train,xgbc.predict(x_train)))

print("Accuracy on test is:",accuracy_score(y_test,xgbc.predict(x_test)))
cm = confusion_matrix(y_test, xgbc1.predict(x_test))

print("Accuracy on train is:",accuracy_score(y_train,xgbc1.predict(x_train)))

print("Accuracy on test is:",accuracy_score(y_test,xgbc1.predict(x_test)))