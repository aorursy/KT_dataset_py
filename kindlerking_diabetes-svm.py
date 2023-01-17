
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import imblearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.describe()
df.head(5)
#VIZUALIZATION NO. 1
df['Outcome'].value_counts().plot(kind='pie',autopct='%.1f%%',radius=1.5)
#VIZUALISATION NO.2
l=[]
l2=[]
not_obese=0
obese=0
very_obese=0
for x in df['Outcome'].index:
    if df['Outcome'][x] == 1:
        l.append(df['BMI'][x])

for j in l:
    temp=int(j)
    l2.append(temp)

for i in l2:
    if i in l2 and i in range(1,26):
        not_obese=not_obese+1
    elif i in l2 and i in range(25,31):
        obese=obese+1
    else:
        very_obese=very_obese+1

data=[not_obese,obese,very_obese]
labels=['not obese','obese','very obese']

plt.pie(data,labels=labels,autopct='%.1f%%',colors=['r','g','b'],radius=1.5)

x=df['DiabetesPedigreeFunction']
y=df['Outcome']
plt.scatter(x,y)
#CLEANING THE 0 values in the data by replacing them with the non zero mean of the data
nonzero_mean_gl = df['Glucose'][ df['Glucose'] != 0 ].mean()
df['Glucose']=df['Glucose'].replace(0,nonzero_mean_gl,inplace=False)
nonzero_mean_bp=  df['BloodPressure'][ df['BloodPressure'] != 0 ].mean()
df['BloodPressure']=df['BloodPressure'].replace(0,nonzero_mean_bp,inplace=False)
nonzero_mean_st= df['SkinThickness'][df['SkinThickness'] != 0].mean()
df['SkinThickness']=df['SkinThickness'].replace(0,nonzero_mean_st)
nonzero_mean_in= df['Insulin'][df['Insulin'] != 0].mean()
df['Insulin']=df['Insulin'].replace(0,nonzero_mean_in)
nonzero_mean_bmi= df['BMI'][df['BMI'] != 0].mean()
df['BMI']=df['BMI'].replace(0,nonzero_mean_bmi)

df.describe()
#ALL The COLUMNS [Glucose - BMI] have been replaced with non-zero means wherever 0 was encountered
#The final step of data pre processing involves dealing with the class imbalance problem in the data by oversampling or undersampling techniques

classifier=svm.SVC(kernel='linear',C=100,gamma='auto',degree=3)

#x is the input that we are going to feed into our data and y is the output such that 1 signifies Patient has diabetes 0 signifies patient dosent have diabetes
X=df.iloc[: , :-1]
y=df.iloc[:, -1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Without SMOTE
pipeline=make_pipeline(classifier)
model=pipeline.fit(x_train,y_train)
model_pred=model.predict(x_test)


#with SMOTE
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4),classifier)
smote_model = smote_pipeline.fit(x_train,y_train)
smote_pred = smote_model.predict(x_test)
#without SMOTE
from sklearn.metrics import classification_report
print(classification_report(y_test,model_pred))
#with SMOTE 
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test,smote_pred))
