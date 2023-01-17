# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
student_data = pd.read_csv("../input/xAPI-Edu-Data.csv")
student_data.shape
student_data.describe()
print("Any data missing or having a null value?:",student_data.isnull().sum().any())
print("Unique values in each column\n")

print(student_data.nunique())
ax = student_data['gender'].value_counts().plot(kind='bar', figsize=(8,8),

                                                   fontsize=8)



ax.set_ylabel("")

ax.set_yticks([sum(student_data['gender']=='M'),sum(student_data['gender']=='F')])

ax.set_xticklabels(["M","F"], rotation=0, fontsize=15)

ax.set_title("Percentage of Boys and Girls")



totals = []



for i in ax.patches:

    totals.append(i.get_height())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_x()+0.15, i.get_height()-15,str(round((i.get_height()/total)*100, 2))+'%', rotation=45,fontsize=12,color='white')

plt.figure(figsize=(8,8))

plt.title("Comparing school attnedance of Boys and Girls")

bar = sns.countplot(hue='StudentAbsenceDays',x='gender',data=student_data)

plt.yticks([])

plt.ylabel("")

for p in bar.patches:

    bar.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))
result=[]

for k in student_data['Class']:

    if k in ['L']:

        result.append('Fail')

    else:

        result.append('Pass')



student_data['Result'] = result



#student_data = student_data.drop('Class',axis=1)



bar = sns.countplot(hue='Result',x='gender',data=student_data)

plt.title("Result Vs Gender")    

plt.ylabel("")

plt.yticks([])



for p in bar.patches:

    bar.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))

plt.title("School level Vs Gender ")

bar = sns.countplot(hue='StageID',x='gender',data=student_data)

plt.ylabel("")

plt.yticks([])

for p in bar.patches:

    bar.annotate(p.get_height(), (p.get_x()+0.1, p.get_height()+1))
result=[]

average = np.mean(student_data['raisedhands'])



for k in student_data['raisedhands']:

    if k > average :

        result.append('Active')

    else:

        result.append('Quiet')



student_data['Participation'] = result



#student_data = student_data.drop('raisedhands',axis=1)



plt.title("Participation Vs Gender ")

bar = sns.countplot(hue='Participation',x='gender',data=student_data)

plt.ylabel("")

plt.yticks([])

for p in bar.patches:

    bar.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))
plt.title(" Parent response on School")



bar = sns.countplot(hue='ParentschoolSatisfaction',x='ParentschoolSatisfaction',data=student_data)

plt.ylabel("")

plt.yticks([])



for p in bar.patches:

    bar.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

Features = pd.DataFrame()

Features['Gender'] = label.fit_transform(student_data['gender'])

Features['StageID'] = label.fit_transform(student_data['StageID'])

Features['Raisedhands'] = label.fit_transform(student_data['raisedhands'])

Features['Discussion'] = label.fit_transform(student_data['Discussion'])

Features['StudentAbsenceDays'] = label.fit_transform(student_data['StudentAbsenceDays'])



Target = label.fit_transform(student_data['Result'])
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2)
log_model = LogisticRegression()

log_model.fit(X_train,y_train)

y_pred = log_model.predict(X_test)

log_model_accuracy = accuracy_score(y_test,y_pred)

print("Logistic Model Acuracy = ",log_model_accuracy)