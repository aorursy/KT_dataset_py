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
data=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()
data.shape
data["Department"].unique()
count1=0

count2=0

count3=0

count4=0

count5=0

count6=0
for i in data.index:

    if(data["Department"][i]=='Sales' and data['Attrition'][i]=='Yes'):

        count1=count1+1;

    if(data["Department"][i]=='Sales' and data['Attrition'][i]=='No'):

        count2=count2+1;

    if(data["Department"][i]=='Research & Development' and data['Attrition'][i]=='Yes'):

        count3=count3+1;

    if(data["Department"][i]=='Research & Development' and data['Attrition'][i]=='No'):

        count4=count4+1;

    if(data["Department"][i]=='Human Resources' and data['Attrition'][i]=='Yes'):

        count5=count5+1;

    if(data["Department"][i]=='Human Resources' and data['Attrition'][i]=='No'):

        count6=count6+1;
print(count1)

print(count2)

print(count3)

print(count4)

print(count5)

print(count6)
import matplotlib.pyplot as plt

x=0

x_=('Sales','Research and development','Human Resources')

ax = plt.subplot(111)

ax.bar(x-0.2, count1, width=0.2, color='g', align='center',label='yes')

ax.bar(x, count2, width=0.2, color='r', align='center',label='no')

ax.bar(x+0.4, count3, width=0.2, color='g', align='center')

ax.bar(x+0.6, count4, width=0.2, color='r', align='center')

ax.bar(x+1.0, count5, width=0.2, color='g', align='center')

ax.bar(x+1.2, count6, width=0.2, color='r', align='center')

plt.xticks([-0.1,0.5,1.1],('Sales','R&D','Human Resources'))

plt.xlabel("Department")

plt.legend();

plt.show()
data["BusinessTravel"].unique()
count1=0

count2=0

count3=0

count4=0

count5=0

count6=0
for i in data.index:

    if(data["BusinessTravel"][i]=='Travel_Rarely' and data['Attrition'][i]=='Yes'):

        count1=count1+1;

    if(data["BusinessTravel"][i]=='Travel_Rarely' and data['Attrition'][i]=='No'):

        count2=count2+1;

    if(data["BusinessTravel"][i]=='Travel_Frequently' and data['Attrition'][i]=='Yes'):

        count3=count3+1;

    if(data["BusinessTravel"][i]=='Travel_Frequently' and data['Attrition'][i]=='No'):

        count4=count4+1;

    if(data["BusinessTravel"][i]=='Non-Travel' and data['Attrition'][i]=='Yes'):

        count5=count5+1;

    if(data["BusinessTravel"][i]=='Non-Travel' and data['Attrition'][i]=='No'):

        count6=count6+1;
print(count1)

print(count2)

print(count3)

print(count4)

print(count5)

print(count6)
x=0

x_=('Travel_Rarely','Travel_Frequently','Non-Travel')

ax = plt.subplot(111)

ax.bar(x-0.2, count1, width=0.2, color='y', align='center',label='yes')

ax.bar(x, count2, width=0.2, color='c', align='center',label='no')

ax.bar(x+0.4, count3, width=0.2, color='y', align='center')

ax.bar(x+0.6, count4, width=0.2, color='c', align='center')

ax.bar(x+1.0, count5, width=0.2, color='y', align='center')

ax.bar(x+1.2, count6, width=0.2, color='c', align='center')

plt.xticks([-0.1,0.5,1.1],('Travel_Rarely','Travel_Frequently','Non-Travel'))

plt.xlabel("BusinessTravel")

plt.legend();

plt.show()
!pip install wordcloud
import os



from os import path

from wordcloud import WordCloud



# get data directory (using getcwd() is needed to support running example in generated IPython notebook)

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()



# Read the whole text.

text = open(path.join(r"/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")).read()



# Generate a word cloud image

wordcloud = WordCloud().generate(text)



# Display the generated image:

# the matplotlib way:

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# lower max_font_size

wordcloud = WordCloud(max_font_size=40).generate(text)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()



# The pil way (if you don't have matplotlib)

# image = wordcloud.to_image()

# image.show()
data.isnull().any()
data.corr()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data['BusinessTravel']=le.fit_transform(data['BusinessTravel'])

data['Department']=le.fit_transform(data['Department'])

data['Gender']=le.fit_transform(data['Gender'])

data['MaritalStatus']=le.fit_transform(data['MaritalStatus'])

data['OverTime']=le.fit_transform(data['OverTime'])

data['Attrition']=le.fit_transform(data['Attrition'])

data['PerformanceRating']=le.fit_transform(data['PerformanceRating'])
data.head()
x=data.iloc[:,[0,2,4,5,10,11,16,17,18,22,23,24,25,28,31,33]].values

#0:age 1:Businesstravel 2:Department 3:Distance from home 4:EnvironmentSatisfaction 5:Gender 6:JobSatisfaction 

#7:MaritalStatus  8:MonthlyIncome 9:OverTime 10:PercentSalaryHike 11:PerformanceRating 12:RelationshipSatisfaction

#13:TotalWorkingYears 14:YearsAtCompany 15:YearsSinceLastPromotion 

y=data.iloc[:,1:2].values


from sklearn.preprocessing import OneHotEncoder

one=OneHotEncoder()

z=one.fit_transform(x[:,1:2]).toarray()#2 more columns

t=one.fit_transform(x[:,2:3]).toarray()#2 more columns

r=one.fit_transform(x[:,4:5]).toarray()#3 more columns

s=one.fit_transform(x[:,6:7]).toarray()#3 more columns

m=one.fit_transform(x[:,7:8]).toarray()#2 more columns

q=one.fit_transform(x[:,12:13]).toarray()#3 more columns

x=np.delete(x,[1,2,4,6,7,12],axis=1)

x=np.concatenate((q,m,s,r,t,z,x),axis=1)
x.shape#(16+2+2+3+3+2+3)
from sklearn.model_selection import train_test_split#then we divided the data into train set and test set so that we can send it for further processing

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=23)
x_train.shape
x_test.shape
#we scaled the data so that we can avoid outliers and we classify the data accurately

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(random_state=23,criterion='entropy')

dtc.fit(x_train,y_train)
dtcpred=dtc.predict(x_test)
dtcpred
y_test
from sklearn.metrics import accuracy_score

dtcacc=accuracy_score(y_test,dtcpred)
dtcacc
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,dtcpred)

cm
import sklearn.metrics as metrics

fpr,tpr,threshold=metrics.roc_curve(y_test,dtcpred)

roc_auc=metrics.auc(fpr,tpr)#false positive rate fpr and true positive rate tpr