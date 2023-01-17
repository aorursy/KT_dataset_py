# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
%matplotlib inline
import datetime as dt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
train.head()
row = train.shape[0]
col = train.shape[1]
print("The number of rows within the dataset are {} and the number of columns are {}".format(row,col))
train.info()
train.isnull().sum()
dates = []
for i in train['published_timestamp']:
    datess=dt.datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ')
    dates.append(datess)

train['time'] = dates
train['year'] = train['time'].dt.year
train.head()
train.drop('published_timestamp',axis=1,inplace=True)

Free = train[train['is_paid'] == 0]
Free.shape

Paid = train[train['is_paid'] == 1]
Paid.shape
columns = train['subject'].unique()
columns
for x in columns:
    maxr = Free[Free['subject'] == x]['num_subscribers'].max()
    course = Free[(Free['num_subscribers'] == maxr)]['course_title'].unique()
    print("-----------------------------------------------------------------------------------")
    print("The best free course offered by udemy for {} is \n{} with {} subscribers\n".format(x,course[0],maxr))

for x in columns:
    maxr = Paid[Paid['subject'] == x]['num_subscribers'].max()
    course = Paid[(Paid['num_subscribers'] == maxr)]['course_title'].unique()
    print("-----------------------------------------------------------------------------------")
    print("The best paid course offered by udemy for {} is \n{} with {} subscribers\n".format(x,course[0],maxr))


print("The top 5 most popular courses with respect to subcribers are:\n")
popular = train.sort_values(['num_subscribers'],ascending=False).head()['course_title'].unique()
i = 0
while i<len(popular):
    print(popular[i])
    i = i+1
train['engagment']  = train['num_subscribers'] * train['num_reviews']
print("The top 5 most engaging courses with respect to subcribers and reviews are are:\n")
engaging = train.sort_values(['engagment'],ascending=False).head()['course_title'].unique()
i = 0
while i<len(engaging):
    print(engaging[i])
    i = i+1
priceprefect = train[(train['price']<=train['price'].mean()) & (train['engagment']>=train['engagment'].mean())].sort_values(('engagment'),ascending=False)['course_title'].head(1).unique()[0]
print("The best course that offers cost benefit is",priceprefect)
years = train['year'].unique()

for x in years:
    maxr = train[train['year'] == x]['num_subscribers'].max()
    course = train[(train['num_subscribers'] == maxr)]['course_title'].unique()
    print("-----------------------------------------------------------------------------------")
    print("The best course offered by udemy in {} was \n{} with {} subscribers\n".format(x,course[0],maxr))
    
sns.countplot('is_paid',data=train)
sns.countplot('is_paid',hue='subject',data=train)
sns.countplot('is_paid',hue='level',data=train)
plt.legend(loc='upper left')
plt.figure(figsize=(8,5))
sns.countplot('subject',hue='level',data=train)
train.price.hist(bins=10)
plt.xlabel("Price")
plt.title("Price range and its frequency")
plt.figure(figsize=(10,5))
sns.barplot('subject','price',hue='level',data=Paid)
sns.lmplot('num_subscribers','num_reviews',data=train)
table1 = pd.pivot_table(train, values=['num_subscribers','num_reviews'], index=['is_paid'],aggfunc=np.sum)
table1
table1.plot(kind='bar')
plt.figure(figsize=(10,5))
sns.barplot('subject','num_subscribers',hue='level',data=train)
plt.figure(figsize=(10,5))
sns.barplot('subject','num_reviews',hue='level',data=train)
sns.lmplot('num_lectures','content_duration',data=train)
table2 = pd.pivot_table(train, values=['num_lectures','content_duration'], index=['is_paid'],aggfunc=np.sum)
table2
table2.plot(kind='bar')
plt.figure(figsize=(10,5))
sns.barplot('subject','content_duration',hue='level',data=train)
sns.countplot('year',data=train)
table3 = pd.pivot_table(train, values=['num_subscribers'], index=['year'],columns=['subject'],aggfunc=np.sum)
table3
table3.plot(kind='bar',figsize=(10,5))
table4 = pd.pivot_table(train, values=['num_lectures'], index=['year'],columns=['subject'],aggfunc=np.sum)
table4
table4.plot(kind='bar',figsize=(8,5))
plt.figure(figsize=(10,5))
sns.barplot('year','price',data=train,estimator=np.sum)
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
model = train
model.describe()
id = model[model['num_lectures'] == 0].index.values[0]
model.drop(id,axis=0,inplace=True)
model.info()
model.drop(['course_title','url','time'],axis=1,inplace=True)
def fun(val): 
  
    if val == 0: 
        return 0
    else: 
        return 1
    
model['is_paid'] = model['is_paid'].apply(fun)
def lev(val): 
  
    if val == 'All Levels': 
        return 0
    elif val == 'Intermediate Level': 
        return 1
    elif val == 'Beginner Level': 
        return 2
    else:
        return 3
    
model['level'] = model['level'].apply(lev)
def sub(val): 
  
    if val == 'Business Finance': 
        return 0
    elif val == 'Graphic Design': 
        return 1
    elif val == 'Musical Instruments': 
        return 2
    else:
        return 3
    
model['subject'] = model['subject'].apply(sub)
def year(val): 
  
    if val == 2017: 
        return 6
    elif val == 2016: 
        return 5
    elif val == 2015: 
        return 4
    elif val == 2014: 
        return 3
    elif val == 2013: 
        return 2
    elif val == 2012: 
        return 1
    else:
        return 0
    
model['year'] = model['year'].apply(year)
X = model.drop(['course_id','is_paid'],axis=1)
y = model['is_paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Import Packages 
from sklearn.linear_model import LogisticRegression
#Object creation and fitting of training set
lmodel = LogisticRegression()
lmodel.fit(X_train,y_train)
#Creation of a prediction variable
lpredictions = lmodel.predict(X_test)
#Accuracy Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,lpredictions))

lscore = round((lmodel.score(X_test, y_test)*100),2)
print ("\nModel Score:",lscore,"%")
#Import Packages 
from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier(n_neighbors=3)
kmodel.fit(X_train,y_train)
#Creation of a prediction variable
kpredictions = kmodel.predict(X_test)
#Accuracy Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,kpredictions))

kscore = round((kmodel.score(X_test, y_test)*100),2)
print ("\nModel Score:",kscore,"%")
#Import Packages 
from sklearn.tree import DecisionTreeClassifier
#Object creation and fitting of training set
dmodel = DecisionTreeClassifier()
dmodel.fit(X_train,y_train)
#Creation of a prediction variable
dprediction = dmodel.predict(X_test)
#Accuracy Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,dprediction))

dscore = round((dmodel.score(X_test, y_test)*100),2)
print ("\nModel Score:",dscore,"%")
#Import Packages 
from sklearn.ensemble import RandomForestClassifier
#Object creation and fitting of training set
rmodel = RandomForestClassifier(n_estimators=100)
rmodel.fit(X_train,y_train)
#Creation of a prediction variable
rprediction = rmodel.predict(X_test)

#Accuracy Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,rprediction))

rscore = round((rmodel.score(X_test, y_test)*100),2)
print ("\nModel Score:",rscore,"%")
#Import Packages 
from sklearn.svm import SVC
#Object creation and fitting of training set
smodel = SVC()
smodel.fit(X_train,y_train)
#Creation of a prediction variable
sprediction = smodel.predict(X_test)
#Accuracy Matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,sprediction))

sscore = round((smodel.score(X_test, y_test)*100),2)
print ("\nModel Score:",sscore,"%")
data = [['Logistic Regression',lscore],['K-Nearest Neighbour',kscore],
        ['Decision Tree',dscore],['Random Forest',rscore],['Support Vector Machine',sscore]]
final = pd.DataFrame(data,columns=['Algorithm','Precision'],index=[1,2,3,4,5])
print("The results of Data Modeling are as follows:\n ")
print(final)