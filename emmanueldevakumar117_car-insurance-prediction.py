


import pandas as pd 

import numpy as np


import matplotlib.pyplot as plt
%matplotlib inline


import seaborn as sns



# 'Scikit-learn' features various classification, regression and clustering algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# 'Statsmodels' is used to explore data, estimate statistical models, and perform statistical tests
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv("../input/carinsurance/carInsurance_train.csv")

df.head(1)
df.shape
df.dtypes
df[['CallStart','CallEnd']] = df[['CallStart','CallEnd']].astype('datetime64[ns]')
df.dtypes
df.isnull().sum().sort_values()
df.describe(include='object')
df['Communication'].value_counts()
pd.crosstab(index=df['Communication'],columns=df['Age'])
AR ={}
for i in df['Age']:
    if i <= 18 or i <= 20:
        AR.setdefault(i,'18-20')
    elif i <20  or i <= 30:
        AR.setdefault(i,'21-30')
    elif i < 30 or i <= 40:
        AR.setdefault(i,'31-40')
    elif i < 40 or i <= 50:
        AR.setdefault(i,'41-50')
    elif i < 50 or i <= 60:
        AR.setdefault(i,'51-60')
    elif i < 60 or i <= 70:
        AR.setdefault(i,'61-70')
    elif i < 70 or i <= 80:
        AR.setdefault(i,'71-80')
    elif i < 80 or i <= 90:
        AR.setdefault(i,'81-90')
    elif i < 90 or i <= 100:
        AR.setdefault(i,'91-100')
    elif i >100:
        AR.setdefault(i,'above 100')
df['Age_Range'] = df['Age'].map(AR)
df.head()
pd.crosstab(index=df['Communication'],columns=df['Age_Range'])
df['Education'].value_counts()
pd.crosstab(index=df['Job'],columns=df['Age_Range'])
df[df['Job'].isnull()]
df=df.drop('Outcome',axis=1)
df.head(1)
for i in df[['Job','Education','Communication']]:
    df[i] = df[i].fillna(df[i].mode()[0])
df.isnull().sum()
df.describe()
cor=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor,annot=True,cmap='YlGnBu')
df=df.drop(columns='Id',axis=1)
df.head()
s = {}
for i in df['LastContactDay']:
    if i in range (0,11):
        s.setdefault(i,'Start of month')
    elif i in range (11,21):
        s.setdefault(i,'Middle of month')
    elif i in range (21,32):
        s.setdefault(i,'End of month')
df['Last_Contact_phase'] = df['LastContactDay'].map(s)
df.head(2)
df['Call_duration'] = df['CallEnd']-df['CallStart']
df.head(5)
a = pd.DataFrame()
a=pd.crosstab(index=df['Age_Range'],columns=df['CarInsurance'])
a['Percent_of_enrolled'] = round((a[1]/(a[0]+a[1]))*100,2)
a=a.sort_values(by='Percent_of_enrolled')
a
plt.bar(x=a.index,height=a['Percent_of_enrolled'])
df
b = pd.DataFrame()
b=pd.crosstab(index=df['Last_Contact_phase'],columns=df['CarInsurance'])
b['Percent_of_enrolled vs Contact_phase'] = round((b[1]/(b[0]+b[1]))*100,2)
b=b.sort_values(by='Percent_of_enrolled vs Contact_phase')
b
plt.bar(x=b.index,height=b['Percent_of_enrolled vs Contact_phase'])
df.head(2)
c = pd.DataFrame()
c=pd.crosstab(index=df['Communication'],columns=df['CarInsurance'])
c['Percent_of_enrolled vs Communication mode'] = round((c[1]/(c[0]+c[1]))*100,2)
c=c.sort_values(by='Percent_of_enrolled vs Communication mode')
c
plt.bar(x=c.index,height=c['Percent_of_enrolled vs Communication mode'])
d = pd.DataFrame()
d=pd.crosstab(index=df['Marital'],columns=df['CarInsurance'])
d['Percent_of_enrolled vs Marital status'] = round((d[1]/(d[0]+d[1]))*100,2)
d=d.sort_values(by='Percent_of_enrolled vs Marital status')
d
plt.bar(x=d.index,height=d['Percent_of_enrolled vs Marital status'])
e = pd.DataFrame()
e=pd.crosstab(index=df['Job'],columns=df['CarInsurance'])
e['Percent_of_enrolled vs Job'] = round((e[1]/(e[0]+e[1]))*100,2)
e=e.sort_values(by='Percent_of_enrolled vs Job')
e
plt.figure(figsize=(15,10))
plt.bar(x=e.index,height=e['Percent_of_enrolled vs Job'])
f = pd.DataFrame()
f=pd.crosstab(index=df['Education'],columns=df['CarInsurance'])
f['Percent_of_enrolled vs Education'] = round((f[1]/(f[0]+f[1]))*100,2)
f=f.sort_values(by='Percent_of_enrolled vs Education')
f
plt.bar(x=f.index,height=f['Percent_of_enrolled vs Education'])
g = pd.DataFrame()
g=pd.crosstab(index=df['HHInsurance'],columns=df['CarInsurance'])
g['Percent_of_enrolled vs HHInsurance'] = round((g[1]/(g[0]+g[1]))*100,2)
g=g.sort_values(by='Percent_of_enrolled vs HHInsurance')
g
plt.bar(x=g.index,height=g['Percent_of_enrolled vs HHInsurance'])
plt.xticks(ticks=[0,1])
h = pd.DataFrame()
h=pd.crosstab(index=df['CarLoan'],columns=df['CarInsurance'])
h['Percent_of_enrolled vs CarLoan'] = round((h[1]/(h[0]+h[1]))*100,2)
h=h.sort_values(by='Percent_of_enrolled vs CarLoan')
h
plt.bar(x=h.index,height=h['Percent_of_enrolled vs CarLoan'])
plt.xticks(ticks=[0,1])
df.describe()
r = df['Call_duration'].median()
e = {}
for i in df['Call_duration']:
    if i < r:
        e.setdefault(i,'Duration less than median duration')
    else:
        e.setdefault(i,'Duration greater than median duration')
df['Call_duration'] = df['Call_duration'].map(e)
df
k = pd.DataFrame()
k=pd.crosstab(index=df['Call_duration'],columns=df['CarInsurance'])
k['Percent_of_enrolled vs Call_duration'] = round((k[1]/(k[0]+k[1]))*100,2)
k=k.sort_values(by='Percent_of_enrolled vs Call_duration')
k
plt.figure(figsize=(7,7))
plt.bar(x=k.index,height=k['Percent_of_enrolled vs Call_duration'])
df_features =list(df.select_dtypes(exclude=['object','datetime64']))
plt.figure(figsize=(15,25))

for i,j in enumerate (df_features):
    plt.subplot(10,2,i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.distplot(df[j])
e = list(df.select_dtypes(exclude=['object','datetime64']))
plt.figure(figsize=(15,30))
for i,j in enumerate (e):
    plt.subplot(8,2,i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.boxplot(df[j])
df.head(1)
sns.pairplot(df)
df=df.drop(columns=['CallStart','CallEnd'])
df.select_dtypes(include='object').columns
df.head()
l = pd.DataFrame()
l=pd.crosstab(index=df['LastContactMonth'],columns=df['CarInsurance'])
l['Percent_of_enrolled vs LastContactMonth'] = round((l[1]/(l[0]+l[1]))*100,2)
l=l.sort_values(by='Percent_of_enrolled vs LastContactMonth')
l
plt.figure(figsize=(10,10))
plt.bar(x=l.index,height=l['Percent_of_enrolled vs LastContactMonth'])
df.columns
s = {}
a=list(l.index)
for i in a:
    s.setdefault(i,a.index(i))
df['LastContactMonth'] = df['LastContactMonth'].map(s) 
df
e = pd.DataFrame()
e=pd.crosstab(index=df['Job'],columns=df['CarInsurance'])
e['Percent_of_enrolled vs Job'] = round((e[1]/(e[0]+e[1]))*100,2)
e=e.sort_values(by='Percent_of_enrolled vs Job')
e
x = list(e.index)
y = {}
for i in x:
    y.setdefault(i,x.index(i))
y
df['Job'] = df['Job'].map(y)
df
a = pd.DataFrame()
a=pd.crosstab(index=df['Age_Range'],columns=df['CarInsurance'])
a['Percent_of_enrolled'] = round((a[1]/(a[0]+a[1]))*100,2)
a=a.sort_values(by='Percent_of_enrolled')
x = list(a.index)
y = {}
for i in x:
    y.setdefault(i,x.index(i))
y
df['Age_Range'] = df['Age_Range'].map(y)
df.head(5)
df_dummies = pd.get_dummies(data=df,columns=['Marital','Education','Communication','Last_Contact_phase','Call_duration'],drop_first=True)
df_dummies
X = df_dummies.drop('CarInsurance',axis=1)
y = df_dummies['CarInsurance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state =100)
log_reg = LogisticRegression()
model = log_reg.fit(X_train,y_train)
model.score(X_train,y_train)
y_pred=model.predict(X_test)
metrics.accuracy_score(y_test,y_pred)
result = classification_report(y_test,y_pred)
print (result)
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
clf.score(X_train,y_train)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
test_scores = []
train_scores = []

# build the model only on odd values of K, as in KNN we choose the odd K 
for i in np.arange(1,15):
    
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(i)
    
    #Fit the model
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_scores.append(knn.score(X_train,y_train))
    
    #Compute accuracy on the test set
    test_scores.append(knn.score(X_test,y_test))
print (train_scores)
print (test_scores)
param_grid_svm = {'kernel': ['linear', 'rbf', 'poly'], 'C' : [0.001, 0.01, 0.1, 1, 100]}
svm = SVC()
svmod = svm.fit(X_train,y_train)
svmod.score(X_train,y_train)
y_pred = svmod.predict(X_test)
metrics.accuracy_score(y_test,y_pred)
df
df_dummies
feature_names = list(X.columns)
feature_names
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp.sort_values()
feature_imp[feature_imp.sort_values()>0.03].index
df_new=df_dummies[['Call_duration_Duration less than median duration', 'Balance',
       'LastContactMonth', 'Age', 'LastContactDay', 'Job', 'NoOfContacts',
       'DaysPassed', 'Age_Range', 'HHInsurance', 'PrevAttempts']]
X1 = df_new
y1 = df_dummies['CarInsurance']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.30, random_state =100)
log_reg = LogisticRegression(max_iter=10000)
model = log_reg.fit(X1_train,y1_train)
model.score(X1_train,y1_train)
y1_pred=model.predict(X1_test)
metrics.accuracy_score(y1_test,y1_pred)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y1_test,y1_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
c1_pred=model.predict_proba(X1_test)
for i in c1_pred:
    if i[0]>0.65:
        i[0]=0
    else:
        i[0]=1
q = []
for i in c1_pred:
    q.append(i[0])
y_pre = pd.Series(q)
metrics.accuracy_score(y1_test,y_pre)
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.30, random_state =100)
gb.fit(X1_train,y1_train)
model.score(X1_train,y1_train)
c1_pred=gb.predict_proba(X1_test)
for i in c1_pred:
    if i[0]>0.65:
        i[0]=0
    else:
        i[0]=1
q = []
for i in c1_pred:
    q.append(i[0])
y_pre = pd.Series(q)
print (metrics.accuracy_score(y1_test,y_pre))
gb.score(X1_train,y1_train)
