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
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot, iplot
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
init_notebook_mode(connected = True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
cf.go_offline()


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


%matplotlib inline
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()

print('Shape of the dataframe is {}'.format(df.shape))
df.describe()
df.isnull().sum()
df.hist(figsize = (20,20))
plt.show()
plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True)
print('Average Age of Employees who left the company : {:.2f}'.format(df[df['Attrition'] == 'Yes']['Age'].mean()))
print('Average Age of Employees who left the company : {:.2f}'.format(df[df['Attrition'] == 'No']['Age'].mean()))
plt.figure(figsize = (10,8))
sns.countplot(x = 'BusinessTravel', hue = 'Attrition', data = df)
le = LabelEncoder()
le.fit(df['Attrition'])
df['Attrition'] = le.transform(df['Attrition'])
a = []
b = [df['EducationField'].unique()]
for i in df['EducationField'].unique():
    a.append((df[(df['EducationField']==i) & (df['Attrition'] == 1)].shape[0]/df[df['EducationField']==i].shape[0])*100)
pd.DataFrame(a,b, columns = ['Attrition percentage by Education Field'])
xticklabels = ['Life Sciences','Other','Medical','Marketing','Technical Degree','Human Resources']
pd.DataFrame(a,b, columns = ['Attrition percentage by Education Field']).iplot(kind = 'bar')
abc = (df1.groupby('WorkLifeBalance').agg('count')['Attrition']/df.groupby('WorkLifeBalance').agg('count')['Attrition']) * 100
fig = px.bar(x = abc.index, y = abc.values, labels={'x': 'Work Life Balance', 'y' : 'Count'})
fig.show()
xticks = ['Female','Male']
abc = ((df[df['Attrition'] == 1].groupby('Gender').agg('count'))/(df.groupby('Gender').agg('count')) *100)['Attrition']
sns.barplot(xticks,abc)

df1 = df[df['Attrition'] == 1]
fig = px.histogram(df1, x= 'MonthlyIncome')
fig.show()
fig1 = px.histogram(df,x = 'HourlyRate')
fig1.show()
plt.figure(figsize = (10,8))
sns.countplot(x = 'PerformanceRating', hue = 'Attrition', data = df)
df_Attr_Yes = df[df['Attrition'] == 1]
abc = df_Attr_Yes['JobSatisfaction'].value_counts()/df['JobSatisfaction'].value_counts() * 100
fig = px.bar(x = abc.index, y = abc.values)
fig.show()
fig2 = px.histogram(df[df['Attrition'] == 1], x = 'PercentSalaryHike', color = 'Attrition')
fig2.show()
fig3 = px.pie(df[df['Attrition'] == 1],values = 'Attrition', names = 'Department',title = 'Attrition by Department')
fig3.show()
for col in df.columns[2:]:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <=2:
            print(col)
df['Gender'] = le.fit_transform(df['Gender'])
df['OverTime'] = le.fit_transform(df['OverTime'])
df.drop(columns = 'Over18',inplace = True)
df = pd.get_dummies(df,drop_first = True)
scaler = StandardScaler()
X = scaler.fit_transform(df)
df.head()
y = df.loc[:,'Attrition']
X = df.drop('Attrition', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)
print('Shape of X_train {}'.format(X_train.shape))
print('Shape of X_test {}'.format(X_test.shape))
print('Shape of y_train {}'.format(y_train.shape))
print('Shape of y_test {}'.format(y_test.shape))
error_rate = []
for i in range(1,40):
    clf = KNeighborsClassifier(n_neighbors = i)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    error_rate.append(np.mean(y_test != y_pred))
plt.figure(figsize = (10,8))
plt.plot(range(1,40),error_rate)
plt.xlabel('K value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K value')
plt.show()
models = [LogisticRegression(max_iter=1200000, dual = False), RandomForestClassifier(), SVC(gamma = 'auto'), DecisionTreeClassifier(),GaussianNB(),KNeighborsClassifier(n_neighbors = 22)]
model_names = ['Logistic Regression', 'Random Forest Classifier', 'SVC','Decision Tree Classifier', 'Naive Bayes','KNeigbors Classifier']
f1_score_ = []
roc_auc = []
overall = []
for i,j in zip(model_names,models):
    clf = j
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_score_ =  f1_score(y_test,y_pred) * 100
    roc_auc = roc_auc_score(y_test,y_pred) * 100
    overall.append([i,f1_score_,roc_auc])
metrics = pd.DataFrame(overall, columns = ['Model Name','F1-Score','ROC AUC Score'])
metrics
fig = plt.figure(figsize = (10,8))
fig = px.bar(x = metrics['Model Name'], y = metrics['ROC AUC Score'])
fig.show()
grid={'C': np.arange(1e-03, 2, 0.01)}
log_GS = GridSearchCV(LogisticRegression(solver='liblinear',
                                         class_weight="balanced", 
                                         random_state=7), param_grid = grid, verbose=True,return_train_score=True,scoring = 'roc_auc', iid = True, cv = 10 )
log_GS.fit(X_train,y_train)
print('Best Estimator : {}'.format(log_GS.best_estimator_))
print('Best Score : {}'.format(log_GS.best_score_))
clf = LogisticRegression(C=1.9109999999999996, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=7, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print('F1-Score',round(f1_score(y_test,y_pred)*100))
print('AUC ROC Score',round(roc_auc_score(y_test,y_pred)*100))
plt.figure(figsize = (8,5))
fig = ff.create_annotated_heatmap(confusion_matrix(y_test,y_pred))
fig.show()
