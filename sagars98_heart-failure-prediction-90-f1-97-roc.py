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
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
data.shape
data.info()
data.describe()
data.isnull().sum()
sns.set_style('darkgrid')
sns.set_palette('coolwarm')
sns.distplot(data['age'])
for i in data.columns:
    plt.figure()
    plt.hist(data[i],density=True)
data_corr = data.corr()
data_corr
plt.figure(figsize=(10,6))
sns.heatmap(data_corr,annot = True, cmap = 'coolwarm' ,vmin=-1)
data_corr[np.absolute(data_corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
data.age.value_counts(ascending=False)
x = data[(data['age'] > 45)
          & (data['DEATH_EVENT'] == 1)]
len(x)
data[data['DEATH_EVENT'] == 1]
male = data[data['sex'] == 1]
male_death = male[male['DEATH_EVENT']==1]
male_alive = male[male['DEATH_EVENT']==0]
male_death.head()
male_alive.head()
female = data[data['sex'] == 0]
female_death = female[female['DEATH_EVENT']==1]
female_alive = female[female['DEATH_EVENT']==0]
female_death.head()

female_alive.head()
labels = ['Male' ,'Female']

values = [len(data[data['sex']==1]) , len(data[data['sex']==0])]

fig = go.Figure(data=[go.Pie(labels=labels,values=values , hole =.4)])

fig.update_layout(
    title_text = "Gender Distribution")

fig.show()
labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]
values = [len(male[data["DEATH_EVENT"]==0]),len(male[data["DEATH_EVENT"]==1]),
         len(female[data["DEATH_EVENT"]==0]),len(female[data["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Survival Analysis- Gender")
fig.show()
labels = ['No','Yes']
diabetes_yes = data[data['diabetes']==1]
diabetes_no = data[data['diabetes']==0]
values = [len(diabetes_no), len(diabetes_yes)]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Diabetes")
fig.show()
diabetes_yes_survi = diabetes_yes[data["DEATH_EVENT"]==0]
diabetes_yes_not_survi = diabetes_yes[data["DEATH_EVENT"]==1]
diabetes_no_survi = diabetes_no[data["DEATH_EVENT"]==0]
diabetes__no_not_survi = diabetes_no[data["DEATH_EVENT"]==1]
labels = ['Diabetes Yes - Survived','Diabetes Yes - Not Survived', 'Diabetes NO - Survived', 'Diabetes NO - Not Survived']
values = [len(diabetes_yes[data["DEATH_EVENT"]==0]),len(diabetes_yes[data["DEATH_EVENT"]==1]),
         len(diabetes_no[data["DEATH_EVENT"]==0]),len(diabetes_no[data["DEATH_EVENT"]==1])]
colors = ['gold', 'mediumturquoise', 'fuchsia', 'lightgreen']
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Survival - Diabetes")
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()
anaemia_yes = data[data['anaemia']==1]
anaemia_no = data[data['anaemia']==0]
labels = ['Yes' , 'No']
values = [len(anaemia_yes) , len(anaemia_no)]

fig = go.Figure(data =[go.Pie(labels=labels, values=values ,hole=.4)])
fig.update_layout(
    title_text = 'Anaemia Analysis',
    )
fig.show()
anaemia_yes_survived = anaemia_yes[anaemia_yes['DEATH_EVENT']==0]
anaemia_yes_not_survived = anaemia_yes[anaemia_yes['DEATH_EVENT']==1]

anaemia_no_survived = anaemia_no[anaemia_no['DEATH_EVENT']==0]
anaemia_no_not_survived = anaemia_no[anaemia_no['DEATH_EVENT']==1]
labels = ['Anaemia Yes - Survived','Anaemia Yes - Not Survived', 'Anaemia No - Survived', 'Anaemia NO - Not Survived']
values = [len(anaemia_yes_survived), len(anaemia_yes_not_survived), len(anaemia_no_survived), len(anaemia_no_not_survived)]

colors = ['cyan','midnightblue','magenta','yellow']
fig = go.Figure(data =[go.Pie(labels=labels, values=values ,hole=.4)])
fig.update_layout(
    title_text = 'Survival Analysis - Anaemia')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()
hbp_yes = data[data['high_blood_pressure']==1]
hbp_not = data[data['high_blood_pressure']==0]
labels = ['High Blood Pressure' ,'Not High Blood Pressure']
values = [len(hbp_yes) , len(hbp_not)]
fig = go.Figure(data =[go.Pie(labels=labels, values=values ,hole=.4)])
fig.update_layout(
    title_text = 'High Blood Pressure Analysis')
fig.show()
high_bp_survived = hbp_yes[hbp_yes['DEATH_EVENT']==0]
high_bp_survived_not = hbp_yes[hbp_yes['DEATH_EVENT']==1]
not_high_bp_survived = hbp_not[hbp_not['DEATH_EVENT']==0]
not_high_bp_not_survived = hbp_not[hbp_not['DEATH_EVENT']==1]
labels = ['High BP - Survived' , 'High BP - Not Survived' , 'Not High BP - Survived' , 'Not High BP - Not Survived']
values = [len(high_bp_survived) , len(high_bp_survived_not) ,len(not_high_bp_survived) ,len(not_high_bp_not_survived)]

colors=['aqua','mistyrose']
fig = go.Figure(data =[go.Pie(labels=labels, values=values ,hole=.4)])
fig.update_layout(
    title_text = 'Survival Analysis - High Blood Pressure')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()
smoking_yes = data[data['smoking']==1] 
smoking_no  = data[data['smoking']==0]
label = ['Yes' ,'No']
values = [len(smoking_yes),len(smoking_no)]

fig = go.Figure(data =[go.Pie(labels=labels, values=values ,hole=.4)])
fig.update_layout(
    title_text = 'Smoking Analysis')
fig.show()
smoking_yes_surv = smoking_yes[smoking_yes['DEATH_EVENT']==0]
smoking_yes_not_surv = smoking_yes[smoking_yes['DEATH_EVENT']==1]
smoking_no_surv = smoking_no[smoking_no['DEATH_EVENT']==0]
smoking_no_not_surv = smoking_no[smoking_no['DEATH_EVENT']==1]
labels = ['Smoking Yes - Survived','Smoking Yes - Not Survived', 'Smoking No - Survived', 'Smoking NO- Not Survived']
values = [len(smoking_yes_surv), len(smoking_yes_not_surv), len(smoking_no_surv), len(smoking_no_not_surv)]

fig = go.Figure(data =[go.Pie(labels=labels, values=values ,hole=.4)])
fig.update_layout(
    title_text = 'Survival Analysis - Smoking')
fig.show()
X = data.drop('DEATH_EVENT',axis=1)
y = data['DEATH_EVENT']
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2 , random_state=101)
X_train.shape , X_test.shape , y_train.shape , y_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),   #Step1 - normalize data
    ('clf', LogisticRegression())       #Step2 - classifier
])
pipeline.steps
from sklearn.model_selection import cross_validate

scores = cross_validate(pipeline, X_train, y_train)
scores

scores['test_score'].mean()
from sklearn.neighbors import KNeighborsClassifier
k = range(1,30)
error_rate = []

for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))
plt.figure(figsize=(10,6))
plt.plot(k,error_rate,color='b',ls='--',marker='o',markerfacecolor='red',markersize=10)
plt.xticks(k)
plt.xlabel('K-Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K')
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=13))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())

for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for k, v in scores.items():
            print(k,' mean ', v.mean())
          

from sklearn.metrics import accuracy_score , f1_score , roc_auc_score
from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate = 0.005)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(X_train,y_train)
y_pred = lgb.predict(X_test)
print(accuracy_score(y_test,y_pred))
print('F1-score: ',f1_score(y_test,y_pred))
print('Roc_Auc_Score: ',roc_auc_score(y_test,y_pred))
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
X_train.head()
features_imp = pd.DataFrame({'feature': X_train.columns, 'importance': rfc.feature_importances_}).sort_values(by='importance', ascending=False)
features_imp = features_imp.reset_index()
features_imp
X = data[['time','serum_creatinine','ejection_fraction','age']]
y = data['DEATH_EVENT']
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2 , random_state=2698)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [80, 90, 100, 110],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Instantiate the grid search model
grid = GridSearchCV(estimator = rfc, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid.fit(X,y)
rfc_tuned = RandomForestClassifier(max_depth=100,min_samples_leaf=3,
                       min_samples_split=12 , random_state=108)
rfc_tuned.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print("""
------------------------
Random Forest Classifier
------------------------""")

print('Accuracy score: ',accuracy_score(y_test,y_pred))
print('F1 - Score: ',f1_score(y_test,y_pred))
print('ROC_AUC Score: ',roc_auc_score(y_test,y_pred))
gbt = GradientBoostingClassifier()
gbt.fit(X_train,y_train)
y_pred = gbt.predict(X_test)
print("""
---------------------------
Gradient Boosted Classifier
---------------------------""")
print('Accuracy score: ',accuracy_score(y_test,y_pred))
print('F1 - Score: ',f1_score(y_test,y_pred))
print('ROC_AUC Score: ',roc_auc_score(y_test,y_pred))
xgb = XGBClassifier()
xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)
print("""
------------------
XgBoost Classifier
------------------""")
print('Accuracy score: ',accuracy_score(y_test,y_pred))
print('F1 - Score: ',f1_score(y_test,y_pred))
print('ROC_AUC Score: ',roc_auc_score(y_test,y_pred))