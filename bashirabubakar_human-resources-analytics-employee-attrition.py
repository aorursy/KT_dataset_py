print('Hello My name is Bashir Abubakar and welcome to this exploration!')
# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install chart_studio
!pip install cufflinks
from chart_studio.plotly import plot, iplot
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly
import chart_studio
chart_studio.tools.set_credentials_file(username='bashman18', api_key='••••••••••')
init_notebook_mode(connected=True)
print('All modules imported')
df = pd.read_csv('../input/HR_COM1.CSV', index_col=None)
# Check to see if there are missing values in data set
df.isnull().any()
df.head()
df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
df.head()
df.shape
df.dtypes
# From observation about 76.1% of employees stayed and 23.8% of employees left. 
turnover_rate = df.turnover.value_counts() / 14999
turnover_rate
df.describe()
# Overview of summary (Turnover V.S. Non-turnover)
turnover_Summary = df.groupby('turnover')
turnover_Summary.mean()
corr = df.corr()
corr = (corr)
ax = plt.axes()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, ax=ax)
ax.set_title('Correlation Matrix Chart')
plt.show()
corr
# Compare the means of our employee turnover satisfaction against the employee population satisfaction
emp_population_satisfaction = df['satisfaction'].mean()
emp_turnover_satisfaction = df[df['turnover']==1]['satisfaction'].mean()

print( 'The mean for the employee population is: ' + str(emp_population_satisfaction) )
print( 'The mean for the employees that had a turnover is: ' + str(emp_turnover_satisfaction) )
import scipy.stats as stats
stats.ttest_1samp(a=  df[df['turnover']==1]['satisfaction'], # Sample of Employee satisfaction who had a Turnover
                  popmean = emp_population_satisfaction)  # Employee Population satisfaction mean
degree_freedom = len(df[df['turnover']==1])

LQ = stats.t.ppf(0.025,degree_freedom)  # Left Quartile

RQ = stats.t.ppf(0.975,degree_freedom)  # Right Quartile

print ('The t-distribution left quartile range is: ' + str(LQ))
print ('The t-distribution right quartile range is: ' + str(RQ))

import plotly.express as px

fig = px.histogram(df, x="satisfaction", y="Emp ID", color="turnover",
                   marginal="box", # or violin, rug
                   hover_data=df.columns)
fig.update_layout(title_text='Employee Distribution Plot - Satisfaction Level',xaxis_title="Satisfaction",
    yaxis_title="No of Employees")
fig.show()
import plotly.express as px

fig = px.histogram(df, x="evaluation", y="Emp ID", color="turnover",
                   marginal="box", # or violin, rug
                   hover_data=df.columns)
fig.update_layout(title_text='Employee Distribution Plot - Evaluation Level',xaxis_title="Evaluation",
    yaxis_title="No of Employees")
fig.show()
import plotly.express as px

fig = px.histogram(df, x="averageMonthlyHours", y="Emp ID", color="turnover",
                   marginal="box", # or violin, rug
                   hover_data=df.columns)
fig.update_layout(title_text='Employee Distribution Plot - Average Monthly Hours',xaxis_title="Average Monthly Hours",
    yaxis_title="No of Employees")
fig.show()
from plotly import graph_objs as go

fig = go.Figure()
for name, group in df.groupby('turnover'):
    trace = go.Histogram()
    trace.name = name
    trace.x = group['salary']
    fig.add_trace(trace)
fig.update_layout(title_text='Employee Evaluation Distribution - Salary V.S. Turnover',xaxis_title="Salary",
    yaxis_title="No of Employees")
fig.show()
import plotly.express as px
fig = go.Figure()
for name, group in df.groupby('turnover'):
    trace = go.Histogram()
    trace.name = name
    trace.x = group['dept']
    fig.add_trace(trace)
fig.update_layout(title_text='Employee Evaluation Distribution - Department V.S. Turnover',xaxis_title="Department",
    yaxis_title="No of Employees")
fig.show()
fig = go.Figure()
for name, group in df.groupby('turnover'):
    trace = go.Histogram()
    trace.name = name
    trace.x = group['projectCount']
    fig.add_trace(trace)
fig.update_layout(title_text='Employee Evaluation Distribution - Turnover V.S. Project Count',xaxis_title="Project Count",
    yaxis_title="No of Employees")    
fig.show()
import plotly.figure_factory as ff

x1 = df.loc[(df['turnover'] == 0),'evaluation']
x2 = df.loc[(df['turnover'] == 1),'evaluation']

group_labels = ['no turnover', 'turnover']

colors = ['blue', 'red']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot([x1, x2], group_labels, bin_size=1.0,
                         curve_type='kde', # override default 'kde'
                         colors=colors, show_hist=False)

# Add title
fig.update_layout(title_text='Employee Evaluation Distribution KDE - Turnover V.S. No Turnover')
fig.show()
import plotly.figure_factory as ff

x1 = df.loc[(df['turnover'] == 0),'averageMonthlyHours']
x2 = df.loc[(df['turnover'] == 1),'averageMonthlyHours']

group_labels = ['no turnover', 'turnover']

colors = ['blue', 'red']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot([x1, x2], group_labels, bin_size=1.0,
                         curve_type='kde', # override default 'kde'
                         colors=colors, show_hist=False)

# Add title
fig.update_layout(title_text='Employee AverageMonthly Hours Distribution KDE - Turnover V.S. No Turnover')
fig.show()
x1 = df.loc[(df['turnover'] == 0),'satisfaction']
x2 = df.loc[(df['turnover'] == 1),'satisfaction']

group_labels = ['no turnover', 'turnover']

colors = ['blue', 'red']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot([x1, x2], group_labels, bin_size=1.0,
                         curve_type='kde', # override default 'kde'
                         colors=colors, show_hist=False)

# Add title
fig.update_layout(title_text='Employee Satisfaction Distribution KDE - Turnover V.S. No Turnover')
fig.show()
fig = px.box(df,x="projectCount", y="averageMonthlyHours", color="turnover")
fig.update_layout(title_text='Project Count V.S. Average Monthly Hours',xaxis_title="Project Count",
    yaxis_title="Average Monthly Hours")
fig.show()
#ProjectCount VS Evaluation
#Looks like employees who did not leave the company had an average evaluation of around 70% even with different projectCounts
#There is a huge skew in employees who had a turnover though. It drastically changes after 3 projectCounts. 
#Employees that had two projects and a horrible evaluation left. Employees with more than 3 projects and super high evaluations left
fig = px.box(df,x="projectCount", y="evaluation", color="turnover")
fig.update_layout(title_text='Project Count V.S. Evaluation',xaxis_title="Project Count",
    yaxis_title="Evaluation")
fig.show()
import plotly.express as px
fig = px.scatter(df, x="satisfaction", y="evaluation",color="turnover")
fig.update_layout(title_text='Satisfaction V.S. Evaluation Cluster Chart', xaxis_title="Satisfaction",
    yaxis_title="Evaluation")
fig.show()
import plotly.express as px

fig = px.histogram(df, x="yearsAtCompany", y="Emp ID", color="turnover",
                   marginal="", # or violin, rug
                   hover_data=df.columns)
fig.update_layout(title_text='Turnover Rate V.S. Years Spent', xaxis_title="Years Spent",
    yaxis_title="No of Employees")
fig.show()
# Import KMeans Model
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(df[df.turnover==1][["satisfaction","evaluation"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]
# Determine number of clusters with K-means elbow method
# The arc of the elbow shows that number 3 is our best fit and as thus we would create  3 clusters of employee
sse={}
br = df[df.turnover==1][["satisfaction","evaluation"]]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df[df.turnover==1][["satisfaction","evaluation"]])

    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.show()
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter(
    x=df["satisfaction"], y=df["evaluation"],
    mode='markers',
    marker=dict(
        color= kmeans_colors,
        opacity=[1, 0.8, 0.6, 0.4],
        size=[40, 60, 80, 100],
    )
)])
fig.update_layout(title_text='Employee Cluster Chart')
fig.show()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler

# Create dummy variables for the 'department' and 'salary' features, since they are categorical 
department = pd.get_dummies(data=df['dept'],drop_first=True,prefix='dep') #drop first column to avoid dummy trap
salary = pd.get_dummies(data=df['salary'],drop_first=True,prefix='sal')
df.drop(['dept','salary'],axis=1,inplace=True)
df = pd.concat([df,department,salary],axis=1)
# Create base rate model
def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y
# Create train and test splits
target_name = 'turnover'
X = df.drop('turnover', axis=1)
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y=df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
# Check accuracy of base rate model
y_base_rate = base_rate_model(X_test)
from sklearn.metrics import accuracy_score
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))
# Check accuracy of Logistic Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1)

model.fit(X_train, y_train)
print ("Logistic accuracy is %2.2f" % accuracy_score(y_test, model.predict(X_test)))
# Using 10 fold Cross-Validation to train our Logistic Regression Model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression(class_weight = "balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# Compare the Logistic Regression Model V.S. Base Rate Model V.S. Random Forest Model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


print ("---Base Model---")
base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print ("Base Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, base_rate_model(X_test)))

# NOTE: By adding in "class_weight = balanced", the Logistic Auc increased by about 10%! This adjusts the threshold value
logis = LogisticRegression(class_weight = "balanced")
logis.fit(X_train, y_train)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, logis.predict(X_test)))

# Decision Tree Model
dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
rf.fit(X_train, y_train)
print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))


# Ada Boost
ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
ada.fit(X_train,y_train)
print ("\n\n ---AdaBoost Model---")
ada_roc_auc = roc_auc_score(y_test, ada.predict(X_test))
print ("AdaBoost AUC = %2.2f" % ada_roc_auc)
print(classification_report(y_test, ada.predict(X_test)))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])

plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)

# Plot AdaBoost ROC
plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

importances = dtree.feature_importances_
feat_names = df.drop(['turnover'],axis=1).columns

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()