import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import eli5 
from eli5.sklearn import PermutationImportance
import shap 
from pdpbox import pdp, info_plots 
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.svm import SVC
np.random.seed(123)
from sklearn.tree import export_graphviz

pd.options.mode.chained_assignment = None 
%matplotlib inline
df = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df.head(5)
df.describe()
df.isnull().sum()
import seaborn as sns
plt.figure(figsize=(10,8), dpi= 80)
sns.heatmap(df.corr(), cmap='RdYlGn', center=0)

# Decorations
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
sns.distplot(df['target'],rug=True)
plt.show()
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
col = "target"
colors = ['gold', 'blue']
grouped = df[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
#trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0],
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))
layout = {'title': 'Target(0 = No, 1 = Yes)'}
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)

sns.distplot(df['sex'],rug=True)
plt.show()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
col = "sex"
grouped = df[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])
layout = {'title': 'Male(1), Female(0)'}
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
sns.countplot(x=df.target,hue=df.sex)
plt.legend(labels=['Female', 'Male'])
dy=pd.DataFrame(df.groupby('sex')['target'].mean().reset_index().values,
                    columns=["gender","target1"])
dy.head()
sns.barplot(dy.gender,dy.target1)
plt.ylabel('rate of heart attack')
plt.title('0=Female, 1=Male')
sns.distplot(df['cp'],rug=True)
plt.show()
content=df['cp'].value_counts().to_frame().reset_index().rename(columns={'index':'c1','C1':'count'})
#content
fig = go.Figure([go.Pie(labels=content['c1'], values=content['cp']
                        ,hole=0.3)])  # can change the size of hole 

fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15)
fig.update_layout(title="Chest Pain Types",title_x=0.5)
fig.show()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
sns.countplot(x=df.target,hue=df.cp)
plt.legend(labels=['0: typical angina', '1: atypical angina','2: non-anginal pain','3: asymptomatic'])
dy=pd.DataFrame(df.groupby('cp')['target'].mean().reset_index().values,
                    columns=["chest_pain","target2"])
dy.head()
sns.barplot(dy.chest_pain,dy.target2)
plt.ylabel('rate of heart attack')
sns.distplot(df['thalach'],rug=True)
plt.show()
col='thalach'
d1=df[df['target']==0]
d2=df[df['target']==1]
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name=0,mode='lines+markers')
trace2 = go.Scatter(x=v2[col], y=v2["count"], name=1,mode='lines+markers')
data = [trace1, trace2]
layout={'title':"target over the person's maximum heart rate achieved",'xaxis':{'title':"Thalach"}}
fig = go.Figure(data, layout=layout)
iplot(fig)
col='fbs'
d1=df[df['target']==0]
d2=df[df['target']==1]
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name=0, marker=dict(color="#a678de"))
trace2 = go.Bar(x=v2[col], y=v2["count"], name=1, marker=dict(color="#6ad49b"))
data = [trace1, trace2]
layout={'title':"target over the person's fasting blood sugar ",'xaxis':{'title':"fbs(> 120 mg/dl, 1 = true; 0 = false)"}}
#layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
iplot(fig)
dy=pd.DataFrame(df.groupby('fbs')['target'].mean().reset_index().values,
                    columns=["fbs","target3"])
print(dy.head())
sns.barplot(dy.fbs,dy.target3)
plt.ylabel('rate of heart attack')
#plt.title('0=Female, 1=Male')
col='age'
d1=df[df['target']==0]
d2=df[df['target']==1]
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name=0, marker=dict(color="blue"),mode='lines+markers')
trace2 = go.Scatter(x=v2[col], y=v2["count"], name=1, marker=dict(color="red"),mode='lines+markers')
data = [trace1, trace2]
layout={'title':"Target With Respect to age",'xaxis':{'title':"Age"}}
fig = go.Figure(data, layout=layout)
iplot(fig)
#df=df.dropna()
#X = df.drop(['target'], axis = 1)
#y = df.target.values
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)
# Mean Encoding
cumsum = df.groupby('sex')['target'].cumsum() - df['target']
cumcnt = df.groupby('sex').cumcount()
df['sex'] = cumsum/cumcnt
df=df.dropna()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), 
                                                    df['target'], test_size = .2, random_state=10)
import xgboost as xgb
from xgboost import XGBClassifier
alg = XGBClassifier(learning_rate=0.01, n_estimators=2000, max_depth=8,
                        min_child_weight=0, gamma=0, subsample=0.52, colsample_bytree=0.6,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, 
                    seed=27, reg_alpha=5, reg_lambda=2, booster='gbtree',
            n_jobs=-1, max_delta_step=0, colsample_bylevel=0.6, colsample_bynode=0.6)
alg.fit(X_train, y_train)
print('train accuracy',alg.score(X_train, y_train))
print('test accuracy',alg.score(X_test,y_test))
import scikitplot as skplt
xgb_prob = alg.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, xgb_prob)
skplt.metrics.plot_ks_statistic(y_test, xgb_prob)
probas_list1 = [alg.predict_proba(X_test)]
xy=['xgb']
skplt.metrics.plot_calibration_curve(y_test,
                                      probas_list1,
                                    xy)
from yellowbrick.classifier import ClassificationReport,ConfusionMatrix
classes=[0,1]
visualizer = ClassificationReport(alg, classes=classes)
#visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
g = visualizer.poof()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=600,random_state=0, n_jobs= -1)
rf = rf.fit(X_train, y_train)
print('train accuracy',rf.score(X_train, y_train))
print('test accuracy',rf.score(X_test,y_test))
import scikitplot as skplt
rdf_prob = rf.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, rdf_prob)
skplt.metrics.plot_ks_statistic(y_test, rdf_prob)
plt.show()
probas_list1 = [rf.predict_proba(X_test)]
xy=['rdf']
skplt.metrics.plot_calibration_curve(y_test,
                                      probas_list1,
                                    xy)
from yellowbrick.classifier import ClassificationReport,ConfusionMatrix
classes=[0,1]
visualizer = ClassificationReport(rf, classes=classes)
#visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
g = visualizer.poof()
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = .2, random_state=10) #split the data
estimator = rf.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values

export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')
perm = PermutationImportance(rf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
base_features = df.columns.values.tolist()
base_features.remove('target')

feat_name = 'cp'
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
feat_name = 'ca'
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
feat_name = 'oldpeak'
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
feat_name = 'exang'
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
feat_name = 'fbs'
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)
#shap_values = explainer.shap_values(X_train.iloc[:50])
#shap.initjs()
#shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[:50])