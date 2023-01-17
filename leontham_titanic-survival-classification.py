#for data and data visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.graph_objs as go
import plotly.figure_factory as ff
%matplotlib inline
#%matplotlib inline
#Classification models
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#Data cleaning
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
#Model validation and preprocessing
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.feature_selection import SelectFromModel
#Helpers
import re
from datetime import datetime
from scipy.stats import boxcox
from collections import Counter

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

%config InlineBackend.figure_format = 'retina'
train = pd.read_csv('../input/train.csv')
val = pd.read_csv('../input/test.csv')
'Training set'
train.sample(5)
'Test set'
val.sample(5)
valID = val['PassengerId'].tolist()
full = pd.concat([train,val],ignore_index=True,sort=False)
full.sample(5)
features = full.drop('Survived',axis=1)
features = features[features.notnull().all(axis=1)].iloc[0].T.to_frame().reset_index()
features.columns = ['Features','Example']
features
missing = full.isna().sum().sort_values()

data= [go.Bar(
    x=missing.values,
    y=missing.index,
    orientation='h',
    opacity=0.8)]

layout = go.Layout(title='Missing values count in columns',
                   autosize=False,
                   xaxis=dict(title='Missing values count',tickangle=0,fixedrange=True),
                   yaxis=dict(title='Feature name',fixedrange=True,tickangle=-30))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
labels = ['Survived','Deceased']
values = [round(train['Survived'].mean(),2),round(1-train['Survived'].mean(),2)]
pie = go.Pie(labels=labels, values=values,opacity=0.9)
layout = go.Layout(title='Survival rate',
                   autosize=False)
fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)
if full.shape[0] == full['PassengerId'].nunique():
    print('PassengerID is unique.\nColumn will be removed')
    #full.drop('PassengerId',axis=1,inplace=True)
else:
    print('PassengerId is not unique')
full.sample(5)
groups = full[['Survived','Pclass']].groupby('Pclass').agg('mean')
data = [go.Bar(x=groups.index,
              y=groups.values.flatten(),
              text=[round(i,4) for i in groups.values.flatten()],
              textposition='outside',
              width=0.5)]
layout = go.Layout(autosize=False,title='Survival rate each Pclass',xaxis=dict(dtick=1))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
full['Title'] = full['Name'].apply(lambda name:re.findall(' ([a-zA-z]+)\.',name)[0])
print('There are {} unique titles. They are as follow:'.format(full['Title'].nunique()),', '.join(full['Title'].unique()))
#Visualize counts for each title
title_counts = full['Title'].value_counts()
x = title_counts.index
y = title_counts.values
data = [go.Bar(x=x, y=y, width = 0.5, marker=dict(color=y,opacity=0.6,showscale=True,colorscale='Portland'))]
layout = go.Layout(title='Counts per title',
                   autosize=False,
                   xaxis=dict(title='Title',tickangle=45,fixedrange=True),
                   yaxis=dict(dtick=100,title='Counts',range=[0,800],fixedrange=True))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
full['Title'] = full['Title'].replace(['Capt','Col','Dr','Major','Rev'], 'Officers')
full['Title'] = full['Title'].replace(['Lady','the Countess','Countess','Sir','Jonkheer','Dona','Don'],'Royalty')
full['Title'] = full['Title'].replace(['Miss','Ms','Mlle'],'Miss')
full['Title'] = full['Title'].replace(['Mrs','Mme'],'Mrs')

#Visualizing the surival rates of the title groups
title_counts = full[['Survived','Title']].groupby('Title').agg('mean')
x = title_counts.index
y = title_counts.values.flatten()
data = [go.Bar(x=x, y=y, width = 0.5, marker=dict(color=y,opacity=0.6,showscale=True,colorscale='Portland'))]
layout = go.Layout(title='Counts per title',
                   autosize=False,
                   xaxis=dict(title='Title',tickangle=45,fixedrange=True),
                   yaxis=dict(dtick=0.1,title='Counts',range=[0,1],fixedrange=True))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
full['Connected_Survival'] = 0
full['Surname'] = full['Name'].apply(lambda name:name.split(',')[0])
for grp, df_grp in full.groupby(['Surname','Fare']):
    if len(df_grp) > 1:
        for idx, row in df_grp.iterrows():
            count = Counter(df_grp.drop(idx)['Survived'])
            lived = count[1]
            died = count[0]
            passID = row['PassengerId']
            if lived+died!=0: 
                full.loc[full['PassengerId'] == passID, 'Connected_Survival'] = (lived-died)/(lived+died)
full.drop(['Name','Surname'],axis=1,inplace=True)
full.sample(5)
full['Sex'].unique()
full['Sex'] = full['Sex'].map({'male':1,'female':0})
full.sample(5)
#Check for missing variable
print('{} missing ages'.format(full['Age'].isna().sum()))
print('{:.2f}% missing values in age column'.format(full['Age'].isnull().sum()*100/full.shape[0]))

##Surival rate for each age group
cm = sns.light_palette("blue", as_cmap=True)
ages = train.loc[train['Age'].notna(),['Survived','Age']]
bins = [i for i in range(0,int(max(ages['Age']))+5,5)]
ages['Age'] = pd.cut(ages['Age'],bins)
ages = ages.groupby('Age').agg(['mean','count'])
ages_styled = ages.style.background_gradient(cmap=cm,subset=[('Survived','mean')])\
                 .format("{:.4f}",subset=[('Survived','mean')])\
                 .set_properties(subset=[('Survived','mean'),('Survived','count')], **{'width': '75px'})
ages_styled

##Visualize survival age over age group
data = [go.Bar(x=bins, y=ages[('Survived','mean')].values, width = 0.9, marker=dict(opacity=0.6))]
layout = go.Layout(title='Survival rate for each age group',
                   autosize=False,
                   xaxis=dict(title='Age group',tickangle=45,fixedrange=True,dtick=5),
                   yaxis=dict(dtick=0.1,title='Surival rate',range=[0,1],fixedrange=True))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
all_age = full[full['Age'].notna()]
columns = ['Pclass','Sex','Title']
for col in columns:
    groupings = all_age.groupby(col)
    keys = groupings.groups.keys()
    lst = [go.Box(x=groupings.get_group(key)['Age'], name='{} {}'.format(col,key)) for key in keys] 
    layout = go.Layout(title=col,
                   autosize=False,
                   xaxis=dict(title=col,tickangle=0,fixedrange=True),
                   yaxis=dict(title='Age',fixedrange=True))
    fig = go.Figure(data=lst,layout=layout)
    py.iplot(fig)
full['Age'] = full.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'].fillna(full['Age'].median(),inplace=True) #In case there are still NaN
full['Age'] = full['Age'].apply(int) #Getting rid of the .5
print('Number of missing values: {}'.format(full['Age'].isna().sum()))
full['Child'] = np.where(full['Age']<12,1,0)
full.sample(5)
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1 #1 to include the passenger himself
full.sample(5)
#Visualizing the surival rates of the family size
famsize = full.loc[full['Survived'].notna(),['FamilySize','Survived']].groupby('FamilySize').agg('mean')
x = famsize.index
y = famsize.values.flatten()
data = [go.Bar(x=x, y=y, width = 0.5, marker=dict(color=y,opacity=0.6,showscale=True,colorscale='Portland'))]
layout = go.Layout(title='Survival rate per family Size',
                   autosize=False,
                   xaxis=dict(title='Family Size',dtick=1,fixedrange=True,range=[0,max(full['FamilySize'])]),
                   yaxis=dict(dtick=0.1,title='Survival rate',range=[0,1],fixedrange=True))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
full['TicketSize'] = full.groupby('Ticket')['Ticket'].transform('count')
full[full['FamilySize']<full['TicketSize']].sort_values('Ticket').head(9)
full['GroupSize'] = full[['FamilySize','TicketSize']].max(axis=1)
full.sample(5)

#Visualizing the surival rates of the family size
grpsize = full.loc[full['Survived'].notna(),['GroupSize','Survived']].groupby('GroupSize').agg('mean')
x = grpsize.index
y = grpsize.values.flatten()
data = [go.Bar(x=x, y=y, width = 0.5, marker=dict(color=y,opacity=0.6,showscale=True,colorscale='Portland'))]
layout = go.Layout(title='Survival rate per Group Size',
                   autosize=False,
                   xaxis=dict(title='Group Size',dtick=1,fixedrange=True,range=[0,max(full['FamilySize'])]),
                   yaxis=dict(dtick=0.1,title='Survival rate',range=[0,1],fixedrange=True))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
full['IsAlone'] = np.where(full['GroupSize']==1,1,0)
full['SmallGroup'] = np.where((2<=full['GroupSize']) & (full['GroupSize']<=4),1,0)
full['LargeGroup'] = np.where(full['GroupSize']>4,1,0)
full.sample(5)
#full['Connected_Survival'] = 0.5 
for grp, df_grp in full.groupby('Ticket'):
    if len(df_grp) > 1:
        for idx, row in df_grp.iterrows(): 
            if row['Connected_Survival'] != 0:
                continue
            count = Counter(df_grp.drop(idx)['Survived'])
            lived = count[1]
            died = count[0]
            passID = row['PassengerId']
            if lived+died!=0: 
                full.loc[full['PassengerId'] == passID, 'Connected_Survival'] = (lived-died)/(lived+died)
full.drop(['Ticket','FamilySize','TicketSize','PassengerId'],axis=1,inplace=True)
full.sample(5)
#Check for missing variable
print('{} missing ages'.format(full['Fare'].isna().sum()))
print('{:.2f}% missing values in Fare column'.format(full['Fare'].isnull().sum()*100/full.shape[0]))
full['Fare'].fillna(full['Fare'].median(),inplace=True)
##Visualize distribution of fares
fares = [list(full['Fare'])]
fig = ff.create_distplot(fares, ['Fares'], show_hist=False, show_rug=False)
fig['layout'].update(title='Fares distribution',
                     autosize=False,
                     yaxis=dict(range=[0,0.04],dtick=0.0025,showgrid=True),
                     xaxis=dict(showgrid=False,title='Fares',dtick=50))
py.iplot(fig)
#Perform boxcox transformation
full['Fare'] = boxcox((1+full['Fare']))[0]
fares = [list(full['Fare'])]
fig = ff.create_distplot(fares, ['Fares'], show_hist=False, show_rug=False)
fig['layout'].update(title='Fares distribution',
                     autosize=False,
                     yaxis=dict(showgrid=True),
                     xaxis=dict(showgrid=False,range=[-1,6]))
py.iplot(fig)
full.sample(5)
full['Cabin'].unique()
full['Cabin'].fillna('X',inplace=True)
full['Cabin'] = full['Cabin'].map(lambda x:x[0])

#Visualizing the surival rates of each cabin
cabins = full.loc[full['Survived'].notna(),['Cabin','Survived']].groupby('Cabin').agg('mean')
x = cabins.index
y = cabins.values.flatten()
data = [go.Bar(x=x, y=y, width = 0.5, marker=dict(color=y,opacity=0.6,showscale=True,colorscale='Portland'))]
layout = go.Layout(title='Survival rate for each cabin',
                   autosize=False,
                   xaxis=dict(title='Cabin',fixedrange=True),
                   yaxis=dict(dtick=0.1,title='Survival rate',range=[0,1],fixedrange=True))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
full.sample(5)
#Check for missing variable
print('{} missing ages'.format(full['Embarked'].isna().sum()))
print('{:.2f}% missing values in age column'.format(full['Embarked'].isnull().sum()*100/full.shape[0]))
full['Embarked'].fillna(full['Embarked'].mode(),inplace=True)
full = pd.get_dummies(full,columns=['Cabin','Embarked','Title'])
full.sample(5)
print('Total independent variables: {}'.format(full.shape[1]-1))
train_set = full[full['Survived'].notna()]
train_features = train_set.drop('Survived',axis=1)
train_target = train_set['Survived']
val_set = full[full['Survived'].isna()].drop('Survived',axis=1)
x_train,x_test,y_train,y_test = train_test_split(train_features,train_target,test_size=0.1,random_state=0,stratify=train_target)
RFC = RandomForestClassifier(n_estimators=100, max_features='sqrt',random_state=0)
RFC = RFC.fit(train_features, train_target)
features = pd.DataFrame(index=train_features.columns)
features['Importance'] = RFC.feature_importances_
features = features.sort_values('Importance',ascending=True)

#Visualing feature importance
data= [go.Bar(
    x=features.values.flatten(),
    y=features.index,
    orientation='h',
    opacity=0.8)]

layout = go.Layout(title='Feature Importance',
                   autosize=True,
                   xaxis=dict(title='Features',tickangle=0,fixedrange=True),
                   yaxis=dict(title='Importance',fixedrange=True,tickangle=0),
                   margin=dict(l=120,t=0))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
##Reducing model
model = SelectFromModel(RFC, prefit=True, threshold='median')
columns = list(train_features.columns[model.get_support()])
#Reduce training features
data = model.transform(train_features)
train_features_reduced = pd.DataFrame(data,columns=columns)
#Reduce validation features
data = model.transform(val_set)
val_set_reduced = pd.DataFrame(data,columns=columns)
import warnings
warnings.filterwarnings("ignore")
#InteractiveShell.ast_node_interactivity = 'last_expr'

classifiers = {'LogReg': LogisticRegression(),
               'RidgeClassifier': RidgeClassifierCV(),
               'KNN': KNeighborsClassifier(),
               'SVC': SVC(gamma='auto'),
               'GaussianNB': GaussianNB(),
               'DecisionTree': DecisionTreeClassifier(),
               'RandomForest': RandomForestClassifier(n_estimators=100),
               'AdaBoost': AdaBoostClassifier(n_estimators=100),
               'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
               'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
               'BaggingClassifier': BaggingClassifier(n_estimators=100),
               'XGB': XGBClassifier(),
               'LDA': LinearDiscriminantAnalysis()}

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'f1_score' : make_scorer(f1_score)}

cv_split = ShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9, random_state=0)
selection_cols = ['Classifier','Mean Train Accuracy','Mean Test Accuracy','Mean F1 train','Mean F1 Test','Prediction']#,'Train Accuracies','Test Accuracies','Train F1 Scores','Test F1 Scores'] 

classifiers_summary = pd.DataFrame(columns=selection_cols)

for name,classifier in classifiers.items():
    print('Validating ',name)
    cv = cross_validate(classifier,train_features,train_target,return_train_score=True,cv=cv_split,scoring=scoring)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    cv_calc = [name,
               cv['train_accuracy'].mean(),
               cv['test_accuracy'].mean(),
               cv['train_f1_score'].mean(),
               cv['test_f1_score'].mean(),
               pred
               #cv['train_accuracy'],
               #cv['test_accuracy'],
               #cv['train_f1_score'],
               #cv['test_f1_score']
              ]
    cv_calc_s = pd.Series(cv_calc,index=selection_cols)
    classifiers_summary = classifiers_summary.append(cv_calc_s,ignore_index=True)
    
classifiers_summary = classifiers_summary.sort_values('Mean F1 Test',ascending=False)

classifiers_summary_styled = classifiers_summary[['Classifier','Mean Train Accuracy','Mean Test Accuracy','Mean F1 train','Mean F1 Test']].style.highlight_max(axis=0).set_properties(**{'width': '150px'})
classifiers_summary_styled

#Comparison visualization
y = list(classifiers_summary['Classifier'].values)[::-1]

trace1 = go.Bar(
    x=(list(classifiers_summary['Mean F1 Test'].values)[::-1]),
    y=y,
    name='Test',
    marker=dict(color='red'),
    orientation='h',
    opacity=0.7)
    
trace2 = go.Bar(
    x=(list(classifiers_summary['Mean F1 train'].values)[::-1]),
    y=y,
    name='Train',
    marker=dict(color='lightgrey'),
    orientation='h',
    opacity=0.8)

data = [trace1,trace2]
layout = go.Layout(title='Mean F1 Scores of classifiers',
                   autosize=True,
                   xaxis=dict(title='Mean F1 Score',tickangle=0,fixedrange=True,range=[0,0.9],dtick=0.05),
                   yaxis=dict(fixedrange=True,tickangle=-30))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
warnings.filterwarnings("ignore")
#InteractiveShell.ast_node_interactivity = 'last_expr'

selected_classifiers = [('RidgeClassifier',RidgeClassifierCV()),
                        ('LDA',LinearDiscriminantAnalysis()),
                        ('LogReg',LogisticRegression()),
                        ('GradientBoosting',GradientBoostingClassifier()),
                        ('XGB',XGBClassifier())]

grid_param = [ #RidgeClassifier
             [{
              }],
               #LDA
            [{'solver': ['svd', 'lsqr'],
             }],
               #LogReg
            [{ 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
               'random_state': [0], 
               'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
             }],
               #GradientBoostingClassifier
            [{'loss': ['deviance', 'exponential'],
               'learning_rate':[0.1,0.05,0.001],
               'n_estimators':[10,50,100,300],
               'max_depth':[2,3,5,10],
               'random_state':[0]
             }],
               #XGB
            [{'learning_rate':np.arange(0.1, 0.6, 0.1),
              'max_depth':np.arange(2, 11, 1),
              'n_estimators':[10,50,100,300],
              'random_state':[0]
             }]]

for i in range(5):
    start = datetime.now()
    print('Now searching for {}'.format(selected_classifiers[i][1].__class__.__name__))
    grid_search = GridSearchCV(estimator = selected_classifiers[i][1], param_grid = grid_param[i], cv = cv_split, scoring = 'f1')
    grid_search = grid_search.fit(train_features, train_target)
    best_parameters = grid_search.best_params_
    selected_classifiers[i][1].set_params(**best_parameters)
    elasped = (datetime.now() - start).total_seconds()
    print('Best parameters found for {} is {} in {}minutes {}seconds'.format(selected_classifiers[i][1].__class__.__name__,best_parameters,int(elasped//60),elasped%60))

corr = classifiers_summary[['Classifier','Prediction']].head(5)
corr_df = pd.DataFrame()
for idx,row in corr.iterrows():
    corr_df[row['Classifier']]=row['Prediction']

sns.heatmap(corr_df.corr(),annot=True,cmap='coolwarm')
warnings.filterwarnings("ignore")
votings = VotingClassifier(estimators=selected_classifiers,voting='hard',n_jobs=-1)
hard = cross_validate(votings,train_features,train_target,cv=cv_split,scoring=scoring)
print('Hard voting\nAccuracy: {}\nF1: {}'.format(hard['test_accuracy'].mean(),hard['test_f1_score'].mean())) 

votings = votings.fit(x_train, y_train)
val_pred = pd.Series(votings.predict(val_set), name="Survived").astype(int)
submission = pd.read_csv('../input/gender_submission.csv')
results = pd.concat([val['PassengerId'],val_pred],axis=1)
results.to_csv("titanic.csv",index=False)


