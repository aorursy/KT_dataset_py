import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
#plt.style.use('bmh')
plt.style.use('ggplot')
import re as re

from sklearn import preprocessing

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#create pandas data frames and create also a single total df so that changes can be made on all of the datapoints.

df_train = pd.read_csv("../input/train.csv", header = 0, dtype={'Age': np.float64})
df_test = pd.read_csv("../input/test.csv", header = 0, dtype={'Age': np.float64})
df_total = [df_train, df_test]
print(df_train.shape)
print(df_test.shape)

df_train.head()
df_train.info()
#Pclass
print('--PClass')
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

#Sex
print("\n",'--Sex')
print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

#SibSp
print("\n",'--SibSp')
print(df_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())

#Parch
print("\n",'--Parch')
print(df_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
#Create FamSize feature
for dataset in df_total:
    dataset['FamSize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
print(df_train[['FamSize', 'Survived']].groupby(['FamSize'], as_index=False).mean())
#check names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

for dataset in df_total:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(df_train['Title'], df_train['Sex']))
for dataset in df_total:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona'], '_rest')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
feats = ['FamSize', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Title']
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(20,10))

start = 0
for j in range (2):
    for i in range(3):
        if start == len(feats):
            break
        sns.barplot(x=feats[start], y='Survived', data=df_train, ax=ax[j,i])
        
        start+=1
fig, ax = plt.subplots(figsize=(15,9))
ax = sns.distplot(df_train[df_train['Survived']==1].Age.dropna(), bins=20, label = 'Survived', ax = ax)
ax = sns.distplot(df_train[df_train['Survived']==0].Age.dropna(), bins=20, label = 'Not Survived', ax = ax)
ax.legend()
plot1 = ax.set_ylabel('Kernal Density Estimation')

fig, ax = plt.subplots(figsize=(15,9))
plot2 = sns.violinplot(x='Sex', y='Age', hue = 'Survived', data=df_train, split=True)
#Print the percentage of missing values
missing = df_train['Age'].isnull().sum()
print(missing, ' values are missing')
print(missing / len(df_train) *100)
#get all the ages:
ages = np.concatenate((df_test['Age'].dropna(), df_train['Age'].dropna()), axis=0)

std_ages = ages.std()
mean_ages = ages.mean()
train_nas = np.isnan(df_train["Age"])
test_nas = np.isnan(df_test["Age"])
np.random.seed(122)
impute_age_train  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = train_nas.sum())
impute_age_test  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = test_nas.sum())
df_train["Age"][train_nas] = impute_age_train
df_test["Age"][test_nas] = impute_age_test
ages_imputed = np.concatenate((df_test["Age"],df_train["Age"]), axis = 0)

#df_train['Age*Class'] = df_train['Age']*df_train['Pclass']
#df_test['Age*Class'] = df_test['Age']*df_test['Pclass']


fig, ax = plt.subplots(figsize=(20,10))
ax = sns.kdeplot(ages_imputed, label = 'After imputation')
ax = sns.kdeplot(ages, label = 'Before imputation')
ax.legend()
plot1 = ax.set_ylabel('Kernal Density Estimation')


#Print the percentage of missing values again
missing = df_train['Age'].isnull().sum()
print(missing, ' values are missing')
print(missing / len(df_train) *100, '%')
# We skip this:
'''
labels = [1,2,3,4]
bins = [0, 12, 25, 55, 100]
df_train['age_group'] = pd.cut(df_train['Age'], bins=bins, labels=labels)
df_test['age_group'] = pd.cut(df_test['Age'], bins=bins, labels=labels)
df_train['age_group'] = df_train['age_group'].astype('int64')
df_test['age_group'] = df_test['age_group'].astype('int64')

df_train.head()

'''

# display the unique values
df_train['Embarked'].unique()
#Print the percentage of missing values from Embarked
missing = df_train['Embarked'].isnull().sum()
print(missing, ' Embarked values are missing')
print(missing / len(df_train) *100, '%')
sns.barplot(x=df_train['Embarked'], y='Survived', data=df_train)
for dataset in df_total: 
    dataset['Embarked'] = dataset['Embarked'].fillna('C')
#Print the percentage of missing values from Embarked
missing = df_train['Embarked'].isnull().sum()
print(missing, ' Embarked values are missing')
print(missing / len(df_train) *100, '%')    

df_train.info()
# Check the unique values of Sex
print(df_train['Sex'].unique())

#Check unique values of Embarked
print(df_train['Embarked'].unique())

#Check unique values of Title
print(df_train['Title'].unique())
# instead of label encode the following feats, we one hot encode them using the pandas version, get_dummies:
df_sex = pd.get_dummies(df_train['Sex'])
df_embarked = pd.get_dummies(df_train['Embarked'])
df_title = pd.get_dummies(df_train['Title'])
df_sex2 = pd.get_dummies(df_test['Sex'])
df_embarked2 = pd.get_dummies(df_test['Embarked'])
df_title2 = pd.get_dummies(df_test['Title'])
drops = ['Sex', 'Embarked', 'Ticket', 'Title', 'PassengerId', 'Name', 'Cabin']
df_train = df_train.drop(drops, axis=1)
df_test = df_test.drop(drops, axis=1)

df_train = pd.concat([df_train, df_sex, df_embarked, df_title], axis=1)
df_test = pd.concat([df_test, df_sex2, df_embarked2, df_title2], axis=1)

df_train.head()
# we are not doing this:

'''
# import the label encoder.
leSex = preprocessing.LabelEncoder()
leSex.fit(df_train['Sex'])
df_train['Sex']=leSex.transform(df_train['Sex'])
df_test['Sex']=leSex.transform(df_test['Sex'])

leEmbarked = preprocessing.LabelEncoder()
leEmbarked.fit(df_train['Embarked'])
df_train['Embarked']=leEmbarked.transform(df_train['Embarked'])
df_test['Embarked']=leEmbarked.transform(df_test['Embarked'])

leTitle = preprocessing.LabelEncoder()
leTitle.fit(df_train['Title'])
df_train['Title']=leTitle.transform(df_train['Title'])
df_test['Title']=leTitle.transform(df_test['Title'])


#Actually, we should use Dummies to one hot encode the Title and the Embarked Feat

#df_dummies = pd.get_dummies(df_train[['Title', 'Embarked']])

#df_train = pd.concat([df_train, df_dummies], axis=1)

df_train.head()
'''
# not doing this either:

'''

# drop feats and plot correlation
drop_feats = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_train = df_train.drop(drop_feats, axis = 1)
df_test = df_test.drop(drop_feats, axis = 1)

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
            
'''
# create a new df based on the new feats after one hot encoding:
df_train_final = df_train[['Survived', 'Pclass','Age','Fare','FamSize','female','male','C','Q','S','Master','Miss','Mr','Mrs','_rest']]
df_test_final = df_test[['Pclass','Age','Fare','FamSize','female','male','C','Q','S','Master','Miss','Mr','Mrs','_rest']]
train = df_train_final.values
print(train)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier


classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier()
   ]

log_cols = ["Classifier", "f1_score"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = f1_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.figure(figsize=(15,9))
plt.xlabel('f1_score')
plt.title('Classifier f1_score')

sns.set_color_codes("muted")
sns.barplot(x='f1_score', y='Classifier', data=log, color="b")
rf_clf = RandomForestClassifier()
ada_clf = AdaBoostClassifier()
gb_clf = GradientBoostingClassifier()
tr_clf = ExtraTreesClassifier()

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
from sklearn.model_selection import cross_val_score

X = train[0::, 1::]
y = train[0::, 0]


for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
rf_feature  = rf_clf.fit(X_train, y_train).feature_importances_
ada_feature = ada_clf.fit(X_train, y_train).feature_importances_
gb_feature = gb_clf.fit(X_train, y_train).feature_importances_
tr_feature = tr_clf.fit(X_train, y_train).feature_importances_
    
train_cols = df_train_final
del train_cols['Survived']
cols = train_cols.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,
                                   'Random Forest feature importances': rf_feature,
                                   'AdaBoost feature importances': ada_feature,
                                   'Gradient Boost feature importances': gb_feature,
                                  'Extra Trees feature importances': tr_feature})
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Extra Trees feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['Extra Trees feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees feature importances',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
feature_dataframe.head(10)
# Finally but not necessary for this notebook I make a predictinon on the test data set
df_test_final = df_test_final.dropna()
predictions_on_test = gb_clf.predict(df_test_final)
#confusion matrix

from sklearn.metrics import confusion_matrix

conf = confusion_matrix(gb_clf.predict(X_test), y_test)
print(conf)
sns.heatmap(conf, annot=True,)
plt.title('Confusion matrix')
plt.ylabel('True labels')
plt.xlabel('Pred labels')
df_age = pd.read_csv("../input/train.csv", header = 0, dtype={'Age': np.float64})
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

df_age['Title'] = df_age['Name'].apply(get_title)

df_age['Title'] = df_age['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona'], '_rest')
df_age['Title'] = df_age['Title'].replace(['Mlle', 'Ms'], 'Miss')
df_age['Title'] = df_age['Title'].replace('Mme', 'Mrs')
df_age['Embarked'] = df_age['Embarked'].fillna('C')

df_sex3 = pd.get_dummies(df_age['Sex'])
df_embarked3 = pd.get_dummies(df_age['Embarked'])
df_title3 = pd.get_dummies(df_age['Title'])

df_age = pd.concat([df_age, df_sex3, df_embarked3, df_title3], axis=1)
df_age.head()
missing = df_age['Age'].isnull().sum()
print(missing, ' values are missing')
print(missing / len(df_train) *100)
df_age_final = df_age[['Survived', 'Pclass','Fare', 'female','male','C','Q','S','Master','Miss','Mr','Mrs','_rest', 'Age']]
test_set = df_age_final.loc[df_age['Age'].isnull()]
test_set = test_set.drop(['Age'], axis=1)
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
dataset = df_age_final.dropna()
dataset = dataset.as_matrix()
X = dataset[:, :-1]
y = dataset[:, -1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) Let's not do that since the dataset is already a bit small...

regr = GradientBoostingRegressor()
rf_feature_age  = regr.fit(X, y).feature_importances_
#regr.fit(X_train, y_train)
train_cols = df_age_final
del train_cols['Age']
cols = train_cols.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,
                                   'Regressor feature importances': rf_feature_age})
feature_dataframe.head()
preds = regr.predict(test_set)
sns.kdeplot(preds, label = 'ages')
#dataset = df_age_final.dropna()
#dataset = pd.DataFrame(preds)
#preds = pd.DataFrame(preds)
#df_new_ = pd.concat([dataset, preds])
df_age_final = df_age[['Survived', 'Pclass','Fare', 'female','male','C','Q','S','Master','Miss','Mr','Mrs','_rest', 'Age']]
df_age_final
df_nas = np.isnan(df_train["Age"])
df_age_final["Age"][train_nas] = regr.predict(df_age_final.drop(['Age'], axis=1)) 
df_age_final.head()
classifiers = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier()
   ]

log_cols = ["Classifier", "f1_score"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

df_age_final_ = df_age_final.values
X = df_age_final_[0::, 1::]
y = df_age_final_[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = f1_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.figure(figsize=(15,9))
plt.xlabel('f1_score')
plt.title('Classifier f1_score')

sns.set_color_codes("muted")
sns.barplot(x='f1_score', y='Classifier', data=log, color="b")
rf_clf = RandomForestClassifier()
ada_clf = AdaBoostClassifier()
gb_clf = GradientBoostingClassifier()
tr_clf = ExtraTreesClassifier()

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
from sklearn.model_selection import cross_val_score


df_age_final_ = df_age_final.values
X = df_age_final_[0::, 1::]
y = df_age_final_[0::, 0]


for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
rf_feature  = rf_clf.fit(X_train, y_train).feature_importances_
ada_feature = ada_clf.fit(X_train, y_train).feature_importances_
gb_feature = gb_clf.fit(X_train, y_train).feature_importances_
tr_feature = tr_clf.fit(X_train, y_train).feature_importances_
print(tr_feature.shape)
print(rf_feature.shape)
    
train_cols = df_age_final
del train_cols['Survived']

#del train_cols['Survived']
cols = train_cols.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,
                                   'Random Forest feature importances': rf_feature,
                                   'AdaBoost feature importances': ada_feature,
                                   'Gradient Boost feature importances': gb_feature,
                                   'Extra Trees feature importances': tr_feature})
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Extra Trees feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 15,

        color = feature_dataframe['Extra Trees feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees feature importances',
    hovermode= 'closest',

    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

