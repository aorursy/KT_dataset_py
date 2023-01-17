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
df=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df
import seaborn as sns
sns.lmplot(x='time',y='DEATH_EVENT',data=df,aspect=6)
import seaborn as sns
sns.heatmap(data=df.corr())
corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
features=corr.nlargest(5).index
features
X=df[features]
y=df['DEATH_EVENT']
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
basic_model = DecisionTreeRegressor(random_state=1)
basic_model.fit(train_X, train_y)
predictions=basic_model.predict(val_X)
pd.options.mode.chained_assignment = None
frame=val_X
frame['death']=val_y
frame['predictions']=predictions
frame['correct?']=frame['death']==frame['predictions']
frame['correct?'].value_counts()
101/120
X=df.drop(columns='DEATH_EVENT')
y=df['DEATH_EVENT']

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
basic_model = DecisionTreeRegressor(random_state=1)
basic_model.fit(train_X, train_y)
predictions=basic_model.predict(val_X)

frame=val_X
frame['death']=val_y
frame['predictions']=predictions
frame['correct?']=frame['death']==frame['predictions']
frame['correct?'].value_counts()
p=frame['correct?'].value_counts()
p[True]/(p[False]+p[True])
def n_inputs_decision_tree(n):
    corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
    features=corr.nlargest(n).index
    X=df[features]
    y=df['DEATH_EVENT']
    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
    basic_model = DecisionTreeRegressor(random_state=1)
    basic_model.fit(train_X, train_y)
    predictions=basic_model.predict(val_X)

    val_X['death']=val_y
    val_X['predictions']=predictions
    val_X['correct?']=val_X['death']==val_X['predictions']
    return val_X['correct?'].value_counts()
dt_tracker=pd.DataFrame()
for i in range(1,12):
    data=n_inputs_decision_tree(i)
    dt_tracker=dt_tracker.append({'n':i,'true':data[True],'false':data[False],'percent':data[True]/(data[True]+data[False])},ignore_index=True)
dt_tracker.sort_values(by='true',ascending=False)
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
features=corr.nlargest(5).index
X=df[features]
y=df['DEATH_EVENT']
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

rf_model.fit(train_X,train_y)
rf_val_predictions = rf_model.predict(val_X)
val_X['death']=val_y
val_X['raw predictions']=rf_val_predictions
val_X['predictions']=rf_val_predictions.round()
val_X['correct?']=val_X['predictions']==val_X['death']
val_X['correct?'].value_counts()
def n_inputs_rf_regressor(n):
    corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
    features=corr.nlargest(n).index
    X=df[features]
    y=df['DEATH_EVENT']
    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
    
    rf_model.fit(train_X,train_y)
    rf_val_predictions = rf_model.predict(val_X)

    val_X['death']=val_y
    val_X['raw predictions']=rf_val_predictions
    val_X['predictions']=rf_val_predictions.round()
    val_X['correct?']=val_X['predictions']==val_X['death']
    return val_X['correct?'].value_counts()
rfr_tracker=pd.DataFrame()
for i in range(1,len(df.columns)):
    data=n_inputs_rf_regressor(i)
    rfr_tracker=rfr_tracker.append({'n':i,'true':data[True],'false':data[False],'percent':data[True]/(data[True]+data[False])},ignore_index=True)
rfr_tracker.sort_values(by='true',ascending=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
features=corr.nlargest(5).index
X=df[features]
y=df['DEATH_EVENT']

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

clf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=1)
clf.fit(train_X, train_y)

y_pred=clf.predict(val_X)
accuracy_score(val_y,y_pred)
def n_inputs_rf_classifier(n):
    corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
    features=corr.nlargest(n).index
    X=df[features]
    y=df['DEATH_EVENT']

    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

    clf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=1)
    clf.fit(train_X, train_y)

    y_pred=clf.predict(val_X) 
    val_X['death']=val_y
    val_X['predictions']=y_pred
    val_X['correct?']=val_X['death']==val_X['predictions']
    return val_X['correct?'].value_counts()
rfc_tracker=pd.DataFrame()
for i in range(1,12):
    data=n_inputs_rf_classifier(i)
    rfc_tracker=rfc_tracker.append({'n':i,'true':data[True],'false':data[False],'percent':data[True]/(data[True]+data[False])},ignore_index=True)
rfc_tracker.sort_values(by='true',ascending=False)
from sklearn.neighbors import KNeighborsClassifier

corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
features=corr.nlargest(5).index
X=df[features]
y=df['DEATH_EVENT']

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
kgmodel = KNeighborsClassifier(n_jobs=-1)
kgmodel.fit(train_X, train_y)

#Getting the accuracy
pred_gen = kgmodel.predict(val_X)
accuracy_score(pred_gen, val_y)
def n_inputs_knn(n):
    corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
    features=corr.nlargest(n).index
    X=df[features]
    y=df['DEATH_EVENT']

    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

    kgmodel = KNeighborsClassifier(n_jobs=-1)
    kgmodel.fit(train_X, train_y)

    pred_gen = kgmodel.predict(val_X)
    val_X['death']=val_y
    val_X['predictions']=pred_gen
    val_X['correct?']=val_X['death']==val_X['predictions']
    return val_X['correct?'].value_counts()
knn_tracker=pd.DataFrame()
for i in range(1,12):
    data=n_inputs_knn(i)
    knn_tracker=knn_tracker.append({'n':i,'true':data[True],'false':data[False],'percent':data[True]/(data[True]+data[False])},ignore_index=True)
knn_tracker.sort_values(by='true',ascending=False)
corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
features=corr.nlargest(5).index
X=df[features]
y=df['DEATH_EVENT']
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

from sklearn import metrics
neigh = KNeighborsClassifier(n_neighbors = 10).fit(train_X,train_y)
ypred=neigh.predict(val_X)
metrics.accuracy_score(val_y, ypred)
for i in range(1,20):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(train_X,train_y)
    ypred=neigh.predict(val_X)
    print(i,' neighbors:  ',metrics.accuracy_score(val_y, ypred))
neigh = KNeighborsClassifier(n_neighbors = 6,algorithm='brute').fit(train_X,train_y)
ypred=neigh.predict(val_X)
metrics.accuracy_score(val_y, ypred)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

# fitting the model to the training data
lr.fit(train_X, train_y)

# use the model to predict on the testing data
lr.predict(val_X)

# Printing the accuracy of the model
score = lr.score(val_X, val_y)
score
def n_inputs_lr(n):
    corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
    features=corr.nlargest(n).index
    X=df[features]
    y=df['DEATH_EVENT']

    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

    lr=LogisticRegression()
    lr.fit(train_X, train_y)
    pred=lr.predict(val_X)

    val_X['death']=val_y
    val_X['predictions']=pred
    val_X['correct?']=val_X['death']==val_X['predictions']
    return val_X['correct?'].value_counts()
lr_tracker=pd.DataFrame()
for i in range(1,12):
    data=n_inputs_lr(i)
    lr_tracker=lr_tracker.append({'n':i,'true':data[True],'false':data[False],'percent':data[True]/(data[True]+data[False])},ignore_index=True)
lr_tracker.sort_values(by='true',ascending=False)
# from sklearn import preprocessing
# le_sex = preprocessing.LabelEncoder()
# le_sex.fit(['F','M'])
# X[:,1] = le_sex.transform(X[:,1]) 


# le_BP = preprocessing.LabelEncoder()
# le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
# X[:,2] = le_BP.transform(X[:,2])


# le_Chol = preprocessing.LabelEncoder()
# le_Chol.fit([ 'NORMAL', 'HIGH'])
# X[:,3] = le_Chol.transform(X[:,3]) 

# X[0:5]
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="gini")
drugTree.fit(train_X,train_y)
predTree = drugTree.predict(val_X)
metrics.accuracy_score(val_y, predTree)
def n_inputs_dtc(n):
    corr=abs(df.corr()['DEATH_EVENT'].drop(labels='DEATH_EVENT'))
    features=corr.nlargest(n).index
    X=df[features]
    y=df['DEATH_EVENT']

    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

    drugTree = DecisionTreeClassifier(criterion="gini")
    drugTree.fit(train_X,train_y)
    predTree = drugTree.predict(val_X)

    val_X['death']=val_y
    val_X['predictions']=predTree
    val_X['correct?']=val_X['death']==val_X['predictions']
    return val_X['correct?'].value_counts()
dtc_tracker=pd.DataFrame()
for i in range(1,12):
    data=n_inputs_dtc(i)
    dtc_tracker=dtc_tracker.append({'n':i,'true':data[True],'false':data[False],'percent':data[True]/(data[True]+data[False])},ignore_index=True)
dtc_tracker.sort_values(by='true',ascending=False)
drugTree.get_params().keys()
from sklearn.model_selection import GridSearchCV

parameters={'min_samples_split' : range(10,500,20),'min_samples_leaf':[1, 5, 10, 20, 50, 100],'max_depth':np.arange(1, 21),
           'criterion':['gini','entropy']}
gd_sr = GridSearchCV(estimator=drugTree,
                     param_grid=parameters)

gd_sr.fit(train_X, train_y)
best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)
pred=gd_sr.best_estimator_.predict(val_X)
pd.Series(pred==val_y).value_counts()