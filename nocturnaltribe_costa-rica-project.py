# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head(2)
df.info()
df.describe()
sns.countplot(x='Target',data=df)
#
# Distribution of wealth in rural vs urban areas
#
df['areadscr'] = df['area1'].apply(lambda x: 'urban' if x==1 else 'rural') 
sns.countplot(x='Target',hue='areadscr',data=df)
df['regiondscr'] = list(
map( lambda x1,x2,x3,x4,x5,x6: 
                    'Central' if x1==1 
               else 'Chorotega' if x2==1
               else 'Pacafico central' if x3==1
               else 'Brunca' if x4==1
               else 'Huetar Atlantica' if x5==1
               else 'Huetar Norte',
                    df['lugar1'],df['lugar2'],df['lugar3'],df['lugar4'],df['lugar5'],df['lugar6'] )
)
sns.factorplot(x='Target',hue='regiondscr',data=df, kind='count', log=True, size=5, aspect=1.8)
df_d=df.groupby(['Target','regiondscr']).mean()
fp=df_d['SQBdependency'].unstack()#.fillna(0)
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis',robust=True,annot=True)
plt.figure(figsize=(12,6))
sns.clustermap(fp,cmap='viridis',annot=True)
for i in list(df.drop(['Id','idhogar','areadscr','regiondscr','dependency','edjefe','edjefa','rez_esc'],axis=1).columns):
    x = np.corrcoef(df['SQBdependency'], df[i])[0,1]
    if abs(x) > 0.15 : 
        print( i, x )
sns.factorplot(x='hogar_mayor', y='SQBdependency', data=df,kind='bar')
df_d=df.groupby(['age','SQBdependency']).mean()
fp=df_d['Target'].unstack().fillna(0)
#plt.figure(figsize=(12,6))
#ax = sns.heatmap(fp,cmap='viridis',robust=True)


yticks = fp.index
keptticks = yticks[::int(len(yticks)/6)]
yticks = ['' for y in yticks]
yticks[::int(len(yticks)/6)] = keptticks

xticks = fp.columns
keptticks = xticks[::int(len(xticks)/10)]
xticks = ['' for y in xticks]
xticks[::int(len(xticks)/10)] = keptticks

plt.figure(figsize=(12,6))
sns.heatmap(fp,linewidth=0,
            yticklabels=yticks, xticklabels=xticks,
            cmap='viridis',robust=True)

# This sets the yticks "upright" with 0, as opposed to sideways with 90.
plt.yticks(rotation=0) 

plt.show()
plt.figure(figsize=(12,6))
sns.clustermap(fp,cmap='viridis')
np.corrcoef(df['SQBdependency'], df['hhsize'])
g = sns.regplot(  x='hhsize',y='SQBdependency',
                  fit_reg=False, data=df )
g.set_xscale('log')
g.set_yscale('log')
plt.ylim(0.01,100)
sns.lmplot(  x='hhsize',y='SQBdependency', hue='Target',
                  fit_reg=False, data=df )
df7 = df.drop(['Id','idhogar','areadscr','regiondscr','dependency'],axis=1)
df7.info()
df7['v18q1'] = list(
                map( lambda x,y: 
                    ( int(0) if y==0 else df7['v18q1'].mean() ) if x!=x 
                      else x,
                    df7['v18q1'],df7['v18q'] )
)
df7['edjefe'] = df7['edjefe'].apply(lambda x: int(0) if x=='no'
                                         else int(1) if x=='yes'
                                         else int(x))
df7['edjefa'] = df7['edjefa'].apply(lambda x: int(0) if x=='no'
                                         else int(1) if x=='yes'
                                         else int(x))
coldf=df7.columns.values.tolist()
x0 = len( df7.index )
for i in coldf:
    x = float(len( df7[pd.isnull(df7[i])].index )) 
    if x!=0 : print( i, x/x0 * 100 )
df8=df7.drop(['v2a1','rez_esc'],axis=1).interpolate()
#df8=df7.drop(['v2a1','rez_esc'],axis=1)
coldf=df8.columns.values.tolist()
x0 = len( df8.index )
for i in coldf:
    x = float(len( df8[pd.isnull(df8[i])].index )) 
    if x!=0 : print( i, x/x0 * 100 )
df8.info()
from sklearn import preprocessing
X_train = df8.drop('Target',axis=1)
y_train = df8['Target']
X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
XX_train, XX_test, yy_train, yy_test = train_test_split(X_scaled, y_train, test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier( )
dtree.fit(XX_train,yy_train)
y_dt = dtree.predict(XX_test)
print(confusion_matrix(yy_test,y_dt))
print('\n')
print(classification_report(yy_test,y_dt))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(XX_train, yy_train)
y_rfc = rfc.predict(XX_test)
print(confusion_matrix(yy_test,y_rfc))
print('\n')
print(classification_report(yy_test,y_rfc))
import tensorflow as tf
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(XX_train)
import tensorflow.contrib.learn as learn
classifier = learn.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[40, 20, 10],
      n_classes=5,
      optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))
classifier.fit(XX_train, yy_train, steps=2500)
y_tf = list(classifier.predict(XX_test))
# Evaluate accuracy.
accuracy_score = classifier.evaluate(XX_test, yy_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
print(confusion_matrix(yy_test,y_tf))
print('\n')
print(classification_report(yy_test,y_tf))
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( verbose=1 )# oob_score = True, )

param_grid = { 
    'n_estimators': [20,50,70,100],# 200, 500,1000],
    'criterion': ['gini', 'entropy'],
    'max_features': [ None, 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(XX_train, yy_train)
print (CV_rfc.best_params_)
y_grdrfc = CV_rfc.predict(XX_test)
print(confusion_matrix(yy_test,y_grdrfc))
print('\n')
print(classification_report(yy_test,y_grdrfc))
from sklearn.svm import SVC
param_grid = {
    'C': [0.1,1, 10, 100, 1000], 
    'gamma': [1,0.1,0.01,0.001,0.0001], 
    'kernel': ['rbf']
} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
# May take awhile!
grid.fit(XX_train,yy_train)
grid.best_params_
grid.best_estimator_
y_grdsvc = grid.predict(XX_test)
print(confusion_matrix(yy_test,y_grdsvc))
print('\n')
print(classification_report(yy_test,y_grdsvc))
dftest = pd.read_csv('../input/test.csv')
dftest.head()
df7test = dftest.drop(['Id','idhogar','dependency'],axis=1)
df7test.info()
len(X_train.columns)
len(df7test.columns)
for i in list(df7test.drop(['edjefe','edjefa','rez_esc'],axis=1).columns):
    x = np.corrcoef(df7test['SQBdependency'], df7test[i])[0,1]
    if abs(x) > 0.15 : 
        print( i, x )
df7test['v18q1'] = list(
    map( lambda x,y: ( 0 if y==0 else df7test['v18q1'].mean() ) if x!=x else x,
                    df7test['v18q1'],df7test['v18q'] )
)
df7test['edjefe'] = df7test['edjefe'].apply(lambda x: int(0) if x=='no'
                                         else int(1) if x=='yes'
                                         else int(x))
df7test['edjefa'] = df7test['edjefa'].apply(lambda x: int(0) if x=='no'
                                         else int(1) if x=='yes'
                                         else int(x))
coldf=df7test.columns.values.tolist()
x0 = len( df7test.index )
for i in coldf:
    x = float(len( df7test[pd.isnull(df7test[i])].index )) 
    if x!=0 : print( i, x/x0 * 100 )
df8test=df7test.drop(['v2a1','rez_esc'],axis=1).interpolate(method='akima')

coldf=df8test.columns.values.tolist()
x0 = len( df8test.index )
for i in coldf:
    x = float(len( df8test[pd.isnull(df8test[i])].index )) 
    if x!=0 : print( i, x/x0 * 100 )
df8test.head()
df8test.info()
df8test.count()
X_test = df8test
import sklearn.pipeline
select = sklearn.preprocessing.StandardScaler()
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
steps = [('feature_selection', select),
        ('random_forest', clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

### fit pipeline on X_train and y_train
pipeline.fit( X_train, y_train )

### call pipeline.predict() on X_test data to make a set of test predictions
y_prediction = pipeline.predict( X_test )
plt.figure(figsize=(10,5),facecolor='w') 
# Assign colors for each airline and the names
colors = ['g', 'b']
names = ['Test Prediction', 'Training Prediction']
         
# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([y_prediction,y_train], normed=True,color = colors, label=names, alpha=0.5)

# Plot formatting
plt.legend()
plt.xlabel('Target')
plt.ylabel('Normalized counts')
plt.title('Side-by-Side Histogram with test prediction and training prediction')
#               TEST HYPOTHESIS
#
testhyp = pd.DataFrame( { 'hhsize':list(X_test['hhsize']),
                       'SQBdependency':list(X_test['SQBdependency']),
                       'Target':list(y_prediction) } )
#data.head()
sns.lmplot(  x='hhsize',y='SQBdependency', hue='Target',
                  fit_reg=False, data=testhyp )
import sklearn.pipeline
scaler = sklearn.preprocessing.StandardScaler()

### Define RF classifier
rfc = RandomForestClassifier( oob_score = True, verbose=1)


parameters = dict(
                random_forest__n_estimators = [50,100,200,300,400,500,600],
                random_forest__criterion = ['gini', 'entropy'],
                random_forest__max_features = ['sqrt', 'log2']
                )
steps = [('scaler', scaler),
         ('random_forest', rfc)]

pipeline = sklearn.pipeline.Pipeline(steps)

CV_rfc = GridSearchCV(pipeline, param_grid=parameters)

### fit the pipeline on X_train and y_train
#CV_rfc.fit(X_train, y_train)
CV_rfc.fit(X_train.astype(np.float64), 
           y_train.astype(np.float64))

### Print best fitted parameters
print (CV_rfc.best_params_)

### .predict() on the X_test data to make a set of test predictions
y_grdrfc = CV_rfc.predict(X_test)
plt.figure(figsize=(10,5),facecolor='w') 
# Assign colors for each airline and the names
colors = ['g', 'b']
names = ['Test Prediction', 'Training Prediction']
         
# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([y_grdrfc,y_train], normed=True,color = colors, label=names, alpha=0.5)

# Plot formatting
plt.legend()
plt.xlabel('Target')
plt.ylabel('Normalized counts')
plt.title('Side-by-Side Histogram with test prediction and training prediction')
#
#               TEST HYPOTHESIS
#
testhyp = pd.DataFrame( { 'hhsize':list(X_test['hhsize']),
                       'SQBdependency':list(X_test['SQBdependency']),
                       'Target':list(y_grdrfc) } )
sns.lmplot( x='hhsize',y='SQBdependency', hue='Target',
            fit_reg=False, data=testhyp)
plt.figure(figsize=(10,5),facecolor='w') 
# Assign colors for each airline and the names
colors = ['g', 'b']
names = ['RF', 'RF Grid']
         
# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([y_prediction,y_grdrfc], normed=True,color = colors, label=names, alpha=0.5)

# Plot formatting
plt.legend()
plt.xlabel('Target')
plt.ylabel('Normalized counts')
plt.title('Side-by-Side Histogram with test predictions')
for x in (1,2,3,4):
    count  = list(y_grdrfc).count(x)
    print(x,count)
pd.DataFrame( { 'Id':list(dftest['Id']),
                'Target':list(y_grdrfc.astype(int)) } ).set_index('Id').to_csv('sample_submission.csv', sep=',')
#
# cross check
#
fpred=pd.read_csv('sample_submission.csv')
plt.hist([fpred['Target'],y_train], normed=True,color = colors, label=names, alpha=0.5)
testhyp = pd.DataFrame( { 'hhsize':list(X_test['hhsize']),
                       'SQBdependency':list(X_test['SQBdependency']),
                       'Target':list(fpred['Target']) } )
#data.head()
sns.lmplot(  x='hhsize',y='SQBdependency', hue='Target',
                  fit_reg=False, data=testhyp )