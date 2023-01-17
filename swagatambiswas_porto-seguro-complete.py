import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)



# Try ploty libraries

import plotly.tools as tls

import warnings



import seaborn as sns

plt.style.use('fivethirtyeight')



from collections import Counter

warnings.filterwarnings('ignore')



# import plotly.graph_objs as go

# import plotly.tools as tls

# import plotly.plotly as plpl
train = pd.read_csv("../input/porto-seguro-safe-driver-prediction/train.csv")

test = pd.read_csv("../input/porto-seguro-safe-driver-prediction/test.csv")

train.head(20)
test.head()
pd.set_option('precision',3)

train.describe()
# Check if there is any null information anywhere

train.isnull().any().any()
train_cp = train

train_cp = train_cp.replace(-1, np.NaN)



data = train
colwithnan = train_cp.columns[train_cp.isnull().any()].tolist()

print("This dataset has %s Rows. \n" % (train_cp.shape[0]))

for col in colwithnan:

    print("Column: %s has %s NaN" % (col, train_cp[col].isnull().sum()))
target_count = train.target.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)');
train_float = train.select_dtypes(include=['float64'])

train_int = train.select_dtypes(include=['int64'])

Counter(train.dtypes.values)
colormap = plt.cm.jet

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
colormap = plt.cm.jet

cotrain_float = train_float.drop(['ps_calc_03', 'ps_calc_02', 'ps_calc_01'], axis=1)

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(cotrain_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
colormap = plt.cm.jet

plt.figure(figsize=(21,16))

plt.title('Pearson correlation of categorical features', y=1.05, size=15)

sns.heatmap(train_int.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
colormap = plt.cm.jet

cotrain = train_int.drop(['id','target', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1)

plt.figure(figsize=(21,16))

plt.title('Pearson correlation of int features withot ps_calc', y=1.05, size=12)

sns.heatmap(cotrain.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
colormap = plt.cm.jet

# train = train.drop(['id', 'target'], axis=1)

plt.figure(figsize=(25,25))

plt.title('Pearson correlation of All the features', y=1.05, size=15)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
train_float.describe()
train_float.plot(kind='box', subplots=True, layout=(2,5), sharey=False, figsize=(18,18))

plt.show()

train_int.plot(kind='box', subplots=True, layout=(10,5), sharey=False, figsize=(18,90))

plt.show()
# Check the binary features

bin_col = [col for col in train.columns if '_bin' in col]

zeros = []

ones = []

for col in bin_col:

    zeros.append((train[col]==0).sum())

    ones.append((train[col]==1).sum())
trace1 = go.Bar(

    x=bin_col,

    y=zeros ,

    name='Zero count'

)

trace2 = go.Bar(

    x=bin_col,

    y=ones,

    name='One count'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of 1 and 0 in binary variables'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
train_int = train_int.drop(['id', 'target'], axis=1)

train_int = train_int.drop(bin_col, axis=1)

some_bin = train_int.describe()

some_bin
cat_asbin = []

for col in some_bin:

    #print(some_bin[col]['max'])

    if (some_bin[col]['max']==1):

        if ((some_bin[col]['min']==0) or (some_bin[col]['min']==-1)):

            cat_asbin.append(col)

cat_asbin
cat_zeros = []

cat_ones = []

for col in cat_asbin:

    cat_zeros.append((train[col]==0).sum())

    cat_ones.append((train[col]==1).sum())
trace1 = go.Bar(

    x=cat_asbin,

    y=cat_zeros ,

    name='Zero count'

)

trace2 = go.Bar(

    x=cat_asbin,

    y=cat_ones,

    name='One count'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of 1 and 0 in binary variables'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
colormap = plt.cm.jet

cotrainnb = cotrain.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin'], axis=1)

plt.figure(figsize=(21,16))

plt.title('Taking away some binary data', y=1.05, size=12)

sns.heatmap(cotrainnb.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)
cat_col = [col for col in train.columns if '_cat' in col]

catds = train[cat_col]

ncatds = catds.drop(cat_asbin, axis=1)

ncatds.describe()
from plotly import tools

hist_cat = []



for col in ncatds:

    hist_cat.append(go.Histogram(x=ncatds[col], opacity=0.75, name =col))



fig = tools.make_subplots(rows=len(hist_cat), cols=1)



for i in range(0,len(hist_cat),1):

    fig.append_trace(hist_cat[i], i+1, 1)

    

fig['layout'].update(height=1500, width=750, title='Cat Features Histogram')

py.iplot(fig, filename='cat-histogram')
no_cat = some_bin.drop(ncatds, axis=1)

no_cat.describe()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



# Test options and evaluation metric

num_folds = 10

seed = 8

scoring = 'accuracy'



X = train.drop(['id','target'], axis=1)

Y = train.target



validation_size = 0.3

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
models = [('LR', LogisticRegression()), 

          ('LDA', LinearDiscriminantAnalysis()),

          #('KNN', KNeighborsClassifier()),

          ('CART', DecisionTreeClassifier()),

          ('NB', GaussianNB())]

results = []

names = []

for name, model in models:

    print("Training model %s" %(name))

    model.fit(X_train, Y_train)

    result = model.score(X_validation, Y_validation)

    #kfold = KFold(n_splits=num_folds, random_state=seed)

    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    #results.append(cv_results)

    #names.append(name)

    msg = "Classifier score %s: %f" % (name, result)

    print(msg)

print("----- Training Done -----")
pipelines = [('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])),

             ('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])),

             # ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])),

             ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])),

             ('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())]))]

results = []

names = []

for name, model in pipelines:

    print("Training model %s" %(name))

    model.fit(X_train, Y_train)

    result = model.score(X_validation, Y_validation)

    #kfold = KFold(n_splits=num_folds, random_state=seed)

    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    #results.append(cv_results)

    #names.append(name)

    msg = "Classifier score %s: %f" % (name, result)

    print(msg)

print("----- Training Done -----")
ensembles = [('ABC', AdaBoostClassifier()), 

             ('GBM', GradientBoostingClassifier()),

             ('RFC', RandomForestClassifier()),

             ('ETC', ExtraTreesClassifier())]

results = []

names = []



for name, model in ensembles:

    print("Training model %s" %(name))

    model.fit(X_train, Y_train)

    result = model.score(X_validation, Y_validation)

    #kfold = KFold(n_splits=num_folds, random_state=seed)

    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    #results.append(cv_results)

    #names.append(name)

    msg = "Classifier score %s: %f" % (name, result)

    print(msg)

print("----- Training Done -----")
toplot = []

for name, model in ensembles:

    trace = go.Bar(x=model.feature_importances_,

                   y=X_validation.columns,

                   orientation='h',

                   textposition = 'auto',

                   name=name

                  )

    toplot.append(trace)



layout = dict(

        title = 'Barplot of features importance',

        width = 900, height = 2000,

        barmode='group')



fig = go.Figure(data=toplot, layout=layout)

py.iplot(fig, filename='features-figure')

    