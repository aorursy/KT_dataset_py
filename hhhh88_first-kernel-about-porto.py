import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import seaborn as sns

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.shape
train.drop_duplicates().shape # default : inplace=False. 
test.shape
train.info()
from collections import Counter

Counter(train.dtypes.values)
data = []

for f in train.columns:

    # Defining the role

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

        

    # Defining the level

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f =='id':

        level = 'nominal'

    elif train[f].dtype == float:

        level = 'interval'

    elif train[f].dtype == int:

        level = 'ordinal'

        

    # Initialize keep to True for all variables except for id

    keep = True

    if f == 'id':

        keep = False

        

    # Defining the data type

    dtype = train[f].dtype

    

    # Creating a Dcit that contains all the metadata for the variable

    f_dict = {

        'varname':f,

        'role':role,

        'level':level,

        'keep':keep,

        'dtype':dtype

    }

    data.append(f_dict)

    

meta = pd.DataFrame(data, columns=['varname','role','level','keep','dtype'])

meta.set_index('varname',inplace=True)    
meta
meta[(meta.level=='nominal')&(meta.keep)].index
pd.DataFrame({'count':meta.groupby(['role','level'])['role'].size()}).reset_index()
v = meta[(meta.level == 'interval')&(meta.keep)].index

train[v].describe()
train['target'].value_counts().plot.pie(autopct='%1.1f%%')

plt.title('Distribution of target variable')

plt.show()
# 멋있게 하려면... (쓸데없어보임ㅎ)

data = [go.Bar(x=train['target'].value_counts().index.values,y=train['target'].value_counts().values, text='Distribution of target variable')]

layout = go.Layout(title='Target variable distribution')

fig = go.Figure(data=data,layout=layout)

py.iplot(fig,filename='basic-bar')
desired_apriori = 0.10



# Get the indices per target value

idx_0 = train[train.target==0].index

idx_1 = train[train.target==1].index



# Get original number of records per target value

nb_0 = len(train.loc[idx_0])

nb_1 = len(train.loc[idx_1])



# Calculate the undersampling rate and resulting number of records with target=0

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

undersampled_nb_0 = int(undersampling_rate*nb_0)

print('Rate to undersample records with target=0 : {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))



# Randomly select records with target=0 to get at the desired a priori

from sklearn.utils import shuffle



undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)



# Construct list with remaining indices

idx_list = list(undersampled_idx) + list(idx_1)



# Return undersample data frame

train = train.loc[idx_list].reset_index(drop=True)
train['target'].value_counts().plot.pie(autopct='%1.1f%%')

plt.title('Distribution of target variable')

plt.show()
train.isnull().any().any()
train_check_null = train

train_check_null = train_check_null.replace(-1,np.NaN)



import missingno as msno

msno.matrix(train_check_null.iloc[:,4:35],figsize=(16,9),color=(0.3,0.1,0.2))


missing_col = []

for c in train_check_null.columns:

    if train_check_null[c].isnull().sum() > 0:

        missing_col.append(c)

        print('col : {:<15}, Nan records : {:>6}, Nan ratio : {:.3f}'.format(c, train_check_null[c].isnull().sum(), 100*(train_check_null[c].isnull().sum()/train_check_null[c].shape[0])))
meta.loc[missing_col,'level']
# dropping the variables

train.drop(['ps_car_03_cat','ps_car_05_cat'],inplace=True, axis=1)

# updating the meta

meta.loc[['ps_car_03_cat','ps_car_05_cat'],'keep']=False
f,ax = plt.subplots(1,2,figsize=(12,6))



sns.distplot(train['ps_reg_03'],ax=ax[0])

sns.distplot(train['ps_car_14'],ax=ax[1])
# Imputing with the mean or mode

mean_imp = Imputer(missing_values=-1,strategy='mean',axis=0)

mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
tmp = ['ps_car_01_cat','ps_car_02_cat',

       'ps_car_07_cat','ps_car_09_cat']



train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()

train['ps_ind_02_cat'] = mode_imp.fit_transform(train[['ps_ind_02_cat']]).ravel()

train['ps_ind_04_cat'] = mode_imp.fit_transform(train[['ps_ind_04_cat']]).ravel()

train['ps_ind_05_cat'] = mode_imp.fit_transform(train[['ps_ind_05_cat']]).ravel()



for c in tmp:

    train[c] = mode_imp.fit_transform(train[[c]]).ravel()







# Serires.ravel(order='C') 

# Return the flattened underlying data as an ndarray. 

# so, the ps_reg_03 column case : (216940, 1) --> (216940,)
v = meta[(meta.level=='nominal')&(meta.keep)].index



for f in v:

    dist_values = train[f].value_counts().shape[0] # == nuique() 

    print('col:{:<10}   distinct values count:{}'.format(f,dist_values))
# Script by https://www.kaggle.com/ogrellier

# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=1, 

                  smoothing=1,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean #agg = aggregate(func_or_funcs)

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 

                             test["ps_car_11_cat"], 

                             target=train.target, 

                             min_samples_leaf=100,

                             smoothing=10,

                             noise_level=0.01)
train_encoded.head()
train['ps_car_11_cat_te'] = train_encoded

train.drop('ps_car_11_cat', axis=1, inplace=True)

meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta

test['ps_car_11_cat_te'] = test_encoded

test.drop('ps_car_11_cat', axis=1, inplace=True)
# if you use one-hot encoding

# xx = train['ps_ind_02_cat'].values.reshape(-1,1)

# xx = ohe.fit_transform(xx).toarray()

# tt = pd.concat([train.drop('ps_ind_02_cat',axis=1,inplace=True), pd.DataFrame(xx)],axis=1)



# I'll create dummy variables

for col in v:

    train = pd.concat([train.drop(col,axis=1),pd.get_dummies(train[col], prefix='dum_'+col)],axis=1)
train[ 'dum_ps_car_11_cat_104']
X_train = train.drop(['id','target'],axis=1)

y_train = train.target



rf = RandomForestClassifier(n_jobs=-1)

rf.fit(X_train, y_train)
rf.score(X_train, y_train)
# create metadata table



data = []

for f in test.columns:

    # Defining the role

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

        

    # Defining the level

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f =='id':

        level = 'nominal'

    elif test[f].dtype == float:

        level = 'interval'

    elif test[f].dtype == int:

        level = 'ordinal'

        

    # Initialize keep to True for all variables except for id

    keep = True

    if f == 'id':

        keep = False

        

    # Defining the data type

    dtype = test[f].dtype

    

    # Creating a Dcit that contains all the meta_test_testdata for the variable

    f_dict = {

        'varname':f,

        'role':role,

        'level':level,

        'keep':keep,

        'dtype':dtype

    }

    data.append(f_dict)

    

meta_test = pd.DataFrame(data, columns=['varname','role','level','keep','dtype'])

meta_test.set_index('varname',inplace=True)    
# Imputing or drop on missing columns

test_check_null = test

test_check_null = test_check_null.replace(-1,np.NaN)



missing_col_test = []

for c in test_check_null.columns:

    if test_check_null[c].isnull().sum() > 0:

        missing_col_test.append(c)

        print('col : {:<15}, Nan records : {:>6}, Nan ratio : {:.3f}'.format(c, test_check_null[c].isnull().sum(), 100*(test_check_null[c].isnull().sum()/test_check_null[c].shape[0])))
missing_col == missing_col_test
# dropping the variables

test.drop(['ps_car_03_cat','ps_car_05_cat'],inplace=True, axis=1)

# updating the meta

meta_test.loc[['ps_car_03_cat','ps_car_05_cat'],'keep']=False



# Imputing values - mean

test['ps_reg_03'] = mean_imp.fit_transform(test[['ps_reg_03']]).ravel()

test['ps_car_14'] = mean_imp.fit_transform(test[['ps_car_14']]).ravel()



# Imputing values - mode

tmp = ['ps_car_01_cat','ps_car_02_cat','ps_car_11','ps_ind_02_cat',

       'ps_car_07_cat','ps_car_09_cat','ps_ind_04_cat','ps_ind_05_cat']



for c in tmp:

    test[c] = mode_imp.fit_transform(test[[c]]).ravel()
# create dummy variables

v = meta_test[(meta_test.level=='nominal')&(meta_test.keep)].index



for f in v:

    dist_values = test[f].value_counts().shape[0] # == nuique() 

    print('col:{:<10}   distinct values count:{}'.format(f,dist_values))
# I'll create dummy variables

for col in v:

    test = pd.concat([test.drop(col,axis=1),pd.get_dummies(test[col], prefix='dum_'+col)],axis=1)
test.head(2)
X_test = test.drop('id',axis=1)



predicted = rf.predict_proba(X_test)
def gini(actual, pred, cmpcol = 0, sortcol = 1):  

    assert( len(actual) == len(pred) )  

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)  

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]  

    totalLosses = all[:,0].sum()  

    giniSum = all[:,0].cumsum().sum() / totalLosses  

    giniSum -= (len(actual) + 1) / 2.  

    return giniSum / len(actual)  



    def gini_normalized(a, p):  

        return gini(a, p) / gini(a, a)  

    

    def test_gini():

        def fequ(a,b):  

            return abs( a -b) < 1e-6  

        def T(a, p, g, n):  

            assert( fequ(gini(a,p), g) )  

            assert( fequ(gini_normalized(a,p), n) )  

        T([1, 2, 3], [10, 20, 30], 0.111111, 1)  

        T([1, 2, 3], [30, 20, 10], -0.111111, -1)  

        T([1, 2, 3], [0, 0, 0], -0.111111, -1)  

        T([3, 2, 1], [0, 0, 0], 0.111111, 1)  

        T([1, 2, 4, 3], [0, 0, 0, 0], -0.1, -0.8)  

        T([2, 1, 4, 3], [0, 0, 2, 1], 0.125, 1)  

        T([0, 20, 40, 0, 10], [40, 40, 10, 5, 5], 0, 0)  

        T([40, 0, 20, 0, 10], [1000000, 40, 40, 5, 5], 0.171428,0.6)  

        T([40, 20, 10, 0, 0], [40, 20, 10, 0, 0], 0.285714, 1)  

        T([1, 1, 0, 1], [0.86, 0.26, 0.52, 0.32], -0.041666, -0.333333)
# gini(y_train,predicted)

# print(len(y_train),len(predicted))
submission = pd.DataFrame({'id':test['id'], 'target':pd.DataFrame(predicted)[1]})

submission.head(2)
submission.to_csv('../submission.csv',index=False)
pd.read_csv('../submission.csv').head()
import os

os.listdir('../')
# from sklearn.metrics import accuracy_score



# predicted = rf.predict(X_test)

# accuracy = accuracy_score(y_test, predicted)



# print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')

# print(f'Mean accuracy score: {accuracy:.3}')
v = meta[(meta.level=='nominal')&(meta.keep)].index



for f in v:

    plt.figure()

    fig,ax = plt.subplots(figsize=(20,10))

    # Calculate the percentage of taget=1

    cat_perc = train[[f,'target']].groupby([f],as_index=False).mean()

    cat_perc.sort_values(by='target',ascending=False,inplace=True)

    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])

    plt.ylabel('% target', fontsize=18)

    plt.xlabel(f, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show();
def corr_heatmap(v):

    correlations = train[v].corr()

    

    # Create color map ranging between two colors

    cmap = sns.diverging_palette(220,10,as_cmap=True)

    

    fig,ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations,cmap=cmap,vmax=1.0,center=0,fmt='.2f',square=True,

               linewidths=.5, annot=True, cbar_kws={'shrink':.75})

    plt.show();

    

v = meta[(meta.level=='interval')&(meta.keep)].index

corr_heatmap(v)
s = train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02',y='ps_reg_03',data=s,hue='target',palette='Set1',scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
v = meta[(meta.level == 'ordinal') & (meta.keep)].index

corr_heatmap(v)
v = meta[(meta.level == 'nominal') & (meta.keep)].index

print('Before dummification we have {} variables in train'.format(train.shape[1]))

train = pd.get_dummies(train, columns=v, drop_first=True)

print('After dummification we have {} variables in train'.format(train.shape[1]))
train_float = train.select_dtypes(include=['float64'])

train_int = train.select_dtypes(include=['int64'])
train_int.columns
colormap = plt.cm.YlGnBu

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train_float.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
data = [

    go.Heatmap(

        z= train_int.corr().values,

        x=train_int.columns.values,

        y=train_int.columns.values,

        colorscale='Viridis',

        reversescale = False,

        text = True ,

        opacity = 1.0 )

]



layout = go.Layout(

    title='Pearson Correlation of Integer-type features',

    xaxis = dict(ticks='', nticks=36),

    yaxis = dict(ticks='' ),

    width = 900, height = 700)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')


# mutual_info_classif(X,y) X:Feature matrix, y:Target vector

# Estimate mutual information for a discrete target variable.

# 0 : 두 변수가 독립적, higher value : 높은 종속성

# 리턴 : 각 feature와 target변수 간 상호정보

# relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances

mf = mutual_info_classif(train_float.values, train.target.values, n_neighbors=3, random_state=1)

print(mf)

# low dependency !
mutual_info_classif(train_int.values, train.target.values, n_neighbors=3, random_state=1)

bin_col = [col for col in train.columns if '_bin' in col]

zero_list = []

one_list = []

for col in bin_col:

    zero_list.append((train[col]==0).sum())

    one_list.append((train[col]==1).sum())
trace1 = go.Bar(

    x=bin_col,

    y=zero_list ,

    name='Zero count'

)

trace2 = go.Bar(

    x=bin_col,

    y=one_list,

    name='One count'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of 1 and 0 in binary variables'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2,

                           n_jobs=-1, random_state=0)

# n_estimator : 트리 갯수

# min_samples_leaf : 각 노드가 가지는 최소 샘플 수

# max_features : 값이 0.2 float이기때문에 int(max_features * n_features). split할때 고려하는 feature 수

# n_jobs : number or jobs. -1이면 전부. default 1

rf.fit(train.drop(['id','target'],axis=1), train.target)

features = train.drop(['id','target'],axis=1).columns.values

print('----Training Done----')
# scatter plot

trace = go.Scatter(

    y = rf.feature_importances_,

    x = features,

    mode = 'markers',

    marker = dict(

        sizemode='diameter',

        sizeref =1,

        size=13,

        color=rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = features

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')

x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), 

                                                            reverse = False)))

trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Random Forest Feature importance',

    orientation='h',

)



layout = dict(

    title='Barplot of Feature importances',

     width = 900, height = 2000,

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

#         domain=[0, 0.85],

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')
from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image,ImageDraw, ImageFont

import re



decision_tree = tree.DecisionTreeClassifier(max_depth=3)

decision_tree.fit(train.drop(['id','target'],axis=1), train.target)



# Export ourt trained model as a .dot file

with open('tree1.dot','w') as f:

    f = tree.export_graphviz(decision_tree,out_file=f,max_depth=4,impurity=False,

                             feature_names=train.drop(['id','target'],axis=1).columns.values,

                            class_names=['No','Yes'], rounded=True, filled=True)

    

# Conver .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open('tree1.png')

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage('sample-out.png')


from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, 

                               min_samples_leaf=4, max_features=0.2, random_state=0)

gb.fit(train.drop(['id', 'target'],axis=1), train.target)

features = train.drop(['id', 'target'],axis=1).columns.values

print("----- Training Done -----")



# n_estimator : The number of boosting stages to perform.

# Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# Scatter plot 

trace = go.Scatter(

    y = gb.feature_importances_,

    x = features,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = features

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Machine Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
x, y = (list(x) for x in zip(*sorted(zip(gb.feature_importances_, features), 

                                                            reverse = False)))

trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Gradient Boosting Classifer Feature importance',

    orientation='h',

)



layout = dict(

    title='Barplot of Feature importances',

     width = 900, height = 2000,

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')