# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')

test = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv')
train.head()
train.info()
batch_indices = [slice(500000*i,500000*(i+1)) for i in range(10)]
import matplotlib.pyplot as plt

import seaborn as sns
fig, axes = plt.subplots(2,5,sharex=False,sharey=True, figsize=(25,10))

for i,ax in enumerate(axes.ravel()):

    train.iloc[batch_indices[i]].plot(kind='line',x='time',y=['signal'],ax=ax,linewidth=.1)

    ax.set_title('Batch_'+str(i))

ax.set_ylim(-5,14)

ax.legend()

fig.suptitle('Training Data',y=1.05)



plt.tight_layout()
fig, axes = plt.subplots(1,4,sharex=False,sharey=True, figsize=(25,10))

for i,ax in enumerate(axes.ravel()):

    test.iloc[batch_indices[i]].plot(kind='line',x='time',y=['signal'],ax=ax,linewidth=.1)

    ax.set_title('Batch_'+str(i))

ax.set_ylim(-5,11)

ax.legend()

fig.suptitle('Testing Data',y=1.05)



plt.tight_layout()
fig, axes = plt.subplots(2,5,sharex=True,sharey=True, figsize=(20,8))

for i,ax in enumerate(axes.ravel()):

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 0')['signal'],ax=ax,label='0')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 1')['signal'],ax=ax,label='1')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 2')['signal'],ax=ax,label='2')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 3')['signal'],ax=ax,label='3')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 4')['signal'],ax=ax,label='4')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 5')['signal'],ax=ax,label='5')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 6')['signal'],ax=ax,label='6')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 7')['signal'],ax=ax,label='7')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 8')['signal'],ax=ax,label='8')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 9')['signal'],ax=ax,label='9')

        sns.distplot(train.iloc[batch_indices[i]].query('open_channels == 10')['signal'],ax=ax,label='10')

        ax.set_title('Batch_'+str(i))

ax.legend()

fig.suptitle('Training Data',y=1.05)



ax.set_ylim(0,2)

plt.tight_layout()
train_seg_boundaries = np.concatenate([[0,500000,600000], np.arange(1000000,5000000+1,500000)])

train_signal = np.split(np.zeros(5000000), train_seg_boundaries[1:-1])

test_seg_boundaries = np.concatenate([np.arange(0,1000000+1,100000), [1500000,2000000]])

test_signal = np.split(np.zeros(2000000), test_seg_boundaries[1:-1])
test['signal_type']=np.concatenate([test_signal[0]+1,test_signal[1]+3,

                             test_signal[2]+4,test_signal[3]+1,

                             test_signal[4]+2,test_signal[5]+5,

                             test_signal[6]+4,test_signal[7]+5,

                             test_signal[8]+1,test_signal[9]+3,

                             test_signal[10]+1,test_signal[11]+1])

test['signal'].plot(kind='line',linewidth=.2, label='Test Signal')

test['signal_type'].plot(kind='line', label='Signal Type')

plt.legend()

del test_signal
train['signal_type']= np.concatenate([train_signal[0]+1,train_signal[1]+1,

                               train_signal[2]+1,train_signal[3]+2,

                               train_signal[4]+3,train_signal[5]+5,

                               train_signal[6]+4,train_signal[7]+2,

                               train_signal[8]+3,train_signal[9]+4,

                               train_signal[10]+5])

train['signal'].plot(kind='line',linewidth=.2, label='Train Signal')

train['signal_type'].plot(kind='line', label='Signal Type')

plt.legend()

del train_signal
del train['time'] # we don't need the time anymore
means = train.groupby(['signal_type','open_channels']).mean().signal

train['scaled_signal'] = train['signal']

test['scaled_signal'] = test['signal']
from sklearn.metrics import f1_score, classification_report
def shift_model(x,sig_type):

    scaled = (train.loc[train.signal_type == sig_type,'signal']-x[0])*x[1]

    target = train.loc[train.signal_type == sig_type,'open_channels']

    return -f1_score(target,scaled.clip(0,10).round(),average='weighted')
from scipy.optimize import minimize
for i in range(1,5):

    print(i)

    min_f=minimize(shift_model,[means.loc[i,0],1/(means.loc[i,1]-means.loc[i,0])],args=(i),method='Powell')

    train.loc[train.signal_type == i,'scaled_signal'] = (train.loc[train.signal_type == i,'signal']-min_f['x'][0])*min_f['x'][1]

    test.loc[test.signal_type == i,'scaled_signal'] = (test.loc[test.signal_type == i,'signal']-min_f['x'][0])*min_f['x'][1]



i=5

min_f=minimize(shift_model,[means.loc[i,1]-1,5/(means.loc[i,6]-means.loc[i,1])],args=(i),method='Powell')

train.loc[train.signal_type == i,'scaled_signal'] = (train.loc[train.signal_type == i,'signal']-min_f['x'][0])*min_f['x'][1]

test.loc[test.signal_type == i,'scaled_signal'] = (test.loc[test.signal_type == i,'signal']-min_f['x'][0])*min_f['x'][1]

del means
fig, axes = plt.subplots(1,5,sharex=True,sharey=True, figsize=(25,5))

for i,ax in enumerate(axes.ravel()):

    for j in range(11):

        sns.distplot(train[(train.signal_type==i+1)&(train.open_channels == j)]['scaled_signal'],ax=ax,label='%i'%j)

        

axes[3].legend()

ax.set_ylim(0,2.2)

ax.set_xticks(np.arange(11))

plt.tight_layout()
print(f1_score(train.open_channels,train['scaled_signal'].clip(0,10).round(),average='macro'))
print(classification_report(train.open_channels,train['scaled_signal'].clip(0,10).round()))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(train.open_channels,train['scaled_signal'].clip(0,10).round(),normalize='true')
fig, axes = plt.subplots(1,1,sharex=True,sharey=True, figsize=(25,10))

axes=sns.heatmap(cm,annot=True)
plt.plot(np.histogram(train['scaled_signal'].clip(0,10).round(),bins=11)[0]/np.histogram(train.open_channels,bins=11)[0]-1)

plt.title('% Error for each label')
fig, axes = plt.subplots(1,1,sharex=True,sharey=True, figsize=(25,10))

sns.distplot(train['scaled_signal'].clip(0,10).round(),ax=axes, kde=False, label='prediction')

sns.distplot(train.open_channels,ax=axes, kde=False, label='truth')

plt.legend()
train.drop(['signal'],axis=1,inplace=True)

test.drop(['signal'],axis=1,inplace=True)
def calc_gradients(s, n_grads=2):

    grads = pd.DataFrame()

    g = s.values

    for i in range(n_grads):

        g = np.gradient(g)

        grads['grad_' + str(i+1)] = g

    return grads



from scipy import signal

def calc_low_pass(s, n_filts=10):

    

    wns = np.logspace(-2, -0.3, n_filts)

    

    low_pass = pd.DataFrame()

    x = s.values

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='low')

        zi = signal.lfilter_zi(b, a)

        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]

        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)

        

    return low_pass



def calc_high_pass(s, n_filts=10):

    

    wns = np.logspace(-2, -0.1, n_filts)

    

    high_pass = pd.DataFrame()

    x = s.values

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='high')

        zi = signal.lfilter_zi(b, a)

        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]

        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)

        

    return high_pass



def calc_roll_stats(s, windows=[10,50, 100, 500,1000]):

    '''

    Calculates rolling stats like mean, std, min, max...

    '''

    roll_stats = pd.DataFrame()

    for w in windows:

        roll_stats['roll_mean_' + str(w)] = s.rolling(window=w).mean()

        roll_stats['roll_median_' + str(w)] = s.rolling(window=w).median()

        roll_stats['roll_std_' + str(w)] = s.rolling(window=w).std()

        roll_stats['roll_min_' + str(w)] = s.rolling(window=w).min()

        roll_stats['roll_max_' + str(w)] = s.rolling(window=w).max()

        roll_stats['roll_range_' + str(w)] = roll_stats['roll_max_' + str(w)] - roll_stats['roll_min_' + str(w)]

        roll_stats['roll_q10_' + str(w)] = s.rolling(window=w).quantile(0.10)

        roll_stats['roll_q25_' + str(w)] = s.rolling(window=w).quantile(0.25)

        roll_stats['roll_q75_' + str(w)] = s.rolling(window=w).quantile(0.75)

        roll_stats['roll_q90_' + str(w)] = s.rolling(window=w).quantile(0.90)

        

    

    # add zeros when na values (std)

    roll_stats = roll_stats.fillna(value=0)

             

    return roll_stats



def calc_ewm(s, windows=[10,50, 100,500, 1000]):

    

    ewm = pd.DataFrame()

    for w in windows:

        ewm['ewm_mean_' + str(w)] = s.ewm(span=w, min_periods=1).mean()

        ewm['ewm_std_' + str(w)] = s.ewm(span=w, min_periods=1).std()

        

    # add zeros when na values (std)

    ewm = ewm.fillna(value=0)

        

    return ewm



def calc_shifts(s, periods=[-3,-2,-1,1,2,3]):

    

    sft = pd.DataFrame()

    for p in periods:

        if p>0:

            sft['signal_shift_' + str(p)] = s.shift(periods=p).fillna(method='bfill')

        else:

            sft['signal_shift_' + str(p)] = s.shift(periods=p).fillna(method='ffill')

    return sft



def add_features(s):

    

    gradients = calc_gradients(s)

    low_pass = calc_low_pass(s)

    high_pass = calc_high_pass(s)

    roll_stats = calc_roll_stats(s)

    ewm = calc_ewm(s)

    sft = calc_shifts(s)

    

    return pd.concat([gradients, low_pass, high_pass, roll_stats, ewm, sft], axis=1)



def divide_and_add_features(s, splits):

    

    ls = []

    for i,v in enumerate(splits[:-1]):

        print(i)

        sig = s[v:splits[i+1]].copy().reset_index(drop=True)

        sig_featured = add_features(sig)

        ls.append(sig_featured)

    

    return pd.concat(ls, axis=0).set_index(s.index)
df = divide_and_add_features(train['scaled_signal'],train_seg_boundaries)

train = pd.concat([train,df],axis=1)

del df
train['signal_quad'] = train.scaled_signal**2

train['signal_cubic'] = train.scaled_signal**3
train.drop(range(3640000,3830000),axis=0,inplace=True)

drop_indexes=train[(train.signal_type==5) & (train.open_channels==0)].index

train.drop(drop_indexes,inplace=True)
train_type=[]

from sklearn.model_selection import train_test_split

for t,df_type in train.groupby('signal_type'):

    train_type.append(dict(zip(['X_train','X_test','y_train','y_test'],

                               train_test_split(df_type.drop(['signal_type','open_channels'],axis=1),

                                                df_type['open_channels'],

                                                random_state=42,

                                                stratify=df_type['open_channels'],

                                                test_size=.2))))

    

del train

    
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV



# def train_models():

#     models = []

#     predictions = []

#     tests = []

#     t = train_type[-1]

#     pipeline=Pipeline([('scaler',PCA(n_components=95,whiten=True)),

#         ('clf',Ridge())])



#     param_grid = {'scaler__n_components':range(10,len(t['X_test'].columns),10),'clf__alpha':[2,5]}



#     grid_model = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', refit=True)



#     grid_model.fit(t['X_train'],t['y_train'])

#     max_y = np.max(t['y_test'])

#     min_y = np.min(t['y_test'])

#     y_pred = np.round(np.clip(grid_model.predict(t['X_test']),min_y,max_y))

#     print('mse:',grid_model.score(t['X_test'], t['y_test']))

#     print('CV mse:',grid_model.best_score_,grid_model.best_params_)

#     print(f1_score(t['y_test'],y_pred, average='weighted'))

#     print(classification_report(t['y_test'],y_pred))

#     models.append(grid_model)

#     predictions.append(y_pred)

#     tests.append(t['y_test'])

#     print(f1_score(np.concatenate(tests),np.concatenate(predictions), average='macro'))

#     print(classification_report(np.concatenate(tests),np.concatenate(predictions)))

#     return models

# models=train_models()
# from lightgbm import LGBMRegressor

# from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# from sklearn.linear_model import Ridge

# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler

# from sklearn.decomposition import PCA

# from sklearn.neural_network import MLPRegressor

# from sklearn.model_selection import GridSearchCV



# def train_models():

#     models = []

#     predictions = []

#     tests = []

#     t = train_type[-1]

#     lgb=LGBMRegressor(num_leaves=30,

#             learning_rate=0.1,

#             n_estimators=900,

#             subsample=.8,

#             colsample_bytree=1,

#             random_state=42,

#             n_jobs=-1

#             )



#     param_grid = {

#     'learning_rate': [ 0.01,.05],

#     'num_leaves': [30,50], 

#     'boosting_type' : [ 'dart'],

#     'subsample' : [0.6,0.8],

#     'min_child_samples': [10,20]

#     }



#     grid_model = GridSearchCV(lgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', refit=True, n_jobs=-1)



#     grid_model.fit(t['X_train'],t['y_train'])

#     max_y = np.max(t['y_test'])

#     min_y = np.min(t['y_test'])

#     y_pred = np.round(np.clip(grid_model.predict(t['X_test']),min_y,max_y))

#     print('mse:',grid_model.score(t['X_test'], t['y_test']))

#     print('CV mse:',grid_model.best_score_,grid_model.best_params_)

#     print(f1_score(t['y_test'],y_pred, average='weighted'))

#     print(classification_report(t['y_test'],y_pred))

#     models.append(grid_model)

#     predictions.append(y_pred)

#     tests.append(t['y_test'])

#     print(f1_score(np.concatenate(tests),np.concatenate(predictions), average='macro'))

#     print(classification_report(np.concatenate(tests),np.concatenate(predictions)))

#     return models

# models=train_models()
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV

# def train_models():

#     models = []

#     predictions = []

#     tests = []

#     t = train_type[-1]

#     pipeline=Pipeline([('scaler',PCA(n_components=20,whiten=True)),

#         ('clf',MLPRegressor(hidden_layer_sizes=(50,20,15),early_stopping=True))])



#     param_grid = {'scaler__n_components':range(21,39)}



#     grid_model = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', refit=True)



#     grid_model.fit(t['X_train'],t['y_train'])

#     max_y = np.max(t['y_test'])

#     min_y = np.min(t['y_test'])

#     y_pred = np.round(np.clip(grid_model.predict(t['X_test']),min_y,max_y))

#     print('mse:',grid_model.score(t['X_test'], t['y_test']))

#     print('CV mse:',grid_model.best_score_,grid_model.best_params_)

#     print(f1_score(t['y_test'],y_pred, average='weighted'))

#     print(classification_report(t['y_test'],y_pred))

#     models.append(grid_model)

#     predictions.append(y_pred)

#     tests.append(t['y_test'])

#     print(f1_score(np.concatenate(tests),np.concatenate(predictions), average='macro'))

#     print(classification_report(np.concatenate(tests),np.concatenate(predictions)))

#     return models

# models=train_models()
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.linear_model import Ridge

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPRegressor

def train_models():

    models = []

    predictions = []

    tests = []

    for t in train_type:

         

        clf1 = Pipeline([('scaler',PCA(n_components=91,whiten=True)),

            ('clf',Ridge(alpha=5))])

        clf2 = Pipeline([

            ('clf',LGBMRegressor(num_leaves=30,

            learning_rate=0.05,

            n_estimators=900,

            subsample=.8,

            colsample_bytree=1,

            random_state=42,

            min_child_samples=10,

            n_jobs=-1

            ))])

        

        clf3 = Pipeline([('scaler',PCA(n_components=25,whiten=True)),

            ('clf',MLPRegressor(hidden_layer_sizes=(50,20,15),early_stopping=True))])

        eclf1 = VotingRegressor(estimators=[('lr', clf1), ('bm', clf2),('nn',clf3)])

        steps = [('clf',eclf1)]

        pipeline = Pipeline(steps)

        

        pipeline.fit(t['X_train'],t['y_train'])

        max_y = np.max(t['y_test'])

        min_y = np.min(t['y_test'])

        y_pred = np.round(np.clip(pipeline.predict(t['X_test']),min_y,max_y)).astype(int)

        

        print(f1_score(t['y_test'],y_pred, average='weighted'))

        print(classification_report(t['y_test'],y_pred))

        models.append(pipeline)

        predictions.append(y_pred)

        tests.append(t['y_test'])

    print(f1_score(np.concatenate(tests),np.concatenate(predictions), average='macro'))

    print(classification_report(np.concatenate(tests),np.concatenate(predictions)))

    return models

models=train_models()
y_pred = []

y_test = []

for i,m in enumerate(models):

    y_pred.append(m.predict(train_type[i]['X_test']))

    y_test.append(train_type[i]['y_test'])

y_pred=np.concatenate(y_pred)

y_test=np.concatenate(y_test)
comps=pd.DataFrame({'pred':y_pred,'test':y_test})
def threshold_model(x,df):

    s = df.pred.copy()

    s[s<x[0]] = 0

    s[(s>=x[0])&(s<x[1])] = 1

    s[(s>=x[1])&(s<x[2])] = 2

    s[(s>=x[2])&(s<x[3])] = 3

    s[(s>=x[3])&(s<x[4])] = 4

    s[(s>=x[4])&(s<x[5])] = 5

    s[(s>=x[5])&(s<x[6])] = 6

    s[(s>=x[6])&(s<x[7])] = 7

    s[(s>=x[7])&(s<x[8])] = 8

    s[(s>=x[8])&(s<x[9])] = 9

    s[s>=x[9]] = 10

    return -f1_score(df.test,s.values,average='macro')
thresholds=minimize(threshold_model,np.arange(.5,10.5,1),args=(comps),method='Powell',tol=.0001)
def threshold_func(x):

    if x<thresholds['x'][0]:

        return 0

    elif (x>thresholds['x'][0]) and (x<thresholds['x'][1]):

        return 1

    elif (x>thresholds['x'][1]) and (x<thresholds['x'][2]):

        return 2

    elif (x>thresholds['x'][2]) and (x<thresholds['x'][3]):

        return 3

    elif (x>thresholds['x'][3]) and (x<thresholds['x'][4]):

        return 4

    elif (x>thresholds['x'][4]) and (x<thresholds['x'][5]):

        return 5

    elif (x>thresholds['x'][5]) and (x<thresholds['x'][6]):

        return 6

    elif (x>thresholds['x'][6]) and (x<thresholds['x'][7]):

        return 7

    elif (x>thresholds['x'][7]) and (x<thresholds['x'][8]):

        return 8

    elif (x>thresholds['x'][8]) and (x<thresholds['x'][9]):

        return 9

    else:

        return 10
comps['pred']=comps.pred.apply(threshold_func)
f1_score(comps.test,comps.pred,average='macro')
df = divide_and_add_features(test['scaled_signal'],test_seg_boundaries)

test=pd.concat([test,df],axis=1)

del df

test['signal_quad'] = test.scaled_signal**2

test['signal_cubic'] = test.scaled_signal**3
test['open_channels']=test.scaled_signal*0

for i in range(1,6):

     data=test.loc[(test.signal_type == i)].drop(['open_channels','time','signal_type'],axis=1)

    

     test.loc[test.signal_type == i,'open_channels'] = models[i-1].predict(data)

test['open_channels']=test['open_channels'].apply(threshold_func)

test[['time','open_channels']].to_csv('ion_submission_en.csv', index=False, float_format='%.4f')