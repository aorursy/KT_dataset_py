import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import os
from time import time
import pickle
import warnings
warnings.filterwarnings('ignore')
sns.set(style="darkgrid")
train = pd.read_csv('../input/dt_funda_train_2018.csv')
test = pd.read_csv('../input/dt_funda_test_2018.csv')
print('train set contains {} rows and {} columns'.format(train.shape[0],train.shape[1]))
print('test set contains {} rows and {} columns'.format(test.shape[0],test.shape[1]))
train.head()
train = train.drop(['flg_missing_n_photos','flg_missing_n_photos_360'],axis=1)
test = test.drop(['flg_missing_n_photos','flg_missing_n_photos_360'],axis=1)
print(train.columns)
ID_train = train['id']
train = train.drop(['id'],axis=1)
ID_test = test['id']
test = test.drop(['id'],axis=1)
#Transform categorical variables type to 'object'
def to_categorical(df,feats):
    
    for col in feats:
        if df[col].dtype!='object':
            df[col] = df[col].astype('object')
            
    return df

#the categorical features
cat_var = ['zipcode','type_of_construction','energy_label','located_on','own_ground']            
train = to_categorical(train,cat_var)
#number of flags columns
n_flag = 12

#flag features
flag_var = train.iloc[:,-n_flag:].columns

#numerical features
num_var = train.iloc[:,0:-n_flag].drop('log_price',axis=1).select_dtypes(exclude='object').columns

#Just in case I need it later
all_var = train.iloc[:,0:-n_flag].drop('log_price',axis=1).columns
#Some categories were mispelled
correction_dict = { 'A ':'A',
                    'A':'A',
                    'A+':'A',
                    'B ':'B',
                    'B':'B',
                    'C ':'C',
                    'C':'C',
                    'D ':'D',
                    'D':'D',
                    'E':'E',
                    'E ':'E',
                    'F ':'F',
                    'F':'F',
                    'G ':'G',
                    'G':'G',
                    'n':'N'}

train['energy_label'] = train['energy_label'].map(correction_dict)
test['energy_label'] = test['energy_label'].map(correction_dict)
#I also transform this for avoiding problems when plotting
train['type_of_construction'] = train['type_of_construction'].map({'Bestaande bouw':'-1','Nieuwbouw':'1','0':'0'})
test['type_of_construction'] = test['type_of_construction'].map({'Bestaande bouw':'-1','Nieuwbouw':'1','0':'0'})
plt.figure(figsize=(8,10))
sns.heatmap(train[flag_var])
#plt.xticks(rotation=45)
n_row = train.shape[0]
nan_count = train[train[flag_var]==1][flag_var].sum()
nan_percentage = round(nan_count/n_row*100).sort_values(ascending=False)
nan_percentage.plot.bar()
var_to_plot = 'log_price'

sns.distplot(train[var_to_plot],bins=30,fit=norm)
plt.xlabel(var_to_plot)
#skewness and kurtosis
print("Skewness: {}".format(train[var_to_plot].skew()))
print("Kurtosis: {}".format(train[var_to_plot].kurt()))
sns.set(style="darkgrid")
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for j,var in enumerate(num_var):
    fig.add_subplot(4,3,j+1)
    #Drop missing values
    if 'flg_missing_'+var in flag_var:
        sns.distplot(train[train['flg_missing_'+var]==0][var],bins=20)   
    else:
        sns.distplot(train[var],bins=20)
    plt.xlabel(var)
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for j,var in enumerate(cat_var):
    fig.add_subplot(3,2,j+1)
    #Drop missing values
    if 'flg_missing_'+var in flag_var:
        sns.countplot(train[train['flg_missing_'+var]==0][var],color='lightsteelblue')   
    else:
        sns.countplot(train[var],color='lightsteelblue')
    plt.xlabel(var)
    plt.xticks(rotation=60)
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for j,var in enumerate(num_var):
    fig.add_subplot(5,4,j+1)
    #Drop missing values
    if 'flg_missing_'+var in flag_var:
        sns.scatterplot(train[train['flg_missing_'+var]==0][var],
                    train[train['flg_missing_'+var]==0]['log_price']) 
    else:
        sns.scatterplot(train[var],train['log_price']) 
    plt.xlabel(var)
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for j,var in enumerate(cat_var):
    fig.add_subplot(3,2,j+1)
    #Drop missing values
    if 'flg_missing_'+var in flag_var:
        sns.boxplot(train[train['flg_missing_'+var]==0][var],
                    train[train['flg_missing_'+var]==0]['log_price']) 
    else:
        sns.boxplot(train[var],train['log_price']) 
    plt.xlabel(var)
#standarizing data
from sklearn.preprocessing import StandardScaler
log_price_scaled = StandardScaler().fit_transform(np.array(train['log_price']).reshape(-1, 1))
outliers = []
threshold = 4.5
g = sns.distplot(log_price_scaled,bins=30)
plt.axvline(x=threshold,c='r')
plt.axvline(x=-threshold,c='r')
plt.xlabel('log_price_scaled')
#indexes of outliers
index = np.where(np.absolute(log_price_scaled) > threshold)[0]
outliers += list(index)
threshold = 4.5
n_photos_scaled = StandardScaler().fit_transform(np.array(train['living_area']).reshape(-1, 1))
g = sns.distplot(n_photos_scaled,bins=30)
plt.axvline(x=threshold,c='r')
plt.axvline(x=-threshold,c='r')
plt.xlabel('living_area_scaled')
#indexes of outliers
index = np.where(np.absolute(n_photos_scaled) > threshold)[0]
outliers += list(index)
index = train[train['flg_missing_year']==0][train['year']<1700].index
outliers += list(index)
fig = plt.figure(figsize=(20,8))
fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig.add_subplot(1,2,1)
plt.scatter(train[train['flg_missing_'+'year']==0]['year'], 
            train[train['flg_missing_'+'year']==0]['log_price'])
plt.axvline(x=1725,c='r')

fig.add_subplot(1,2,2)
plt.scatter(train[train['flg_missing_'+'year']==0]['year'].drop(index), 
            train[train['flg_missing_'+'year']==0]['log_price'].drop(index))
anomalies = train[train['other_indoor_space']>train['living_area']].index
anomalies = list(anomalies)
fig = plt.figure(figsize=(16,4))
fig.subplots_adjust(hspace=0.4)

fig.add_subplot(1,2,1)
sns.scatterplot(train[train['flg_missing_'+'other_indoor_space']==0]['living_area'],
                train[train['flg_missing_'+'other_indoor_space']==0]['other_indoor_space']) 
sns.scatterplot(train['living_area'].iloc[anomalies],train['other_indoor_space'].iloc[anomalies])

new_train = train.drop(anomalies)
fig.add_subplot(1,2,2)
sns.scatterplot(new_train[train['flg_missing_'+'other_indoor_space']==0]['living_area'],
                new_train[train['flg_missing_'+'other_indoor_space']==0]['other_indoor_space']) 
fig = plt.figure(figsize=(16,4))
fig.subplots_adjust(hspace=0.4)

fig.add_subplot(1,2,1)
sns.scatterplot(train[train['flg_missing_'+'other_area']==0]['living_area'],
                train[train['flg_missing_'+'other_area']==0]['other_area'])

index = train[train['other_area']>train['living_area']].index
sns.scatterplot(train['living_area'].iloc[index],train['other_area'].iloc[index])
anomalies += list(index)

new_train = train.drop(index)
fig.add_subplot(1,2,2)
sns.scatterplot(new_train[train['flg_missing_'+'other_area']==0]['living_area'],
                new_train[train['flg_missing_'+'other_area']==0]['other_area']) 
anomalies = set(anomalies)
train = train.drop(anomalies)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
y = np.array(train['log_price'])

def benchmark(y,n_folds=10,n_iters=5):
    
    kf = KFold(n_splits=n_folds)
    cv_scores = []
    
    for iter in range(n_iters):
        for train_index, test_index in kf.split(y):

            n_train = test_index.shape[0]
            y_pred = np.zeros(n_train) + y[train_index].mean()
            y_true = y[test_index]
            cv_scores.append(mean_squared_error(y_true,y_pred))
            
    return np.sqrt(cv_scores)

mean_baseline = benchmark(y)
print('Benchmark:')
print('RMSE score: {}'.format(round(np.mean(mean_baseline),2)))
print('Std : {}'.format(np.std(mean_baseline)))
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Imputer
warnings.filterwarnings('ignore',category=DeprecationWarning)
estimator = LinearRegression()
n_splits = 10
n_iters = 5
lr_baseline_train = []
lr_baseline_test = []

X = train[['living_area','flg_missing_living_area']]
y = train['log_price']

for iter in range(n_iters):

    cv = KFold(n_splits=n_splits,shuffle=True)
    cv_iter = list(cv.split(X)) 

    #cross validation
    for train_index, test_index in cv_iter:

        X_train, X_val = (X.iloc[train_index,:], X.iloc[test_index,:])
        y_train, y_val = (y.iloc[train_index], y.iloc[test_index])

        #Calculate the average from train set and impute on train and validation set
        imputer = Imputer(missing_values=0)
        X_train['living_area'] = imputer.fit_transform(np.array(X_train['living_area']).reshape(-1, 1))
        X_val['living_area'] = imputer.transform(np.array(X_val['living_area']).reshape(-1, 1))

        #Calculate the scale from train set and impute on train and validation set
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        #fit the estimator
        estimator.fit(X_train,y_train)

        #predictions
        y_pred = estimator.predict(X_train)
        lr_baseline_train.append(mean_squared_error(y_train,y_pred))
        y_pred = estimator.predict(X_val)
        lr_baseline_test.append(mean_squared_error(y_val,y_pred))
            
lr_baseline_test = np.sqrt(lr_baseline_test)
lr_baseline_train = np.sqrt(lr_baseline_train)
    
print('mean train score: {}'.format(np.mean(lr_baseline_train)))
print('std test score: {}'.format(np.std(lr_baseline_train)))
print('--'*20)
print('mean test score: {}'.format(np.mean(lr_baseline_test)))
print('std test score: {}'.format(np.std(lr_baseline_test)))
#Let´s concatenate the data sets
all_data = pd.concat([train.drop('log_price',axis=1),test],axis=0)
#save the target in other variable
y = train['log_price']
for var in all_var:
    
    if 'flg_missing_'+var in flag_var:
        index = all_data[all_data['flg_missing_'+var]==1][var].index
        all_data.loc[index,var] = -999
        
all_data = all_data.drop(flag_var,axis=1)
#for debugging
all_data.head()
nan_count = train[flag_var].sum(axis=1)
all_data['nan_count'] = nan_count
all_data['n_photos'] = all_data['n_photos'].apply(lambda x: 1 if x > 0 else 0)
all_data['n_photos_360'] = all_data['n_photos_360'].apply(lambda x: 1 if x > 0 else 0)
all_data['energy_label'] = all_data['energy_label'].map({ 'A':'1',
                                                        'B':'2',
                                                        'C':'3',
                                                        'D':'4',
                                                        'E':'5',
                                                        'F':'6',
                                                        'G':'7',
                                                        'N':'8'})
n_train = train.shape[0]
rf_train = all_data.iloc[:n_train,:]
test = all_data.iloc[n_train:,:]
#levels with less than threshold observations go into the new category level
threshold = 30
count = train.groupby('located_on')['located_on'].count()
all_data['located_on'] = all_data['located_on'].apply(lambda x: '9' if x in count[count<threshold].index else x)
def mean_target_encoding(x_tr, x_val,y,cols):
    '''
    Calculate mean target encoding 
    
    Arguments:
    ------------
    x_tr : Pandas.DataFrame
        dataframe to calculate encoding
    x_val : Pandas.DataFrame
        dataframe to map encoding
    y : Pandas.Serie
        the target
    cols : list
        cols to calculate encoding on
    
    Return:
    ------------
    x_tr : Pandas.DataFrame
        encoded dataframe
    x_val : Pandas.DataFrame
        encoded dataframe

    '''
    alpha = 2
    temp = pd.concat([x_tr,y],axis=1)
    global_mean = y.mean()
    
    for col in cols:
        
        means = temp.groupby(col)[y.name].mean()
        #print(means)
        n_rows = temp.groupby(col)[col].count()
        #print(n_rows)
        encode = (means*n_rows + global_mean*alpha)/(n_rows+alpha)
        #print(encode)
        x_tr[col+'_mean_target'] = x_tr[col].map(encode)
        x_val[col+'_mean_target'] = x_val[col].map(encode)
        x_val[col+'_mean_target'].fillna(global_mean,inplace=True)
        
    return x_tr, x_val
def freq_encoding(serie_tr, serie_val):
    '''
    Calculate the frequency encoding and map the inputs.
    
    Arguments:
    ------------
    serie_tr : Pandas.Series
         serie to calculate the encoding on.
    serie_val : Pandas.Series
         serie to map the encoding on.
    
    Return:
    ------------
    serie_tr : Pandas.Series
        encoded serie
    serie_val : Pandas.Series
        encoded serie

    '''
    
    n_row = serie_tr.shape[0]
    freq_dict = {}
    categories = serie_tr.unique()
    
    for category in categories:
        
        category_freq = serie_tr[serie_tr==category].count()/n_row*100
        freq_dict[category] = category_freq
    
    global_mean = np.mean([value for value in freq_dict.values()])
    #print([value for value in freq_dict.values()])
    #print(np.mean([value for value in freq_dict.values()]))
    
    serie_tr = serie_tr.map(freq_dict)
    serie_val = serie_val.map(freq_dict)
    serie_val = serie_val.fillna(global_mean)
    
    return serie_tr,serie_val
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold,cross_val_score,validation_curve,StratifiedKFold
def rmse_cv(estimator,X,y,n_splits,n_iter,mean_target=True,cols_to_encode=[]):
    '''
    Evaluate a score by cross-validation.
    Arguments:
    ------------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : pandas.DataFrame
        The data to fit.
    y : pandas.Series
        The target variable to try to predict.
    n_splits : int
        Number of folds for cross-validation
    n_iter : int 
        Times to repeat validation.
    mean_target : bool.    
    cols_to_encode : list. 
        Columns to encode.
        
     Return:
    ------------- 
    train_scores : list. 
                    Training scores.
    test_scores : list. 
                    Validation scores.
    '''
    train_scores = []
    test_scores = []

    for i in range(n_iter):
        
        cv = KFold(n_splits=n_splits,shuffle=True)
        cv_iter = list(cv.split(X, y)) 

        for train_index, test_index in cv_iter:

            X_train, X_val = (X.iloc[train_index,:], X.iloc[test_index,:])
            y_train, y_val = (y.iloc[train_index], y.iloc[test_index]) 

            if mean_target==True:
                #Calculate the average of target from train folds and impute on train and validation folds
                X_train, X_val = mean_target_encoding(X_train,X_val,y_train,cols_to_encode)          
                #fit the estimator
            estimator.fit(X_train.drop(cols_to_encode,axis=1).values,y_train.values)
            #predicts on train folds
            y_pred = estimator.predict(X_train.drop(cols_to_encode,axis=1).values)
            train_scores.append(mean_squared_error(y_train,y_pred))
            #predicts on validation fold
            y_pred = estimator.predict(X_val.drop(cols_to_encode,axis=1).values)
            test_scores.append(mean_squared_error(y_val,y_pred))
            
        #for debugging
        #print(X_train.drop(cols_to_encode,axis=1).head())

    test_scores = np.sqrt(test_scores)
    train_scores = np.sqrt(train_scores)
    return train_scores, test_scores
var_to_drop = ['total_area','other_indoor_space','other_area','n_weeks_old','own_ground']

X  = rf_train.drop(var_to_drop,axis=1)


rf_model = RandomForestRegressor(200,
                                 max_depth=8,
                                 max_features=6,
                                 verbose=0)

rf_model_train, rf_model_test = rmse_cv(rf_model,X,y,5,10,True,['zipcode'])


print('mean test score: {}'.format(np.mean(rf_model_test)))
print('std test score: {}'.format(np.std(rf_model_test)))
print('--'*20)