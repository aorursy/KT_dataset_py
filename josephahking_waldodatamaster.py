# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import re



# Modelling Algorithms



from sklearn.svm import SVC, LinearSVC

from sklearn import linear_model



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.feature_selection import RFECV



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns





from sklearn.utils import shuffle
# get TMDB Box Office Prediction train & test csv files as a DataFrame

train = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")

test  = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")
# deal with release day

#train.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'

test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'

def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

test['release_date'] = pd.to_datetime(test['release_date'])

print(type(train['release_date'][0]))

def process_date(df):

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    

    return df



train = process_date(train)

test = process_date(test)

print(train['release_date'])
# Shuffle data

tr_shuffle = shuffle(train, random_state = 43).reset_index(drop=True)



#selected_features = {"budget","popularity", "release_date_year", "release_date_day", "release_date_weekday", "release_date_month"}

# Split into training and validation data set

#data_train=tr_shuffle[0:2499]

#data_validate=tr_shuffle[2500:2999]

# Create input and out for training and validation set

#data_tr_x = data_train[selected_features]

#data_tr_y = data_train[{"revenue"}]

#data_val_x = data_validate[selected_features]

#data_val_y = data_validate[{"revenue"}]

#data_val_title = data_validate[{"original_title"}]



# Select features

selected_features = {"budget","popularity", "release_date_year", "release_date_day", "release_date_weekday", "release_date_month"}



tr_shuffle_revenue = tr_shuffle["revenue"]

#create x

tr_shuffle_feature_list = tr_shuffle[selected_features]

# create title for validation

#tr_shuffle_title = tr_shuffle[{"original_title"}]

train_genres = tr_shuffle[{"genres"}]

#print(train_genres.genres.unique())

#print(train_genres)

#vec = DictVectorizer()

#train_genres.head()

p = re.compile(r'\d+')

idList = []

for index, row in train_genres.iterrows():

    #print(row)

    for col in row.iteritems():

        if(type(col[1]) is not str):

            continue

        if(len(col[1]) <= 2):

                continue

        allIDs = p.findall(col[1])

        for id in allIDs:

            idNum = eval(id);

            if idNum not in idList:

                idList.append(idNum)

#train_genres.genres.unique()





print(idList)

Column_Names = []

for id in idList:

    name= 'Genre'+ str(id)

    Column_Names.append(name)





train_genre_list = pd.DataFrame()

idx = 0

for name in Column_Names:

    train_genre_list.insert(idx,name,np.zeros(len(train_genres)))

    idx = idx + 1





#print(train_genre_list)

for rowidx, row in train_genres.iterrows():

    for col in row.iteritems():

        if(type(col[1]) is not str):

            continue

        if(len(col[1]) <= 2):

                continue

        allIDs = p.findall(col[1])

        for id in allIDs:

            idNum = eval(id);

            train_genre_list.loc()

            train_genre_list[Column_Names[idList.index(idNum)]].loc[rowidx]=1

            #print(each_row)   

    #train_genre_list.

#vec.fit_transform(train_genres).toarray()





print(train_genre_list)

    

tr_shuffle_feature_list.join(train_genre_list )

# Split into training and validation data set of feature list

data_train=tr_shuffle_feature_list[0:2499]

data_validate=tr_shuffle_feature_list[2500:2999]

#slip into training and validation for y

data_train_y=tr_shuffle_revenue[0:2499]

data_validate_y=tr_shuffle_revenue[2500:2999]



data_tr_x = data_train

data_tr_y = data_train_y

data_val_x = data_validate

data_val_y = data_validate_y

data_val_title = tr_shuffle[{"original_title"}][2500:2999]



def plot_correlation_map( df ):

    corr = train.corr()

    _ , ax = plt.subplots( figsize =( 23 , 22 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()
train.corr()

print(train["release_date_day"])
# edit cast Show first main caracter in first movie with ID and name

print(train.cast.shape[0])

#for index in train.cast:

#    print(int((index.split('\'id\':'))[1].split(',')[0]))

#    print((index.split('\'name\': \''))[1].split('\',')[0])

print(int((train.cast[1].split('\'id\':'))[1].split(',')[0]))

print((train.cast[1].split('\'name\': \''))[1].split('\',')[0])
np.count_nonzero(train.budget)
train.describe()

data=pd.concat([train['budget'],train['revenue']],axis=1)

data.plot.scatter(x='budget',y='revenue',xlim=(0,1e7),ylim=(0,1e8))
#plot_correlation_map(train)
## Splitting into Test and validation data and feature selection

#

## Selecting features Budget and Popularity

#train_mod = train[{"budget","popularity"}]

#

## Selecting the first 2001 indices of the training data for training

#train_train = train_mod[0:2000]

## Selecting the rest of the training data for validation

#train_val= train_mod[2001:2999]

#

## Obtain labels

#train_mod_y = train[{"revenue"}]

#train_train_y = train_mod_y[0:2000]

#train_val_y= train_mod_y[2001:2999]

#train_val_title = train["original_title"][2001:2999]
# Initialize and train a linear regression (Lasso) model



model1 = linear_model.Lasso(alpha=1.2, normalize = False, tol = 1e-3)

model1.fit(data_tr_x,data_tr_y.values.ravel())

# Evaluate on the training data

res1 = model1.predict(data_val_x)

# Obtain R2 score (ordinary least square)

acc_lasso = model1.score(data_val_x, data_val_y)

acc_lasso_trained = model1.score(data_tr_x, data_tr_y)

print(acc_lasso, acc_lasso_trained)

# Create the table for comparing predictions with labels

evaluation = pd.DataFrame({'Title': data_val_title.values.ravel(),'Prediction': res1.round(), 'Actual revenue': data_val_y.values.ravel(), 'Relative error': res1/data_val_y.values.ravel()})

evaluation
# Initialize and train a ridge regression model



model2 = linear_model.Ridge(alpha=100000.0)

model2.fit(data_tr_x,data_tr_y.values.ravel())

# Evaluate on the training data

res2 = model2.predict(data_val_x)

# Obtain R2 score (ordinary least square)

acc_ridge = model2.score(data_val_x, data_val_y)

acc_ridge_trained = model2.score(data_tr_x, data_tr_y)

# Create the table for comparing predictions with labels

evaluation = pd.DataFrame({'Title': data_val_title.values.ravel(),'Prediction': res2.round(), 'Actual revenue': data_val_y.values.ravel(), 'Relative error': res2/data_val_y.values.ravel()})

evaluation
# Initialize and train a elasticNet model

model3 = linear_model.ElasticNet(random_state = 0)

model3.fit(data_tr_x,data_tr_y.values.ravel())

# Evaluate on the training data

res3 = model3.predict(data_val_x)

# Obtain R2 score (ordinary least square)

acc_elasticNet = model3.score(data_val_x, data_val_y)

acc_elasticNet_trained = model3.score(data_tr_x, data_tr_y)

# Create the table for comparing predictions with labels

evaluation = pd.DataFrame({'Title': data_val_title.values.ravel(),'Prediction': res3.round(), 'Actual revenue': data_val_y.values.ravel(), 'Relative error': res3/data_val_y.values.ravel()})

evaluation
# model 4

import lightgbm as lgb

import eli5

#data_tr_y = np.log1p(data_tr_y)

#data_val_y = np.log1p(data_val_y)

params = {'num_leaves': 50,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 6,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

model4 = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

model4.fit(data_val_x, data_val_y, 

        eval_set=[(data_tr_x,data_tr_y.values.ravel()), (data_val_x, data_val_y)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=2000)

acc_lgb = model4.score(data_val_x, data_val_y)

acc_lgb_trained = model4.score(data_tr_x, data_tr_y)



eli5.show_weights(model4, feature_filter=lambda x: x != '<BIAS>')
models = pd.DataFrame({

    'Model': ['Ridge regression', 'Lasso', 'ElasticNet', 'lgb'],

    'Score': [acc_ridge, acc_lasso, acc_elasticNet, acc_lgb],

    'Score for training data': [acc_ridge_trained, acc_lasso_trained, acc_elasticNet_trained, acc_lgb_trained]})

models.sort_values(by='Score', ascending=False)
# Function for displaying evaluation metrics

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2=[0] * 3

rms= [0] * 3

mae = [0] * 3



def evaluateModels(predictions, ground_truth, model_names):

    for idx, prediction in enumerate(predictions):

        r2[idx] = r2_score(ground_truth, prediction)

        rms[idx] = np.sqrt(mean_squared_error(ground_truth, prediction))

        mae[idx] = mean_absolute_error(ground_truth, prediction)

    

    scores = pd.DataFrame({

    'Model': model_names,

    'R2': r2,

    'RMS': rms,

    'MAE': mae})

    print(scores)

    # Create error array

    prediction_error = ground_truth - predictions

    print(type(prediction_error))

    ax = sns.boxplot(data = np.transpose(prediction_error), orient = 'h')

    #return [r2, rms, mae]

                   
# Evaluate and compare the different models

prediction_vector = [res1, res2, res3]

model_names = ['Ridge regression', 'Lasso', 'ElasticNet']

evaluateModels(prediction_vector, data_val_y.values.ravel(), model_names)
# Show the top 5 prediction (with minimum error) and the bottom 5 predictions

# 

modelres = res1

absolute_error =  np.abs(modelres - data_val_y.values.ravel())

relative_error = absolute_error/data_val_y.values.ravel()



evaluation = pd.DataFrame({'Title': data_val_title.values.ravel(), 'budget': data_val_x['budget'].values.ravel(), 'popularity': data_val_x['popularity'].values.ravel(),'Prediction': modelres.round(), 'Actual revenue': data_val_y.values.ravel(), 'Absolute error 1': absolute_error, 'Relative error 1': relative_error})



evaluation.sort_values(by=['Relative error 1'])