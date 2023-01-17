#Fundation packages

import pandas as pd

import numpy as np

import re

#pd.set_option('precision', 3)        #Setting the third-decimal point



#Visualisation packages

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

%matplotlib inline

warnings.filterwarnings('ignore')



#Model packages

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.base import clone

from xgboost import XGBRegressor
# Load in the train and test datasets

tr = pd.read_csv('../input/train.csv')

te = pd.read_csv('../input/test.csv')

# The relevant address files are derived through GoogleMap API from latitude and longitude values

train_ad = pd.read_csv('../input/train_address.csv')

test_ad = pd.read_csv('../input/test_address.csv')



# Store ID for easy access

submission = pd.read_csv('../input/sampleSubmission.csv', index_col='Id')

tr.head()
tr.info()
te.info()
#Now get the postarea of the address first

postareas=['London EC','London E','London NW','London N','London SE','London SW','London WC','London W']

def get_postarea(address):

    postareas=['London EC','London E','London NW','London N','London SE','London SW','London WC','London W']

    for area in postareas:

        search = re.search(area, address)

        if search:

            return area
train_ad['postarea'] = pd.Series( np.zeros(1000),index=train_ad.index)

test_ad['postarea'] = pd.Series( np.zeros(1000),index=test_ad.index)

for i in range(0,1000):

    train_ad['postarea'][i] = get_postarea(train_ad['address'][i])

    test_ad['postarea'][i]= get_postarea(test_ad['address'][i]) 
train_ad[train_ad['postarea'].isnull()]
test_ad[test_ad['postarea'].isnull()]
#Then fill the none value manually

train_ad['postarea'][3]='London WC'

train_ad['postarea'][112]='London SW'

train_ad['postarea'][143]='London WC'

train_ad['postarea'][184]='London WC'

train_ad['postarea'][266]='London SW'

train_ad['postarea'][277]='London SE'

train_ad['postarea'][595]='London SE'

train_ad['postarea'][715]='London SW'



test_ad['postarea'][318]='London SW'

test_ad['postarea'][530]='London WC'

test_ad['postarea'][611]='London E'

test_ad['postarea'][870]='London SW'
tr = pd.concat([tr,train_ad],axis = 1)

te = pd.concat([te,test_ad],axis = 1)

tr = tr.drop(columns='address')

te = te.drop(columns='address')

#Copy the dataframe for most of the models

train= tr.copy()

test= te.copy()
train.head()
#Showing the unique values of the non-numerical columns

str_lis = train.select_dtypes(include='object').columns.tolist()

fig = tls.make_subplots(rows=4, cols=3)

loc=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]

for i in range(0,len(str_lis)):

    trace0 = go.Histogram(x=train[str_lis[i]],

                          name = str_lis[i],

                          opacity=0.8)

    fig.append_trace(trace0,loc[i][0],loc[i][1])

    

fig['layout'].update(height=800, width=1000,title='Histograms regarding string features: Training set')    

py.iplot(fig)
#Showing the box plots of the numerical columns

num_lis = train.select_dtypes(exclude='object').columns.tolist()

num_lis.remove('Id')

num_lis.remove('price')

fig = tls.make_subplots(rows=6, cols=4)

loc=[(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),

     (4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4),(6,1),(6,2),(6,3),(6,4)]

for i in range(0,len(num_lis)):

    trace0 = go.Box(y=train[num_lis[i]],

                    name = num_lis[i],

                    boxpoints = 'outliers',

                    marker = dict(

                        outliercolor = 'rgb(8,81,156)'))

    fig.append_trace(trace0,loc[i][0],loc[i][1])

    

fig['layout'].update(height=1500, width=1000,title='Box plots regarding numerical values: Training set')    

py.iplot(fig)
fig = tls.make_subplots(rows=6, cols=4)

loc=[(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),

     (4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4),(6,1),(6,2),(6,3),(6,4)]

for i in range(0,len(num_lis)):

    trace0 = go.Scatter(x=train[num_lis[i]],

                        y=train['price'],

                        name=num_lis[i],

                        mode='markers')

    fig.append_trace(trace0,loc[i][0],loc[i][1])

    

fig['layout'].update(height=1500, width=1000,title='Scatter plots with price: Training set')    

py.iplot(fig)
#Showing the box plots of the numerical columns

num_lis = train.select_dtypes(exclude='object').columns.tolist()

num_lis.remove('Id')

num_lis.remove('price')

fig = tls.make_subplots(rows=6, cols=4)

loc=[(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),

     (4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4),(6,1),(6,2),(6,3),(6,4)]

for i in range(0,len(num_lis)):

    trace0 = go.Box(y=test[num_lis[i]],

                    name = num_lis[i],

                    boxpoints = 'outliers',

                    marker = dict(

                        outliercolor = 'rgb(8,81,156)'))

    fig.append_trace(trace0,loc[i][0],loc[i][1])

    

fig['layout'].update(height=1500, width=1000,title='Box plots regarding numerical values: Test set')    

py.iplot(fig)
#Showing the box plots of the numerical columns: 'host_listings_count'

fig = tls.make_subplots(rows=1, cols=2)

trace0 = go.Box(y=train['host_listings_count'],

                name = 'host_listings_count',

                boxpoints = 'outliers',

                marker = dict(

                outliercolor = 'rgb(8,81,156)'))

trace1 = go.Scatter(x=train['host_listings_count'],

                    y=train['price'],

                    name='host_listings_count',

                    mode='markers')



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)

fig['layout'].update(height=400, width=700,title='EDA regarding numerical values: host_listings_count')    

py.iplot(fig)
#Showing the box plots of the numerical columns: 'security_deposit'

fig = tls.make_subplots(rows=1, cols=2)

trace0 = go.Box(y=train['security_deposit'],

                name = 'security_deposit',

                boxpoints = 'outliers',

                marker = dict(

                outliercolor = 'rgb(8,81,156)'))

trace1 = go.Scatter(x=train['security_deposit'],

                    y=train['price'],

                    name='security_deposit',

                    mode='markers')



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)

fig['layout'].update(height=400, width=700,title='EDA regarding numerical values: security_deposit')    

py.iplot(fig)
train[num_lis].corrwith(train['price']).abs().sort_values(ascending=False)
#Numerical features relevant parameters

train[num_lis].head(10)
train[num_lis].mean()
train[num_lis].std()
#Define a function to replace the null values in numerical features

#There are 2 ways for replacing the null values,(avg-std,avg+std) or avg

def fill_na(feature,X):

    np.random.seed(1234)

    avg = X[feature].mean()

    std = X[feature].std() 

    median = X[feature].median()

    null_count = X[feature].isnull().sum()

    

    if feature == 'reviews_per_month':

        null_random_list = np.random.uniform( 0, avg + std, size = null_count).round(2)

    elif feature == 'host_listings_count':

        null_random_list = np.random.randint( avg - 1 , avg + 1, size = null_count)

    else:

        null_random_list = np.random.randint(avg - std, avg + std, size = null_count)



    X[feature][np.isnan(X[feature])] = null_random_list

    

    return X[feature]
def preprocessing(dataset):

#Build a new feature regarding the distance to National Gallery, London Coordinates: (51.5089° N, 0.1283° W)

    dataset['distance_to_NationalPark'] = np.sqrt((dataset['latitude']-51.5089)**2+(dataset['longitude']+0.1283)**2)

    

#Replace types in experiences offerd into numerical values  

    dataset['experiences_offered'] = dataset['experiences_offered'].replace('none', int(0))    

    dataset['experiences_offered'] = dataset['experiences_offered'].replace('family', int(1))

    dataset['experiences_offered'] = dataset['experiences_offered'].replace('social', int(2)) 

    dataset['experiences_offered'] = dataset['experiences_offered'].replace('business', int(3)) 

    dataset['experiences_offered'] = dataset['experiences_offered'].replace('romantic', int(4)) 

        

#Replace all columns with False and Ture into 0 and 1

    dataset['host_is_superhost'] = dataset['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

    dataset['host_identity_verified'] = dataset['host_identity_verified'].apply(lambda x: 1 if x == 't' else 0)

    dataset['instant_bookable'] = dataset['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)

    dataset['require_guest_profile_picture'] = dataset['require_guest_profile_picture'].apply(lambda x: 1 if x == 't' else 0)

    dataset['require_guest_phone_verification'] = dataset['require_guest_phone_verification'].apply(lambda x: 1 if x == 't' else 0)

 

    

#Fill host response rate null 0

    dataset['host_response_rate'] = dataset['host_response_rate'].fillna(int(0)) 

    

#Replace host response time strings to numerical values

    dataset['host_response_time'] = dataset['host_response_time'].fillna(int(0)) 

    dataset['host_response_time'] = dataset['host_response_time'].replace('within an hour', int(1))

    dataset['host_response_time'] = dataset['host_response_time'].replace('within a day', int(24)) 

    dataset['host_response_time'] = dataset['host_response_time'].replace('within a few hours', int(6)) 

    dataset['host_response_time'] = dataset['host_response_time'].replace('a few days or more', int(48))  



#Fill the outlier value

    median = dataset.drop(dataset[dataset['host_listings_count'] > 400].index)['host_listings_count'].median()

    dataset['host_listings_count']= dataset['host_listings_count'].mask(dataset['host_listings_count'] > 400, median)

    median = dataset.drop(dataset[dataset['security_deposit'] > 1300].index)['security_deposit'].median()

    dataset['security_deposit']= dataset['security_deposit'].mask(dataset['security_deposit'] > 1300, median)

    

#Fill the null value in following 10 numerical features 

    dataset['security_deposit'] = fill_na('security_deposit',dataset)

    dataset['review_scores_rating'] = fill_na('review_scores_rating',dataset)

    dataset['review_scores_accuracy'] = fill_na('review_scores_accuracy',dataset)

    dataset['review_scores_cleanliness'] = fill_na('review_scores_cleanliness',dataset)

    dataset['review_scores_checkin'] = fill_na('review_scores_checkin',dataset)

    dataset['review_scores_communication'] = fill_na('review_scores_communication',dataset)

    dataset['review_scores_location'] = fill_na('review_scores_location',dataset)

    dataset['review_scores_value'] = fill_na('review_scores_value',dataset)

    dataset['reviews_per_month'] = fill_na('reviews_per_month',dataset)

    

    dataset['cleaning_fee'] = dataset['cleaning_fee'].fillna(0)

#    dataset['security_deposit'] = dataset['security_deposit'].fillna(0)

#    dataset['review_scores_rating'] = dataset['review_scores_rating'].fillna(0)

#    dataset['review_scores_accuracy'] = dataset['review_scores_accuracy'].fillna(0)

#    dataset['review_scores_cleanliness'] = dataset['review_scores_cleanliness'].fillna(0)

#    dataset['review_scores_checkin'] = dataset['review_scores_checkin'].fillna(0)

#    dataset['review_scores_communication'] = dataset['review_scores_communication'].fillna(0)

#    dataset['review_scores_location'] = dataset['review_scores_location'].fillna(0)

#    dataset['review_scores_value'] = dataset['review_scores_value'].fillna(0)

#    dataset['reviews_per_month'] = dataset['reviews_per_month'].fillna(0)

    

#Applying one-hot code labeling to following 4 features

    s = pd.get_dummies(dataset['room_type'])

    dataset = pd.concat([dataset,s],axis = 1)

    s = pd.get_dummies(dataset['property_type'])

    s = s[['Apartment', 'House','Townhouse', 'Serviced apartment', 'Other']]

    dataset = pd.concat([dataset,s],axis = 1)

    s = pd.get_dummies(dataset['cancellation_policy'])

    s = s.drop(columns=['super_strict_60', 'super_strict_30'])

    dataset = pd.concat([dataset,s],axis = 1)

    s = pd.get_dummies(dataset['postarea'])

    dataset = pd.concat([dataset,s],axis = 1)

    

    return dataset
#Preprocess train and test set

train = preprocessing(train)

train = train.dropna()

test = preprocessing(test)



drop_elements = ['Id', 'room_type', 'bed_type', 'property_type', 'cancellation_policy', 'postarea', #The columns that need to be dropped

                 'Other', 'require_guest_phone_verification', 'require_guest_profile_picture', #The columns with low feature importances

                 'Shared room', 'Townhouse', 'experiences_offered','London E','London NW','London SE'] #The columns with low feature importances

train = train.drop(columns = drop_elements)

test  = test.drop(columns = drop_elements)



test_fillna_elements = ['host_listings_count', 'bathrooms', 'bedrooms', 'beds']

for feature in test_fillna_elements:

    test[feature] = fill_na(feature,test)

    #test[feature] = test[feature].fillna(0)



print('The shape of the preprocessed training set:',train.shape)

train.head()
train.corrwith(train['price']).drop('price').abs().sort_values(ascending=False).head(20)
Y = train['price'].ravel()

train = train.drop(['price'],axis=1)

X_train = train

X_test = test
feature_lis = X_train.columns.tolist()
#Scaling process

Xtr_number=X_train.shape[0]

alldata = pd.concat([X_train,X_test])

scaler = StandardScaler()

alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

X_tr = alldata[:Xtr_number]

X_te = alldata[Xtr_number:]



X=X_tr.values

Xt=X_te.values

print(X.shape)

print(Y.shape)

print(Xt.shape)
#Random_state and number of folds

SEED = 4321 # for reproducibility

NFOLDS = 25 # set folds for out-of-fold prediction

#The function to calculate the oof predictions        

def get_oof(clf, x_train, y_train, x_test, n_folds):

    oof_train = np.zeros(x_train.shape[0])

    oof_test = np.zeros(x_test.shape[0])

    feature_importance_ = np.zeros((n_folds, x_train.shape[1]))

    intercept_ = np.zeros(n_folds)

    oof_test_kf = np.zeros((n_folds, x_test.shape[0]))

    kf = KFold(n_splits= n_folds, shuffle=True, random_state=SEED)

    for i, (train_index, fold_index) in enumerate(kf.split(x_train,y_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_fo = x_train[fold_index]



        clf.fit(x_tr, y_tr)

        if clf == elnet:

            feature_importance_[i]=clf.coef_

            intercept_[i]=clf.intercept_

        elif clf == svr:

            feature_importance_[i]=clf.coef_ 

            intercept_[i]=clf.intercept_

        else:

            feature_importance_[i]=clf.feature_importances_

                



        oof_train[fold_index] = clf.predict(x_fo)

        oof_test_kf[i] = clf.predict(x_test)



    oof_test[:] = oof_test_kf.mean(axis=0)

    

    if clf == elnet:

        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), feature_importance_, intercept_

    elif clf == svr: 

        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), feature_importance_, intercept_

    else:

        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), feature_importance_





#The function to print the RMSE

def get_RMSE(a,b):

    print('RMSE: ',np.sqrt(mean_squared_error(a,b)))

    return 
elnet_params = {

    'alpha': 0.24,

    'l1_ratio': 0.7,

    'random_state':SEED,

}



elnet = ElasticNet(**elnet_params)

elnet.fit(X,Y)

elnet_train=elnet.predict(X)

elnet_test=elnet.predict(Xt)

print('ELNET:',elnet.score(X,Y))

get_RMSE(elnet_train,Y)

elnet_oof_train, elnet_oof_test, elnet_oof_coef, elnet_oof_intercept = get_oof(elnet, X, Y, Xt, n_folds=NFOLDS)

print('ELNET CV:')

get_RMSE(elnet_oof_train,Y)
feature_coef = pd.DataFrame({'feature': feature_lis, 'score': elnet_oof_coef.mean(axis=0)},columns=['feature','score'])

feature_coef = feature_coef.sort_values(by='score',ascending=True)

data = [go.Bar(

            x=feature_coef['score'],

            y=feature_coef['feature'],

            orientation = 'h'

)]

layout = go.Layout(height=900, width=800,title='ElasticNet OOF averaged feature coefficient')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
print('ElasticNet OOF averaged intercept:',elnet_oof_intercept.mean())

feature_coef
#Support Vector Regression

svr_params = {

    'kernel':'linear',

    'C': 5,

    'epsilon': 0.05

}



svr = SVR(**svr_params)

svr.fit(X,Y)

svr_train=svr.predict(X)

svr_test=svr.predict(Xt)

print('SVR:',svr.score(X,Y))

get_RMSE(svr_train,Y)

svr_oof_train, svr_oof_test, svr_oof_coef, svr_oof_intercept= get_oof(svr, X, Y, Xt, n_folds=NFOLDS)

print('SVR CV:')

get_RMSE(svr_oof_train,Y)

print('SVR OOF averaged intercept:',svr_oof_intercept.mean())

feature_importance = pd.DataFrame({'feature': feature_lis, 'score': svr_oof_coef.mean(axis=0)},columns=['feature','score'])

feature_importance = feature_importance.sort_values(by='score',ascending=True)

data = [go.Bar(

            x=feature_importance['score'],

            y=feature_importance['feature'],

            orientation = 'h'

)]

layout = go.Layout(height=900, width=800,title='SVR OOF averaged feature coefficient')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 100,

    'max_depth': 15,

    'min_samples_split': 3,

    'min_samples_leaf':1,

    'max_features' : 0.3,

    'verbose': 0,

    'random_state':SEED,

    'oob_score' : True

}



rf = RandomForestRegressor(**rf_params)

rf.fit(X,Y)

rf_train=rf.predict(X)

rf_test=rf.predict(Xt)

print('RF:',rf.score(X,Y))

get_RMSE(rf_train,Y)

rf_oof_train, rf_oof_test, rf_oof_importance = get_oof(rf, X, Y, Xt, n_folds=NFOLDS) # Random Forest

print('RF CV:')

get_RMSE(rf_oof_train,Y)
feature_importance = pd.DataFrame({'feature': feature_lis, 'score': rf_oof_importance.mean(axis=0)},columns=['feature','score'])

feature_importance = feature_importance.sort_values(by='score',ascending=True)

data = [go.Bar(

            x=feature_importance['score'],

            y=feature_importance['feature'],

            orientation = 'h'

)]

layout = go.Layout(height=900, width=800,title='RF OOF averaged feature importance')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# Gradient Boosting parameters

gb_params = {

    'n_estimators': 200,

    'learning_rate' : 0.05,

    'max_features' : 0.3,

    'max_depth': 6,

    'min_samples_leaf': 5,

    'subsample': 1,

    'verbose': 0,

    'random_state':SEED

}



gb = GradientBoostingRegressor(**gb_params)

gb.fit(X,Y)

gb_train=gb.predict(X)

gb_test=gb.predict(Xt)

print('GB:',gb.score(X,Y))

get_RMSE(gb_train,Y)

gb_oof_train, gb_oof_test, gb_oof_importance= get_oof(gb, X, Y, Xt, n_folds=NFOLDS) # Gradient Boosting

print('GB CV:')

get_RMSE(gb_oof_train,Y)
feature_importance = pd.DataFrame({'feature': feature_lis, 'score': gb_oof_importance.mean(axis=0)},columns=['feature','score'])

feature_importance = feature_importance.sort_values(by='score',ascending=True)

data = [go.Bar(

            x=feature_importance['score'],

            y=feature_importance['feature'],

            orientation = 'h'

)]

layout = go.Layout(height=900, width=800,title='GB OOF averaged feature importance')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#XGB

xgb_params = {

    'n_estimators': 100,

    'learning_rate' : 0.03,

    'gamma': 0.51,

    'max_depth': 25,

    'subsample': 1,

    'colsample_bytree': 1,

    'n_jobs': -1,

    'random_state':SEED

}



xgb = XGBRegressor(**xgb_params)

xgb.fit(X,Y)

xgb_train=xgb.predict(X)

xgb_test=xgb.predict(Xt)

print('XGB:',xgb.score(X,Y))

get_RMSE(xgb_train,Y)

xgb_oof_train, xgb_oof_test, xgb_oof_importance= get_oof(xgb, X, Y, Xt, n_folds=NFOLDS)

print('XGB CV:')

get_RMSE(xgb_oof_train,Y)

feature_importance = pd.DataFrame({'feature': feature_lis, 'score': xgb_oof_importance.mean(axis=0)},columns=['feature','score'])

feature_importance = feature_importance.sort_values(by='score',ascending=True)

data = [go.Bar(

            x=feature_importance['score'],

            y=feature_importance['feature'],

            orientation = 'h'

)]

layout = go.Layout(height=900, width=800,title='XGB OOF averaged feature importance')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#The class for stacking models to build a new model containing 2 tiers of models

class StackedModel():

    def __init__(self, base_models, meta_model, n_folds):

        self.base_models = base_models

        self.meta_model = clone(meta_model)

        self.n_folds = n_folds

   

    # Fit the data on clones of the tier 1 models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)

        

        # Train cloned tier 1 models then create out-of-fold predictions

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kf.split(X, y):

                instance = clone(model)

                instance.fit(X[train_index], y[train_index])

                self.base_models_[i].append(instance)

                y_pred= instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  tier 2 model using the out-of-fold predictions as new feature

        self.meta_model.fit(out_of_fold_predictions, y)

        return 

   

    #Do the predictions of all base models on the test data and use the averaged predictions as meta-features 

    #for the final prediction which is done through the tier 2 model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) 

            for base_models in self.base_models_ ])

        return self.meta_model.predict(meta_features)

    

    def coef_(self):

        return self.meta_model.coef_

    

    def intercept_(self):

        return self.meta_model.intercept_
lasso_params = {

    'alpha':0.01, 

    'random_state':SEED

}



lasso = Lasso(**lasso_params)

stacked_model = StackedModel(base_models =(gb, xgb, rf, svr, elnet), 

                            meta_model = lasso, n_folds = NFOLDS) 

stacked_model.fit(X, Y)

stacked_train = stacked_model.predict(X)

stacked_test = stacked_model.predict(Xt)

print('Stacked train prediction:',r2_score(Y, stacked_train))

get_RMSE(stacked_train,Y)
print('Stacked model intercept:',stacked_model.intercept_())

model_importance = pd.DataFrame({'Model': ['GB', 'XGB', 'RF', 'SVR', 'ELNET'], 'Score': stacked_model.coef_()},columns=['Model','Score'])

model_importance = model_importance.sort_values(by='Score',ascending=True)

data = [go.Bar(

            x=model_importance['Score'],

            y=model_importance['Model'],

            orientation = 'h'

)]

layout = go.Layout(height=400, width=600,title='Stacked model coefficient')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
submission['price'] = elnet_oof_test

submission.to_csv('ELNET.csv',index=True)

submission['price'] = svr_oof_test

submission.to_csv('SVR.csv',index=True)

submission['price'] = rf_oof_test

submission.to_csv('RF.csv',index=True)

submission['price'] = gb_oof_test

submission.to_csv('GB.csv',index=True)

submission['price'] = xgb_oof_test

submission.to_csv('XGB.csv',index=True)

submission['price'] = stacked_test

submission.to_csv('Stacked.csv',index=True)

print(submission.head())