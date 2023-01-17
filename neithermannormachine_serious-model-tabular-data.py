# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train_df
def feature_engineer(data):

    '''

    method to feature engineer any df, train or test

    '''

    

    df = data.copy()

    

    #add columns that indicate when first measurement was taken for each patient

    df['FirstWeek'] = df.groupby('Patient')['Weeks'].transform('min')

    

    first_fvc = (df.loc[df['Weeks'] == df['FirstWeek']][['Patient','FVC']]

                        .groupby('Patient')

                        .first() #some patients have multiple measurements in same week - get the first

                        .reset_index()

                         .rename(columns = {'FVC': 'FirstFVC'}) )

    

    df = df.merge(first_fvc, on = 'Patient') #add FirstFVC column

    

    #add column that indicates num weeks since first measurement

    

    df['WeeksPassed'] = df['Weeks'] - df['FirstWeek']

    

    #use PolyFeatures here instead, this isn't scalable

    #df['WeeksPassed_sqrt'] = np.maximum(0,df['WeeksPassed']) ** (1/2)

    df['WeeksPassed_sqrt'] = np.power(df['WeeksPassed'].abs(), 1/2) * np.sign(df['WeeksPassed'])

    df['WeeksPassed_square'] = df['WeeksPassed'] ** (2)

    

    

    '''

    

    

    '''

    

    

    

    def calculate_height(row): #height can be predictor of FVC -- this estimates the height of patients

        if row['Sex'] == 'Male':

            return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])

        else:

            return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])



    df['Height'] = df.apply(calculate_height, axis=1)

    

    df['HeightWeeks'] = df['WeeksPassed'] * df['Height']

    df['AgeWeeks'] = df['WeeksPassed'] * df['Age']

    

    return df





#feature_engineer(train_df) #just looking
#try to make sklearn estimator for feature engineering

from sklearn.base import BaseEstimator, TransformerMixin



class MyFeatureEngineerer(BaseEstimator, TransformerMixin):

    '''

    this is class so that feature engineering can be done on separate sets

    

    

    To use, call fit on a DataFrame to compute and record values that need to be saved before modification (ie before adding new weeks)

    Examples of values needed to be saved are: FirstFVC, FirstWeek, ...

    Then transform after modifications are done

    

    can just fit_transform if not modifying DataFrame further

    

    '''

    def __init__(self):

        #_ convention indicates that this variable is result of fitting

        pass

    

    def fit(self, X, y = None):

        try:

            self.df_ = feature_engineer(X)

        except AttributeError: #fit should only be called on pandas DataFrame

            raise ValueError('Can only use this estimator on Pandas DataFrame')

        return self #return fitted self for further method calls

    

    def transform(self, X): #honestly fix this up, it's not scalable at all

        '''

        X has been modified with additional weeks

        '''

        

        #check if X has been modified (assume everything same except number of weeks)

        if len(X) != len(self.df_):

            #recompute WeeksPassed if it has been

            drop = X.columns.values 

            df = self.df_.drop(drop, axis = 1).join(self.df_['Patient']) #drop columns already in X, except for patient

            df = X.merge(df, on = 'Patient')

            df['WeeksPassed'] = df['Weeks'] - df['FirstWeek']

            df['WeeksPassed_sqrt'] = np.power(df['WeeksPassed'].abs(), 1/2) * np.sign(df['WeeksPassed'])

            df['WeeksPassed_square'] = df['WeeksPassed'] ** (2)

            df['HeightWeeks'] = df['WeeksPassed'] * df['Height']

            df['AgeWeeks'] = df['WeeksPassed'] * df['Age']



        else:

            df = self.df_ #if not, just return self.df_

        return df

    

'''

from sklearn.utils.estimator_checks import check_estimator

check_estimator(MyFeatureEngineerer())

'''
from sklearn.base import BaseEstimator, TransformerMixin



class ParamMinMaxScaler(BaseEstimator, TransformerMixin):

    '''

    custom minmax scaler where min and max are not based on data,

    but are passed in as parameters

    

    pretty good for percentages

    '''

    def __init__(self, min_val = 0, max_val = 100):

        self.min_val = min_val

        self.max_val = max_val

    

    

    def fit(self, X, y=None): #don't need to fit at all

        return self



    def transform(self, X): #do minmax scaling

        data = (X - self.min_val) / (self.max_val - self.min_val)

        return data

'''

from sklearn.utils.estimator_checks import check_estimator

check_estimator(ParamMinMaxScaler())'''
def transformed_col_names(col_trans):

    '''

    helper function to get column names of dataframe back after column transforming

    because col_trans.get_feature_names() doesn't work very well

    Use this after fitting col_trans

    '''

    import re

    

    new_colnames = []

    for _, t, col in col_trans.transformers_: #loop thru all transformers

        try: #try to get new column names

            temp = t.get_feature_names()

            temp2 = []

            #gotta do some legwork to replace the ugly 'x0', 'x1' prefixes returned by default

            for name in temp: #loop thru feature names returned by t

                match = re.search('x(\d+)+_', name) #look for this ugly bit

                i = int(match.group(1)) #get the feature number

                new_name = col[i] + '_' + name[match.end():] #replace x0 or whatever number with meaningful feature name

                temp2.append(new_name)

            col = temp2

        except AttributeError: #if transformer t does not provide get_feature_names()

            pass #no big deal, just ignore it; we'll extend with original column names

        new_colnames.extend(col) #then append column names to list

        

    return new_colnames
from sklearn_pandas import DataFrameMapper #yes it works!

help(DataFrameMapper) #todo: work this into the pipeline, refactor code so it's less crud

#use this in some way for feature engineering instead of my crap custom class

#todo: determine differences between this and ColumnTransformer
#where all the preprocessing goes on



from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler



#list features to transform



#this isn't scalable honestly

passthru_features = ['Patient', 'FVC']

onehot_features = ['Sex', 'SmokingStatus']

hundred_features = ['Percent', 'Age']

minmax_features = ['FirstFVC', 'FirstWeek', 'WeeksPassed',

                   'WeeksPassed_sqrt', 'WeeksPassed_square', 'Height', 'HeightWeeks',

                  'AgeWeeks']



#question -- should i do a custom scaling of age, percent columns?

#instead of minmax scaling, just divide by 100? will ultimately give similar scale (0 to 1)

#also, maybe do a custom scaling of Weeks as well -- set min to -12, max to 133

#right now it's doing minmax based on whatever's in train



#define the transformers

oh_enc = OneHotEncoder(sparse = False, drop = 'if_binary')

hundred_minmax = ParamMinMaxScaler()

week_minmax = ParamMinMaxScaler(min_val = -12, max_val = 133)

minmax = MinMaxScaler()



#ordered like this to kinda preserve order

col_trans = ColumnTransformer([

                ('original', 'passthrough', passthru_features),

                ('week_minmax', week_minmax, ['Weeks']),

                ('hundred_minmax', minmax, hundred_features),

                ('minmax', minmax, minmax_features),

                ('onehot', oh_enc, onehot_features)

            ], remainder = 'passthrough', sparse_threshold=0)
train_df = MyFeatureEngineerer().fit_transform(train_df)



new_df = col_trans.fit_transform(train_df)



#get the names of the columns back and convert to dataframe

train_df = pd.DataFrame(new_df, columns = transformed_col_names(col_trans))

train_df
from osic_loss_metrics import *

#this is from my utility script OSIC Loss Metrics
import tensorflow as tf

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import VotingRegressor, StackingRegressor



def make_model(): #let's start with a simple tabular model; integrate images later

    '''

    creates and returns a model, but does not fit it

    '''

    

    

    #define the loss function to use

    loss = mloss(0.8) #loss has signature f(y_true, y_pred)

    

    #model = GammaRegressor(alpha = 0)

    

    

    model_est = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], alphas = [0.1,0.3,1,3,10], cv = 6) #problem -- this is using KFold, as opposed to GroupKFold

    model_knn = KNeighborsRegressor(n_neighbors = 13)

    

    model = StackingRegressor(estimators = [('elasticnet', model_est),

                                            ('knn', model_knn)])

    #model = TweedieRegressor(power = 0, alpha = 1, link = 'log', max_iter = 500)

    

    return model
train_df #just look over train_df again
#instantiate and fit model





model = make_model()



drop_features = ['Patient', 'FVC', 'Weeks', 'Percent'] #features to drop from X training data

#Why drop?

#Patient -- id doesn't (or shouldn't, at least) give any info

#FVC -- it's the target value, duh

#Weeks -- redundant, we have WeeksPassed, which should be better anyway

#Percent -- seems like for each patient, Percent is perfectly correlated to FVC over each week

#can't generate Percent on the fly for each week, since that means we could perfectly generate FVC, so we drop Perccent



X_train = train_df.drop(drop_features, axis = 1)

y_train = train_df['FVC']



model.fit(X_train, y_train)

X_train
#randomized might be better due to time constraints

from sklearn.model_selection import RandomizedSearchCV



#LinearRegression is model

'''

params = {

    

            }

hyper_search = RandomizedSearchCV(model, param_distributions=params, n_iter = 20)'''
import matplotlib.pyplot as plt







plt.bar(X_train.columns.values, model.estimators_[0].coef_)

plt.xticks(rotation = 70)



print(model.estimators_[0].alpha_, model.estimators_[0].l1_ratio_)
pred_train = model.predict(X_train)

pred_train
import random

from sklearn.base import clone

#visualize



#see how validation set is predicted

model_copy = clone(model)

i = random.choice(range(NFOLDS)) #choose a random fold

train_index, test_index = list(gkf.split(X_train, y_train, groups))[i]



#for train_index, test_index in gkf.split(X_train, y_train, groups):

p = random.choice(train_df.loc[test_index, 'Patient'].unique()) #get random patient from validation set

print(p)

#print(p, mask.mean())



mask = (train_df['Patient'] == p)



#print(train_df.loc[mask, ['Weeks', 'FVC']])





model_copy.fit(X_train.iloc[train_index, :], y_train.iloc[train_index]) #do training

pred_val = model_copy.predict(X_train[mask]) #do predicting



#print(pred)



ser = pd.Series(pred_val, name = 'FVC_pred', index = train_df[mask].index)

#print(ser)



temp_df = train_df.loc[mask, ['Weeks', 'FVC']].join(ser)

temp_df.plot(x = 'Weeks', y = ['FVC', 'FVC_pred'])

plt.title(p)

#print(temp_df)









#see ground truth
from sklearn.pipeline import Pipeline





#Maybe feature engineer outside of pipeline as compromise -- hmm, but that doesn't work for cross-validation

'''

pipeline = Pipeline([

                ('fe', MyFeatureEngineerer()),

                ('ct', col_trans),

                ('model', make_model())

            ])





pipeline.fit(X_train, y_train)'''
input_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

input_df #preprocess this to turn into test_df
#ideally, can use pipeline we used for preprocessing training data





#problem with infrastructure: can't feature_engineer before adding weeks -12 to +133, because doesn't update WeeksPassed

#but can't feature engineer AFTER adding other weeks, because FirstWeek & FirstFVC would be computed wrong





#this code is honestly scuffed, gotta refactor so it's not crap

eng = MyFeatureEngineerer()

eng.fit(input_df)

#then add weeks -12 to +133 to each patient in test set

input_df2 = input_df.drop(['FVC', 'Weeks'], axis = 1) #this info is stored in FirstFVC and FirstWeek of eng

print(input_df)

all_weeks = pd.DataFrame(np.array(range(-12, 134)), columns = ['Weeks'])

patient_weeks = pd.DataFrame()





#could probably vectorize this

for p in input_df['Patient'].unique(): #this loop creates rows for every week/patient combo

    tdf = all_weeks.copy()

    tdf['Patient'] = p

    patient_weeks = patient_weeks.append(tdf, ignore_index = True)



temp_df = patient_weeks.merge(input_df2, on = 'Patient')



print(temp_df)

print(eng.df_)

new_df = eng.transform(temp_df)

new_df
new_df['FVC'] = 0 #need this for column transforming, can drop afterwards

print(new_df.columns)

new_df = col_trans.transform(new_df) #col_trans already fit on train, don't worry

test_df = pd.DataFrame(new_df, columns = transformed_col_names(col_trans))

test_df
X_test = test_df.drop(drop_features, axis = 1) 

X_test
pred = model.predict(X_test)





pred
sub_df = patient_weeks.join(pd.Series(pred, name = 'FVC'))

sub_df
#visualize results

plt.figure(figsize = (17,10))

for i, (patient, frame) in enumerate(sub_df.groupby('Patient')):

    ax = plt.subplot(2,3, i+1)

    frame[['Weeks', 'FVC']].plot(x = 'Weeks', y = 'FVC', title = patient, ax = ax)
#format the output



sub_df['Patient_Week'] = sub_df['Patient'] + '_' + sub_df['Weeks'].astype(str)

sub_df['Confidence'] = 260 #choose best confidence (best worst case), as determined by cross-val





sub_df
sub_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index = False)