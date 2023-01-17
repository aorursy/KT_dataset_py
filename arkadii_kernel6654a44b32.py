import pandas as pd

io_params = { 'parse_dates' : ['epoch'] }

data = pd.read_csv('/kaggle/input/sputnik/train.csv', **io_params)
train = data[data.type == 'train'].drop('type',axis = 1)

test = data[data.type == 'test'].drop('type',axis = 1)
class ARModel:

    

    def __init__(self, p, model):

        self.p = p

        self.model = model

    

    def fit(self, path):

        

        n = path.strides[0]

        x_trick = np.lib.stride_tricks.as_strided(path, shape=(path.shape[0], self.p), strides=(n, n))

        if x_trick.shape[0] > self.p: #For preserving exceptions

            X = x_trick[:-self.p]

            Y = path[self.p:]

        else:

            X = x_trick

            Y = path

        # Save the most recent history for later usage

        # Conceptually history is a list, but  give it an extra dimension because sklearn eats matrices

        self.history = path[-self.p:].reshape(1, -1)

        

        self.model.fit(X, Y)

        

    def forecast(self, steps):

        

        history = self.history.copy()

        predictions = np.empty(steps)

        

        for i in range(steps):

            

            y_pred = self.model.predict(history)[0]    

            predictions[i] = y_pred

            

            # Shift forward (faster than np.roll)

            history[0, :-1] = history[0, 1:]

            history[0, -1] = y_pred

            

        return predictions
import sklearn

from sklearn import compose

from sklearn import linear_model

from sklearn import pipeline

from sklearn import preprocessing

import tqdm



preds = []





class Pipeline:

    """Comfortable implementation"""

    

    def __init__(self, *steps):

        self.steps = steps

    

    def fit(self, X, y):

        for transformer in self.steps[:-1]:

            X = transformer.fit_transform(X, y)

        try:

            self.steps[-1].fit(X, y)

        except:

            print(X,y)

            return y

        return self

    

    def predict(self, X):

        for transformer in self.steps[:-1]:

            X = transformer.transform(X)

        return self.steps[-1].predict(X)





class StandardScaler(preprocessing.StandardScaler):

    """Comfortable implementation for sat prediction."""

    

    def transform(self, X):

        if (X.shape[1] < self.mean_.shape[0]):

            return (X - self.mean_[:X.shape[1]]) / self.var_[:X.shape[1]] ** .5

        return (X - self.mean_) / self.var_ ** .5 #Because

    

    

class LinearRegression(linear_model.LinearRegression):

    """Comfortable implementation for sat prediction"""

    

    def predict(self, X):

        return np.dot(X, self.coef_[:X.shape[1]]) + self.intercept_



    

model = ARModel(

    p=48, # Chosen experementally

    model=Pipeline(

        StandardScaler(),

        LinearRegression()

    )

)
import numpy as np

together = pd.concat((train, test), sort=False)

together['is_train'] = data['x'].notnull()

together = together.sort_values(['sat_id', 'epoch']) 
'''Predict separatelly x,y,z - score will be better'''

'''Make sure that these predistions are shifted'''

preds = []



train_sats = together.query('is_train')

test_sats = together.query('not is_train')



for sat in tqdm.tqdm(test_sats['sat_id'].unique(), position=0):



    train = train_sats.query('sat_id == @sat')

    test = test_sats.query('sat_id == @sat')



    for var in ('x', 'y', 'z'):



        model.fit(train[var].to_numpy())

        pred = model.forecast(len(test))



        preds.append(pd.DataFrame({

            'sat_id': test['sat_id'],

            'id': test['id'],

            'epoch': test['epoch'],

            'y_pred': pred,

            'variable': var

        }))

        

preds = pd.concat(preds)

preds.head()
'''Now group predictions by target  varriables''' 

preds = preds.groupby('sat_id').apply(lambda g: g.pivot_table(index=['id', 'epoch'], columns='variable', values='y_pred')).reset_index()

preds.head()
import datetime as  dt

'''Shift predictions back'''

correct_preds = []



cols_to_shift = ['x', 'y', 'z']



for _, g in tqdm.tqdm(preds.groupby('sat_id'), position=0):

    

    g = g.copy()

    dups = g[g['epoch'].diff() < dt.timedelta(seconds=60)].index

    

    for i in dups:

        g.loc[i:, cols_to_shift] = g.loc[i:, cols_to_shift].shift()

    g[cols_to_shift] = g[cols_to_shift].ffill()

    

    correct_preds.append(g)

    

correct_preds = pd.concat(correct_preds)
req_ind = ['x','y','z']

req_sim =['x_sim','y_sim','z_sim']

errors = np.linalg.norm(correct_preds[req_ind].values - data.query('type == \'test\'')[req_sim].values,axis = 1)
sub = pd.DataFrame([])

sub['id'] = data.query('type == \'test\'').id.values

sub['error'] = errors

sub.to_csv('sumis.csv',index = False)