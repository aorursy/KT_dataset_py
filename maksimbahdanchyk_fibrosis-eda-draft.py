import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filen|ame))



def make_submission(patient_week,predictions,confidence):

    submission = pd.DataFrame({'Patient_Week':new_test.Patient_Week,'FVC':predictions,'Confidence':confidence})

    submission.to_csv('submission.csv',

                      index = False)

    return submission
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
from tqdm import tqdm

#  also cnsider back time

train_exp = pd.DataFrame()



for patient in tqdm(train.Patient.unique()):

    df = train.loc[train.Patient == patient,:]



    for idx,week,percent in zip(df.index,df.Weeks,df.Percent):

        

        temp_df_pos            = df.loc[idx:,:'SmokingStatus']

        temp_df_pos['Percent'] = percent

        temp_df_pos['Weeks']   = week

        temp_df_pos['target']  = temp_df_pos['FVC']

        temp_df_pos['delta']   = df.loc[idx:,'Weeks'] - df.loc[idx,'Weeks']

        temp_df_pos['FVC']     = temp_df_pos.loc[idx,'FVC']

        

        

        temp_df_neg            = df.loc[:idx,:'SmokingStatus']

        temp_df_neg['Weeks']   = week

        temp_df_neg['Percent'] = percent

        temp_df_neg['target']  = temp_df_neg['FVC']

        temp_df_neg['delta']   = df.loc[:idx,'Weeks'] - df.loc[idx,'Weeks']

        temp_df_neg['FVC']     = temp_df_neg.loc[idx,'FVC']

        

        train_exp = pd.concat([train_exp,temp_df_pos,temp_df_neg],axis = 0)

        train_exp = train_exp[train_exp.delta!=0].drop_duplicates().dropna(axis = 0).reset_index(drop =True)        
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV,GroupKFold

from sklearn.pipeline        import Pipeline, make_pipeline

from sklearn.compose         import ColumnTransformer, make_column_transformer

from sklearn.metrics         import mean_squared_error,mean_absolute_error



from sklearn.preprocessing   import OneHotEncoder,OrdinalEncoder

from sklearn.preprocessing   import MinMaxScaler,StandardScaler,RobustScaler

from sklearn.preprocessing   import FunctionTransformer

from sklearn.ensemble        import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

from sklearn.ensemble        import StackingRegressor

from sklearn.linear_model    import LinearRegression

from sklearn.naive_bayes     import MultinomialNB



from sklearn.svm             import SVR
def confidence(pipe,regressor,X_val,transformer):

    

    val = transformer.transform(X_val)

    predictions = []

    for tree in pipe[regressor]:

        predictions.append(tree.predict(val))



    confidence = np.std(predictions,axis=0)

    return confidence





def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values = False):

    """

    Calculates the modified Laplace Log Likelihood score for this competition.

    """

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)



    if return_values:

        return metric

    else:

        return np.mean(metric)



def pred_ints(model, X, percentile=.95):

    

    err_down = []

    err_up = []

    for x in range(len(X)):

        preds = []

        for pred in model['randomforestregressor'].estimators_:

            preds.append(pred.predict(X[x].reshape(1,-1))[0])

        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))

        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))

        

    return err_down, err_up
def mean_encoding(df, cols, target):

    for c in cols:

        means = df.groupby(c)[target].mean()

        df[c].map(means)

    return df
X = train_exp.drop(['Patient','target'],axis = 1) 

y = train_exp['target']



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.1, 

                                               random_state = 42, 

                                               shuffle = True)



transformer = make_column_transformer( 

    (MinMaxScaler() , ['Age','Percent','delta','FVC','Weeks']), 

    (OneHotEncoder(),['Sex','SmokingStatus']), 

    remainder = 'passthrough' )



X_train = transformer.fit_transform(X_train) 

X_val = transformer.transform(X_val)
import tensorflow as tf 

import keras



def tilted_loss(q,y,f): 

    e = (y-f) 

    return keras.backend.mean(keras.backend.maximum(q*e, (q-1)*e), axis=-1)



model = tf.keras.models.Sequential([



tf.keras.layers.Dense(128,activation = 'relu'),

tf.keras.layers.Dense(128,activation = 'relu'),

tf.keras.layers.Dense(64,activation = 'relu'),

tf.keras.layers.Dense(1)

])



quntiles = [0.25, 0.5, 0.75]



y_val_predictions = [] 

y_train_predictions = [] 

models = []



for q in quntiles: 

    print(q,' quantile')

    model.compile(loss = lambda y_true,y_pred: tilted_loss(q,y_true,y_pred), 

                  optimizer= tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07,decay = 0.01,amsgrad=False), 

                  metrics = [tf.keras.metrics.RootMeanSquaredError()])



    model.fit(X_train, y_train, 

              epochs=50, 

              batch_size=32, 

              steps_per_epoch = X_train.shape[0]//32,

              validation_data = (X_val,y_val),

              verbose=1)



    y_val_pred_temp = model.predict(X_val)

    y_train_pred_temp = model.predict(X_train)





    y_val_predictions.append(y_val_pred_temp)

    y_train_predictions.append(y_train_pred_temp)

    

print('Train RMSE score: ',np.sqrt(mean_squared_error(y_train, y_train_predictions[1]))) 

print('Val RMSE score: ',np.sqrt(mean_squared_error(y_val, y_val_predictions[1])))



confidence_train = (y_train_predictions[2] - y_train_predictions[0])

confidence_val = (y_val_predictions[2] - y_val_predictions[0])



print('Train OSCI score: ',laplace_log_likelihood(y_train, y_train_predictions[1].reshape(-1,), confidence_train.reshape(-1,), return_values = False)) 

print('Val OSCI score: ',laplace_log_likelihood(y_val, y_val_predictions[1].reshape(-1,), confidence_val.reshape(-1,), return_values = False))
new_test = pd.DataFrame()

for i in np.arange(-12,134,1):

    temp_df = test.copy()

    temp_df['stamps'] = i

    temp_df['delta'] = temp_df['Weeks'] +  temp_df['stamps']

    new_test = pd.concat([new_test,temp_df])



new_test.reset_index(drop=True,inplace = True)

new_test['Patient_Week'] = new_test['Patient'] + '_' + new_test['stamps'].astype(str)





X_test = new_test.drop(['Patient','stamps','Patient_Week'],axis = 1)





X_test = transformer.transform(X_test)





new_test = pd.DataFrame()

for i in np.arange(-12,134,1):

    temp_df = test.copy()

    temp_df['stamps'] = i

    temp_df['delta'] = temp_df['Weeks'] +  temp_df['stamps']

    new_test = pd.concat([new_test,temp_df])



new_test.reset_index(drop=True,inplace = True)

new_test['Patient_Week'] = new_test['Patient'] + '_' + new_test['stamps'].astype(str)





X_test = new_test.drop(['Patient','stamps','Patient_Week'],axis = 1)





X_test = transformer.transform(X_test)



predictions = []



model = tf.keras.models.Sequential([



tf.keras.layers.Dense(256,activation = 'relu'),

tf.keras.layers.Dense(128,activation = 'relu'),

tf.keras.layers.Dense(32,activation = 'relu'),

tf.keras.layers.Dense(1)

])



quntiles = [0.25, 0.5, 0.75]





for q in quntiles:

    print(q,' quantile')

    model.compile(loss = lambda y_true,y_pred: tilted_loss(q,y_true,y_pred),

                 optimizer= tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07,decay = 0.01,amsgrad=False), 

                 metrics = [tf.keras.metrics.RootMeanSquaredError()])





    model.fit(X_train, y_train, 

              epochs=50, 

              batch_size=32, 

              steps_per_epoch = X_train.shape[0]//32,

              validation_data = (X_val,y_val),

              verbose=0)

    

    pred_temp = model.predict(X_test)    

    predictions.append(pred_temp)





print('Done')

confidence = abs(predictions[2] - predictions[0])



make_submission(new_test.Patient_Week,predictions[1].reshape(-1,),confidence.reshape(-1,))
plt.hist(confidence)