# https://www.kaggle.com/the1owl/two-sigma-financial-modeling/initial-script



import kagglegym

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression

import math

import gc



def _reward(y_true, y_fit):

    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)

    R = np.sign(R2) * math.sqrt(abs(R2))

    return(R)



env = kagglegym.make()

o = env.reset()

excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]

col = [c for c in o.train.columns if c not in excl]



pd.options.mode.chained_assignment = None  # default='warn'

train = o.train[col]

d_mean= train.median(axis=0)

n = train.isnull().sum(axis=1)

for c in train.columns:

    train[c + '_nan_'] = pd.isnull(train[c])

    d_mean[c + '_nan_'] = 0

train = train.fillna(d_mean)

train['znull'] = n

n = []



rfr = ExtraTreesRegressor(n_estimators=70, max_depth=4, n_jobs=-1, random_state=1, verbose=0)

model1 = rfr.fit(train, o.train['y'])



#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189

low_y_cut = -0.075

high_y_cut = 0.075

y_is_above_cut = (o.train.y > high_y_cut)

y_is_below_cut = (o.train.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

model2 = LinearRegression(n_jobs=-1)

model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), o.train.loc[y_is_within_cut, 'y'])

train = []

gc.collect()
# setup

models_to_choose_from=[model1,model2]

models_names=["ExtraTreesRegressor", "LinearRegression"]

current_model=models_to_choose_from[0]

alternative_model= models_to_choose_from[1]



o = env.reset()

i = 0; reward_=[]

while True:

    test = o.features[col]

    n = test.isnull().sum(axis=1)

    for c in test.columns:

        test[c + '_nan_'] = pd.isnull(test[c])

    test = test.fillna(d_mean)

    test['znull'] = n

    pred = o.target

    test2 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)

    

    if models_names[models_to_choose_from.index(current_model)]=="ExtraTreesRegressor":

        pred['y'] = current_model.predict(test).clip(low_y_cut, high_y_cut)

        alternative_pred=alternative_model.predict(test2).clip(low_y_cut, high_y_cut)

    else:

        pred['y'] = current_model.predict(test2).clip(low_y_cut, high_y_cut)

        alternative_pred=alternative_model.predict(test).clip(low_y_cut, high_y_cut)



        

    r_diff_pred= _reward(pred['y'], alternative_pred)

    #pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) + model2.predict(test2).clip(low_y_cut, high_y_cut))/2

    #pred['y'] = pred.apply(get_weighted_y, axis = 1)

    o, reward, done, info = env.step(pred[['id','y']])

    

    reward_.append(reward)

    #if i % 100 == 0:

    print(reward, np.mean(np.array(reward_)))

    

    try:

        if reward < -0.2 :

            print("switching models")

            alternative_model_new_index = models_to_choose_from.index(current_model)

            current_model_new_index = models_to_choose_from.index(alternative_model)

            current_model=models_to_choose_from[current_model_new_index]

            alternative_model= models_to_choose_from[alternative_model_new_index]

            print("current model is {}".format(models_names[current_model_new_index]))

    except: pass



    gc.collect()

    

    i += 1

    if done:

        print("el fin ...", info["public_score"])

        break