import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
train.columns
import matplotlib.pyplot as plt

import seaborn as sns

#TODO

sns.regplot(x = "AveragePrice",y = "4046",data = train)

plt.show()
import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import xgboost

%matplotlib inline
tfeatures = test[['id', 'Total Volume', '4046', '4225', '4770', 'Total Bags','Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year']]

#target = train['AveragePrice']
features = train[['id', 'Total Volume', '4046', '4225', '4770', 'Total Bags','Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year']]

target = train['AveragePrice']
def rmsle(evaluator,X,real):

    sum = 0.0

    predicted = evaluator.predict(X)

    print("Number predicted less than 0: {}".format(np.where(predicted < 0)[0].shape))



    predicted[predicted < 0] = 0

    for x in range(len(predicted)):

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p-r)**2

    return (sum/len(predicted))**0.5
params = {'n_estimators': 100, 'seed':0, 'colsample_bytree': 1, 

             'max_depth': 7, 'min_child_weight': 1,'learning_rate': 0.1, 'subsample': 0.8}
cv_params = {'max_depth': [5,7,10], 'min_child_weight': [1,3,5]}

ind_params = {'learning_rate': 0.08, 'n_estimators': 100, 'seed':0, 'subsample': 0.75, 'colsample_bytree': 1}

optimized_GBM = GridSearchCV(xgboost.XGBRegressor(**ind_params), 

                            cv_params,scoring = rmsle, cv =4) 

optimized_GBM.fit(features, np.ravel(target))
#print(optimized_GBM.grid_scores_)


reg = xgboost.XGBRegressor(n_estimators=200, seed=0,learning_rate=0.2, subsample=0.8,

                           colsample_bytree=1, max_depth=10,min_child_weight= 2)



cv = ShuffleSplit(n_splits=6, test_size=0.1, random_state=0)

print(cross_val_score(reg, features, np.ravel(target), cv=cv,scoring=rmsle))

reg.fit(features,target)
pred = reg.predict(tfeatures)

print(np.where(pred < 0)[0].shape)

pred[pred < 0] = 0

#test.apply(lambda col: col.drop_duplicates().reset_index(drop=True))

test['AveragePrice']=pred.astype(float)

out = test[['id','AveragePrice']]

out['AveragePrice'].isnull().values.any()

out.to_csv('pred_xgboost2.csv',index=False)
import pickle

pickle.dump(reg, open('xgb_model_2.sav','wb'),protocol=2)