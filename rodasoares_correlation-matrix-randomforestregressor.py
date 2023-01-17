import numpy as np

import pandas as pd
train = pd.read_csv("../input/train.csv", index_col='id') 

test = pd.read_csv("../input/test.csv", index_col='id') 
corr_m=train.corr()

corr_m['target'].sort_values(ascending=False)
train_y=train['target']

train_x=train.drop('target', axis=1)
attributes=['33','65','24','183','199','194','189','80','73','295','91','117','217']

train_x=train_x[attributes]

test=test[attributes]
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=7),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='roc_auc', random_state=42)

rnd_search.fit(train_x,train_y)
#Save results

prediction = rnd_search.predict(test)

final=pd.DataFrame(prediction,columns=['target'])

final['id']=test.index

final=final[['id','target']]

final.to_csv("submission.csv", index=False)