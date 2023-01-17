import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble

train = pd.read_csv("../input/train.csv")
target = train['target'].values

test = pd.read_csv("../input/test.csv")
id_test = test['ID'].values
test['target'] = -1

data = train.append(test)

train.shape, test.shape, data.shape
for data_name, data_series in data.iteritems():
    if data_series.dtype == np.object:
        #for objects: factorize
        data[data_name], tmp_indexer = pd.factorize(data[data_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(data[data_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            data.loc[data_series.isnull(), data_name] = -999 

campi = ['v3', 'v24', 'v30', 'v31', 'v38', 'v47', 'v52', 'v62', 'v66', 
         'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125', 'v129',
         'v50', 'v10', 'v14', 'v34', 'v114', 'v21', 'v12']

'''
ci sono    : 'v3' ,'v24','v30','v31','v38','v47','v52','v62','v66'
            ,'v71','v72','v74','v75','v79','v91','v107','v110','v112','v113','v125','v129']

non ci sono: 'v10' ,'v12','v14','v21','v22','v34','v40','v50','v56','v114'

corr: 50, 10, 14, 34, 114, 21, 12

'''

train = data[data["target"] >=0]
test  = data[data["target"] < 0]

train = train[campi]
test = test[campi]

train.shape, test.shape
extc = ExtraTreesClassifier(n_estimators=1200,
                            max_features= 28,
                            criterion= 'entropy',
                            min_samples_split= 2,
                            #max_depth= 100, 
                            min_samples_leaf= 2, 
                            n_jobs = -1)    

extc.fit(train, target) 

y_pred = extc.predict_proba(test)
pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('trees.csv',index=False)
