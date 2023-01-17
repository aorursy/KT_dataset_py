#Load Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
"""
Note, Kaggle has a bug, it seems you can not access the data from kernels for in-class competitions.
More info in: https://www.kaggle.com/product-feedback/64997#latest-382080
"""
#Read data
train=pd.read_csv("../input/kaggle_train.csv")
#test=pd.read_csv("../input/kaggle_test.csv")
#sub=pd.read_csv("../input/kaggle_sample.csv")
#Remove Ids and save target
del train['id']
#del test['id']

target=train['is_pump_broken']
del train['is_pump_broken']
#Find easy-to-use features (numeric only)
easy_features=list(train.select_dtypes(include=[np.number]).columns)
#Train the model
clf = RandomForestClassifier(n_estimators=160, n_jobs=-1, random_state=0)
clf.fit(train[easy_features],target)
#Save submission file
#sub['is_pump_broken']=clf.predict_proba(test[easy_features])[:,1]
#sub.to_csv("my_first_submission_rfx160.csv",index=False)
