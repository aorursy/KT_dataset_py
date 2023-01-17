import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



random_seed = 91

np.random.seed(random_seed)
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")



X = train.sample(frac=1).copy()

y = X.pop("eyeDetection")
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score



train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=.2, random_state=random_seed)



cls_gb = GradientBoostingClassifier(random_state=random_seed)

cls_gb.fit(train_X,train_y)



predictions_train = cls_gb.predict(train_X)

predictions_val = cls_gb.predict(val_X)

print("%f %f"%(accuracy_score(predictions_train, train_y),accuracy_score(predictions_val, val_y)))
help(GradientBoostingClassifier)
#Note: the ranges used to be bigger, but I narrowed them down eventually

for maxdepth in range(10,14,1):

    for nestimators in range(100,300,50):

        for sub_sample in [0.75]:

            cls_gb = GradientBoostingClassifier(n_estimators=nestimators,max_depth=maxdepth,max_features=0.5,subsample=sub_sample,random_state=random_seed)

            cls_gb.fit(train_X,train_y)

            predictions_train = cls_gb.predict(train_X)

            predictions_val = cls_gb.predict(val_X)

            print("%i %i %f %f %f"%(maxdepth,nestimators,sub_sample,accuracy_score(predictions_train, train_y),accuracy_score(predictions_val, val_y)))
cls_gb = GradientBoostingClassifier(n_estimators=250,max_depth=10,max_features=0.5,subsample=0.75,random_state=random_seed)



cls_gb.fit(X,y)



#code for submission

index_col = test.pop("index")

predictions_test = cls_gb.predict_proba(test)

out_tmp = pd.DataFrame()

out_tmp["index"] = index_col

out_tmp["eyeDetection"] = predictions_test[:,1]

print(out_tmp.head())

out_tmp.to_csv("submission.csv",index=False)
from sklearn.model_selection import cross_val_predict



predictions_val = cross_val_predict(cls, X, y, cv=3)

print("%f"%(accuracy_score(predictions_val, y)))
from sklearn.model_selection import GridSearchCV



parameters = {'criterion': ["gini", "entropy"], 'max_depth':range(2,20,2)}

cls_G = GridSearchCV(cls, parameters,cv=3)

cls_G.fit(X,y)

print(cls_G.best_estimator_)

print(cls_G.best_score_)
predictions = cls_G.best_estimator_.predict(X)

print("%f"%(accuracy_score(predictions, y)))