import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score,make_scorer



import os

print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
train.head()
test.head()
id_train=train.ID_code

id_test=test.ID_code

y_train=train.target



X_train=train.drop(['ID_code','target'], axis=1) 

X_test=test.drop(['ID_code'], axis=1) 
pipe=Pipeline([('ss',StandardScaler()),

               ('pca',PCA()),

               ('clf',LogisticRegression(random_state=0,n_jobs=-1,solver='lbfgs',class_weight="balanced"))

            ])
C_range=[1e-3,1e-2,1e-1,1,1e1,1e2,1e3]

component_range=np.arange(2,10,1)



param_grid=[{'pca__n_components':component_range,'clf__C':C_range}]



ras=make_scorer(roc_auc_score)

gs=GridSearchCV(estimator=pipe,param_grid=param_grid,scoring=ras,cv=StratifiedKFold(n_splits=5),n_jobs=-1)

gs.fit(X_train,y_train)
print(gs.best_score_)
clf=gs.best_estimator_

clf.fit(X_train,y_train)
probs=clf.predict_proba(X_test)[:,0]

submission=pd.DataFrame({"ID_code":id_test.values,"target":probs})

submission.to_csv("submission.csv",index=False)