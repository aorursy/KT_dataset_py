# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
d = pd.read_csv("../input/diabetes1.csv")
d.head()
d.columns
colnames = list(d.columns)
predictors = colnames[:9]
target = colnames[9]
X = d[predictors]
Y = d[target]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
np.shape(d)
# Attributes that comes along with RandomForest function

rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 9 here.
rf.n_outputs_ # Number of outputs when fit performed
rf.oob_score_  
rf.predict(X)
d['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Diabetic']
d[cols].head()
d["Diabetic"]
from sklearn.metrics import confusion_matrix
# Confusion matrix

confusion_matrix(d['Diabetic'],d['rf_pred']) 
pd.crosstab(d['Diabetic'],d['rf_pred'])
print("Accuracy",(9957+5000)/(9957+5000+0+3)*100)
d["rf_pred"]