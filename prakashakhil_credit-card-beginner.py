from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.qda import QDA

from sklearn.cross_validation import cross_val_score

from sklearn import preprocessing





import pandas as pd

import numpy as np





all_data= pd.read_csv("../input/creditcard.csv")



all_data = all_data.drop('Time', 1)

all_data = all_data.drop('V23', 1)

all_data = all_data.drop('V13', 1)

all_data = all_data.drop('V28', 1)

all_data = all_data.drop('V25', 1)

all_data = all_data.drop('V22', 1)

all_data = all_data.drop('Amount', 1)

all_data = all_data.drop('V27', 1)

all_data = all_data.drop('V24', 1)

all_data = all_data.drop('V8', 1)

all_data = all_data.drop('V15', 1)

all_data = all_data.drop('V5', 1)

all_data = all_data.drop('V2', 1)

all_data = all_data.drop('V20', 1)

all_data = all_data.drop('V19', 1)

all_data = all_data.drop('V21', 1)

all_data = all_data.drop('V1', 1)

all_data = all_data.drop('V6', 1)

all_data = all_data.drop('V7', 1)

all_data = all_data.drop('V26', 1)

   

label =all_data["Class"].values

all_data = all_data.drop('Class', 1)



    

rf=RandomForestClassifier()

scores = cross_val_score(rf, all_data.values,label, cv=5)

print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))
