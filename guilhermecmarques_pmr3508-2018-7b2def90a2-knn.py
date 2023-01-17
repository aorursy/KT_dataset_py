%matplotlib inline

import matplotlib
import pandas as pd
import numpy as np

# remove error in pd manipulation
pd.options.mode.chained_assignment = None

# import dataset as a dataframe
pov = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
pov = pov.replace(np.nan,'0', regex=True)
pov = pov.replace("no",'0', regex=False)
pov = pov.replace("yes",'0', regex=False)
# view dataframe
pov.head(20)
# import test data
testPov = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testPov = testPov.replace(np.nan,'0', regex=True)
testPov = testPov.replace("no",'0', regex=False)
testPov = testPov.replace("yes",'0', regex=False)

# dataframe manipulation

# convert sex into binary
testPov["edjefe"] = testPov["edjefe"].str.replace("no", "0", regex = False)
pov["edjefe"] = pov["edjefe"].str.replace("no", "0", regex = False)

# select used categories
Xpov = pov[["v2a1","rooms","v18q","r4t1","r4t3","tamhog","escolari","cielorazo","abastaguadentro","abastaguafuera","abastaguano","public","sanitario2","sanitario3","energcocinar2","energcocinar3","elimbasu1","elimbasu3","epared1","epared2","etecho1","etecho2","eviv1","eviv2","male","parentesco1","parentesco3","hogar_nin","hogar_adul","hogar_mayor","hogar_total","edjefe","edjefa","meaneduc","instlevel5","instlevel7","instlevel8","instlevel9","bedrooms","overcrowding","tipovivi1","tipovivi2","tipovivi3","tipovivi4","computer","television","mobilephone","lugar1","lugar2","lugar3","lugar4","lugar5","area1"]]
XtestPov = testPov[["v2a1","rooms","v18q","r4t1","r4t3","tamhog","escolari","cielorazo","abastaguadentro","abastaguafuera","abastaguano","public","sanitario2","sanitario3","energcocinar2","energcocinar3","elimbasu1","elimbasu3","epared1","epared2","etecho1","etecho2","eviv1","eviv2","male","parentesco1","parentesco3","hogar_nin","hogar_adul","hogar_mayor","hogar_total","edjefe","edjefa","meaneduc","instlevel5","instlevel7","instlevel8","instlevel9","bedrooms","overcrowding","tipovivi1","tipovivi2","tipovivi3","tipovivi4","computer","television","mobilephone","lugar1","lugar2","lugar3","lugar4","lugar5","area1"]]
# interest variable
Ypov = pov.Target
# import classifier and X-validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# test k = 1-200
h = 0
b = 0
# for k in range(1,200):
#     # the classifier   
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # results in 10 fold validation
#     scores = cross_val_score(knn, Xpov, Ypov, cv=10)
#     # mean of x-validations
#     mean = np.mean(scores)
#     if mean > h:
#         b = k
#         h = mean
knn = KNeighborsClassifier(n_neighbors=20)
scores = cross_val_score(knn, Xpov, Ypov, cv=10)
mean = np.mean(scores)
print(mean)
# estimates train data against the interest variable
knn.fit(Xpov,Ypov)
# predicts test data Target var
YtestPred = knn.predict(XtestPov)
# put the predictions in a dataframe
preds = pd.DataFrame(testPov.Id)
preds["Target"] = YtestPred
preds
# save predictions
preds.to_csv("prediction.csv", index=False)
