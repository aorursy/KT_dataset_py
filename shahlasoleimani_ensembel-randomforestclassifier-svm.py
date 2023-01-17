import csv

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



data = pd.read_csv("../input/iust-nba-rookies/train.csv")
test_data = pd.read_csv("../input/iust-nba-rookies/test.csv")
submission =pd.read_csv("../input/submission/submission.csv")
labels = data.TARGET_5Yrs
data = data.drop('Name', 1)
data = data.drop('TARGET_5Yrs', 1)
data = data.drop('PlayerID', 1)

test_data = test_data.drop('Name', 1)
test_data = test_data.drop('PlayerID', 1)

data = data.fillna(data.mean())

#data = data.where(pd.notna(data), data.mean(), axis='columns')
#data = np.array(data).astype("float")

print(np.where(np.isnan(data)))


pca = PCA(n_components=10)
pca.fit(data)
data = pca.transform(data)


pca = PCA(n_components=10)
pca.fit(test_data)
test_data = pca.transform(test_data)


clf = RandomForestClassifier(max_depth=9, random_state=2)
clf.fit(data, labels)

#print(clf.feature_importances_)
print ('score', clf.score(data, labels) )

predict = clf.predict(test_data)

print  ('pred label', predict )

with open('submission.csv', 'w') as csvfile:
    fieldnames = ['PlayerID', 'TARGET_5Yrs']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    count = 900

    for i in range(predict.size):
        count+=1
        writer.writerow({'PlayerID': count, 'TARGET_5Yrs': int(predict[i])})
    print(submission)
   # submission.to_csv("submission.csv", index=False)
    
clf = []
clf = svm.SVC(kernel='linear') 
clf.fit(data,labels )  #train
#print ('score', clf.score(x1, y1) )
print ('score', clf.score(data, labels) )
predict = clf.predict(test_data)
print  ('pred label', predict )
