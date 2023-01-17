import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/train.csv')
label = pd.DataFrame(train.label)
train = train[train.columns.drop('label')]
test = pd.read_csv('../input/test.csv')
train.head()
label.label.value_counts(sort=False)
#how to put the pixels into an image
from PIL import Image

i = 0
im = Image.new("RGB",(28, 28))
pix = im.load()
for y in range(28):
    for x in range(28):
        pix[x,y] = (train.iloc[1,i],0,0)
        i = i + 1

im
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics

y = label
X = train

error_rate = []

cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True)

for train_index, test_index in cv:
    model = RandomForestClassifier(n_estimators=75, max_depth=15, min_samples_leaf=6).fit(X.iloc[train_index], y.iloc[train_index])
    df = y.iloc[test_index]
    df['predict'] = model.predict(X.iloc[test_index])
    error_rate.append(float(len(df[df.label != df.predict]))/float(len(df)))
    
    
print("Error Rate:", np.mean(error_rate))
test['label'] = pd.DataFrame(model.predict(test.fillna(0)))
pd.DataFrame(test.label)
