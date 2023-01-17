import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
import sys
import os
import matplotlib
from PIL import Image

#%matplotlib inline

def main ():
    train = pd.read_csv ('../input/train.csv')
    label = pd.DataFrame (train.label)
    train = train[train.columns.drop ('label')]
    test = pd.read_csv ('../input/test.csv')
    print (label)
    print(train.head())
    print (label.label.value_counts (sort=False))
    #print (type (label))
    i = 0
    im = Image.new ("RGB", (28, 28))
    pix = im.load ()
    for y in range (28):
        for x in range (28):
            pix[x,y] = (train.iloc[1,i], 0, 0)
            i += 1
    im.show()
    #im.save ("./im.jpg")

    y = label
    X = train

    error_rate = []
    cv = cross_validation.KFold (len(X), n_folds=5, shuffle=True)

    for train_index, test_index in cv:
        model = RandomForestClassifier (n_estimators=75, max_depth=15, min_samples_leaf=6)
        model.fit (X.iloc[train_index], y.iloc[train_index])
        df = y.iloc[test_index]
        df['predict'] = model.predict (X.iloc[test_index])
        error_rate.append (float (len (df[df.label != df.predict]))/ float(len(df)))
    print ("Error Rate:", np.mean (error_rate))


if __name__ == '__main__':
    main()
