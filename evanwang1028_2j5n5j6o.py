# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json(open("../input/train.json", "r"))

test = pd.read_json(open("../input/test.json", "r"))
test_id = test.listing_id
badcols = [ i for i in train.columns if 'id' in i]

badcols = badcols+['photos','description','display_address','street_address']
train = train.drop(train[badcols],axis=1)

test = test.drop(test[badcols],axis=1)
train.head()


def date_clean(dat):

    dat["created"] = pd.to_datetime(dat["created"])

    dat["year"] = dat["created"].dt.year

    dat["month"] = dat["created"].dt.month

    dat["day"] = dat["created"].dt.day

    dat["hour"] = dat["created"].dt.hour

    del dat['created']

    return(dat)



train=date_clean(train)

test=date_clean(test)
def n_feat(dat):

    dat['features'] =     dat.features.apply(lambda x: len(x))

    return(dat)



train = n_feat(train)

test = n_feat(test)
trainY = train['interest_level']



del train['interest_level']



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500)

clf.fit(train, trainY)
result = clf.predict_proba(test)
labels2idx = {label: i for i, label in enumerate(clf.classes_)}

labels2idx
sub = pd.DataFrame()

sub["listing_id"] = test_id



for label in ["high", "medium", "low"]:

    sub[label] = result[:, labels2idx[label]]

sub.head()
sub.to_csv("submission_rf.csv", index=False)