import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/train.csv')

# simply follow Fred's encode part

for c in train.columns:

    if train[c].dtype == "object":

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values))

        train[c] = lbl.transform(list(train[c].values))
y_train = train.y

y = y_train.as_matrix()

train_ID = train.ID

train = train.drop(['y','ID'],axis=1)
## Searching for cases with exactly the same features

S = -np.ones((train.shape[0],train.shape[0]))

for i in range(train.shape[0]):

    temp = train.subtract(train.loc[i],axis=1).as_matrix()

    S[:,i] = np.sum(np.abs(temp),1)

    S[:(i+1),i] = -1

[d1,d2] = np.where(S==0) # find pairs with the same features
s1 = set(d1)

s2 = set(d2)

grps = s2.difference(s1.intersection(s2))

print ("Found total number of groups with the same features: " + str(len(grps)))
err = np.array([])

y_grps = np.array([])

for i in grps:

    inx = np.append(d1[d2==i],i)

    err = np.append(err,y[inx]-np.mean(y[inx]))

    ## this error is defined as the difference y values of cases with exactly the same features

    y_grps = np.append(y_grps,y[inx])
SS_res = np.sum(err*err)

SS_tot = np.sum(np.square(y_grps-np.mean(y_grps)))

r2 = 1-SS_res/SS_tot

print ("Theoretically, r2 will not be larger than " + str(r2))
## add some plots

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(8,6))

plt.scatter(range(len(err)),err)

plt.ylabel('err',fontsize=18)

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(range(len(y_grps)),y_grps)

plt.ylabel('y values in groups',fontsize=18)

plt.show()