import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots



Train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
column_nan = np.zeros(len(test))

column_nan.fill(np.nan)

test.insert(1, "Survived", column_nan)

train = pd.concat([Train,test])
# Name

train['Title'] = train.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)

desig = train.iloc[:,12].values

X_desig = np.zeros(len(desig))

for i in range(0, len(desig)):

    str = desig[i]

    if(str == 'Mr'):

        X_desig[i] = 1 

    elif(str == 'Mrs'):

        X_desig[i] = 2

    elif(str == 'Ms'):

        X_desig[i] = 3

    elif(str == 'Master'):

        X_desig[i] = 4

    else:

        X_desig[i] = 5



# Removing NaN values from Age

X_age = train.iloc[:, 5].values

age_mean = np.nanmean(X_age)

age_std = np.nanstd(X_age)

age_nan_index = np.argwhere(np.isnan(X_age))

X_age[age_nan_index] = np.random.uniform(low=age_mean - age_std, high=age_mean + age_std, size=age_nan_index.shape)



# Age grouping

age_grouped = np.zeros(X_age.shape)

for i in range(0,len(X_age)):

    ag = X_age[i]

    if ag >64:

        age_grouped[i] = 1

    elif ag>48:

        age_grouped[i] = 2

    elif ag>32:

        age_grouped[i] = 3

    elif ag>16:

        age_grouped[i] = 4

    else:

        age_grouped[i] = 5    

X_age = age_grouped



# Removing NaN values from fare

X_fare = train.iloc[:, 9].values

fare_mean = np.nanmean(X_fare)

fare_std = np.nanstd(X_fare)

fare_nan_index = np.argwhere(np.isnan(X_fare))

X_fare[fare_nan_index] = np.random.uniform(low = fare_mean - fare_std, high = fare_mean + fare_std, size = fare_nan_index.shape)



# Removing NaN values from Embarked

X_em = train.iloc[:,11].values

X_emb = np.zeros(len(X_em))

for i in range(0,len(X_em)):

    ch = X_em[i]

    if(ch == 'C'):

        X_em[i]  = 1

    elif(ch == 'S'):

        X_emb[i] = 2

    elif(ch == 'Q'):

        X_emb[i] = 3

    else:

        X_emb[i] = np.nan

emb_mean = np.nanmean(X_emb)

emb_std = np.nanstd(X_emb)

emb_nan_index = np.argwhere(np.isnan(X_emb))

X_emb[emb_nan_index] = np.random.uniform(low = emb_mean - emb_std, high = emb_mean + emb_std, size = emb_nan_index.shape)



# Grouping Fare 

fare = train.iloc[:,9].values

X_fare = np.zeros(len(fare))

for i in range(0,len(X_fare)):

    if(fare[i]>85):

        X_fare[i] = 1

    elif(fare[i]>50):

        X_fare[i] = 2

    elif(fare[i]>20):

        X_fare[i] = 3

    elif(fare[i]>5):

        X_fare[i] = 4

    elif(fare[i]>=-1):

        X_fare[i] = 5



# Grouping sex

sex = train.iloc[:, 4].values

X_sex = np.zeros(len(sex))

for i in range(0,len(X_sex)):

    if(sex[i] == 'male'):

        X_sex[i] = 1

    else:

        X_sex[i] = 2

# Just setting up numpy array for Pclass

X_pclass =  train.iloc[:,2].values

# Total family members ie. SibSp + Parch

sibsp = train.iloc[:,6].values

parch= train.iloc[:,7].values

X_fam = sibsp + parch + 1
X = np.c_[X_desig,X_pclass,X_sex,X_age,X_fam,X_fare,X_emb]

X2 = X**2

X = np.c_[X,X2]


X_train = X[0:891,:]

X_test = X[891:len(X),:]

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X. transform(X_test)

y_train = Train.iloc[:, 1].values
# Importing Classifier Modules

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier


clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred_decision_tree = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_decision_tree

    })



submission.to_csv('submission.csv', index=False)