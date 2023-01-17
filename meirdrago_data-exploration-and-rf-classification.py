import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np

import pandas as pd

sns.set()



data = pd.read_csv("../input/adult.csv")

# clean the ?

data = data.ix[(data.workclass!="?") 

               & (data.occupation!="?") & (data['native.country']!="?") & (data.education!="?")]

data.head()
# plot distinct features except  fnlwgt

data.ix[:,data.columns!='fnlwgt'].apply(pd.Series.nunique).plot(kind='barh')
# plt fnlwgt distribution

sns.kdeplot(data.fnlwgt,shade=True)
# confusion matrix

nominals = ["workclass","education","marital.status","occupation","relationship","race","sex","income"]

us_data = data[data['native.country']=="United-States"]

us_data = us_data.ix[:,nominals]

for c in us_data.columns:

    us_data[c] = pd.factorize(np.array(us_data[c]))[0]

plt.xticks(rotation=45)

sns.heatmap(us_data.corr())    
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split , cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



# replace income with country name

nominals = ["native.country" if i=="income" else i for i in nominals]

X = np.zeros((len(data),len(nominals)))

for i,f in enumerate(nominals):

    X.T[i] = LabelEncoder().fit_transform(data[f])

y = np.where(data.income == '<=50K', 0, 1)

    

X_train, X_test, y_train, y_test = train_test_split(

    X , y, test_size=0.4, random_state=111)



# random forest

rfc = RandomForestClassifier(n_jobs=-1)

rfc.fit(X_train,y_train)



y_pos = np.arange(len(nominals))

plt.bar(y_pos , rfc.feature_importances_)

plt.xticks(y_pos + .5 , nominals)

print(classification_report(y_test,rfc.predict(X_test)))
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{

        "n_estimators":[6,8,20],

        "max_depth":[4,6,None],

        "min_samples_leaf":[1,2,3],

        "class_weight":["balanced",None]

    }]



clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), tuned_parameters, cv=5)

clf.fit(X_train, y_train)

print("Best parameters :")

print(clf.best_params_)

print(classification_report(y_test,rfc.predict(X_test)))