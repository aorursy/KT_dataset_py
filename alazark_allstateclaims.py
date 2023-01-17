# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt # for plotting

# Any results you write to the current directory are saved as output.
#read data from CSV file

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#Some info about the data

#train.head()
#checking some of the categorical data

print(train.cat1.value_counts())

print(train.cat51.value_counts())

print(train.cat101.value_counts())
#Data housekeeping

#Category variables 116

train_cat = train.ix[:,'cat1':'cat116']

#Continuous variable 14

train_cont = train.ix[:,'cont1':'cont14']

#train_cat_dummy = pd.get_dummies(train_cat)

#print(train_cat.head(2))

#print(train_cat_dummy.head(2))

#Number of training samples

print("number of traning samples : {}".format(train.shape[0]))

print("number of test samples: {}".format(test.shape[0]))

#count the number of unique values under the categorical variables

print("number of unique categorical values : {}".format(len(pd.unique(train_cat[train_cat.columns[1:]].values.ravel()))))

#Check if there are any null values

#print(train.isnull().values.any())

#print(test.isnull().values.any())
#correlation between continuous predictors and traget

train_corr_loss = train.corr()["loss"]

ax = train_corr_loss.iloc[1:-1].plot(kind='bar',title="continuous features correlation with target", figsize=(5,5), fontsize=12)

ax.set_ylabel("correlation value")
#classifcation just considering the continous features(why?)

y_train1 = np.asarray(train['loss'])

X_test1 = test.ix[:,'cont1':'cont14']

lreg = LinearRegression()

lreg.fit(train_cont,y_train1)

y_pred = lreg.predict(X_test1)

print("Training set score: {:.2f}".format(lreg.score(train_cont, y_train1)))
#Categorical features analysis

from sklearn.preprocessing import LabelEncoder

catFeatures = []

for colName in train_cat.columns:

    le = LabelEncoder()

    le.fit(train_cat[colName].unique())

    train_cat[colName] = le.transform(train_cat[colName])

train_cat.head(2)
catFeatures = []

test_cat = test.ix[:,'cat1':'cat101']

for colName in test_cat.columns:

    le = LabelEncoder()

    le.fit(test_cat[colName].unique())

    test_cat[colName] = le.transform(test_cat[colName])

test_cat.head(2)
test_cont = test.ix[:,'cont1':'cont14']

X_train = train_cat.ix[:,'cat1':'cat101'].join(train_cont)

X_test = test_cat.join(test_cont)

lreg.fit(X_train,y_train1)

y_pred = lreg.predict(X_test)

print("Linear: Training set score: {:.2f}".format(lreg.score(X_train, y_train1)))
#Categorical features analysis using  dummy variables

train_cat_dummy = pd.get_dummies(train_cat.ix[:,'cat1':'cat101'])

test_cat_dummy = pd.get_dummies(test_cat)



print("new dummy DF shapes(train)(test):{}{}".format(train_cat_dummy.shape,test_cat_dummy.shape))
X_train = train_cat_dummy.join(train_cont)

X_test = test_cat_dummy.join(test_cont)

lreg.fit(X_train,y_train1)

y_pred = lreg.predict(X_test)

print("Linear: Training set score: {:.2f}".format(lreg.score(X_train, y_train1)))
#Other SGD regressor

from sklearn.linear_model import SGDRegressor

reg= SGDRegressor()

reg.fit(X_train, y_train1)

y_pred = reg.predict(X_test)

print("SGD: Training set score: {:.2f}".format(reg.score(X_train, y_train1)))
#Using decisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor(max_depth=100, random_state=0)

tree.fit(X_train,y_train1)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train1)))

y_pred = tree.predict(X_test)

print(y_pred[0])
#checking out important features

n_features = X_train.shape[1]

imp = tree.feature_importances_

top_imps = imp[imp > 0.015]

indices_imps = np.where(imp > 0.015)

plt.barh(range(len(top_imps)),top_imps, align="center")

plt.yticks(np.arange(len(top_imps)), X_train.columns[indices_imps])

plt.xlabel("Feature importance")

plt.ylabel("Feature")
#RandomForest regressor

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=190, random_state=50)

rfr.fit(X_train,y_train1)

print("Accuracy on training set: {:.3f}".format(rfr.score(X_train, y_train1)))

y_pred2 = rfr.predict(X_test)

print(y_pred2[0])
#Gradient boosting classifier

from sklearn.ensemble import GradientBoostingRegressor

grbr = GradientBoostingRegressor(n_estimators=200, random_state=0, max_depth=10, learning_rate=0.01)

grbr.fit(X_train, y_train1)

print("Accuracy on training set: {:.3f}".format(grbr.score(X_train, y_train1)))

y_pred3 = grbr.predict(X_test)

print(y_pred2[0])
submission7 = pd.DataFrame({

        "id": test["id"],

        "loss": y_pred2

    })

submission7.to_csv('sample_submission6.csv', index=False)