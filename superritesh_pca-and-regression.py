import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

from sklearn.preprocessing import Imputer

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.metrics import make_scorer

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import tflearn

import tensorflow as tf

import seaborn

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

labels=train["SalePrice"]

test = pd.read_csv('../input/test.csv')

data = pd.concat([train,test],ignore_index=True)

data = data.drop("SalePrice", 1)

ids = test["Id"]
train.head()
# Count the number of rows in train

train.shape[0]
# Count the number of rows in total

data.shape[0]
# Count the number of NaNs each column has.

nans=pd.isnull(data).sum()

nans[nans>0]
# Remove id and columns with more than a thousand missing values

data=data.drop("Id", 1)

data=data.drop("Alley", 1)

data=data.drop("Fence", 1)

data=data.drop("MiscFeature", 1)

data=data.drop("PoolQC", 1)

data=data.drop("FireplaceQu", 1)
# Count the column types

data.dtypes.value_counts()
all_columns = data.columns.values

non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 

                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", 

                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", 

                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 

                   "ScreenPorch","PoolArea", "MiscVal"]



categorical = [value for value in all_columns if value not in non_categorical]
# One Hot Encoding and nan transformation

data = pd.get_dummies(data)



imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

data = imp.fit_transform(data)



# Log transformation

data = np.log(data)

labels = np.log(labels)



# Change -inf to 0 again

data[data==-np.inf]=0
pca = PCA(whiten=True)

pca.fit(data)

variance = pd.DataFrame(pca.explained_variance_ratio_)

np.cumsum(pca.explained_variance_ratio_)
pca = PCA(n_components=36,whiten=True)

pca = pca.fit(data)

dataPCA = pca.transform(data)
# Split traing and test

train = data[:1460]

test = data[1460:]
# R2 Score



def lets_try(train,labels):

    results={}

    def test_model(clf):

        

        cv = KFold(n_splits=5,shuffle=True,random_state=45)

        r2 = make_scorer(r2_score)

        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=r2)

        scores=[r2_val_score.mean()]

        return scores



    clf = linear_model.LinearRegression()

    results["Linear"]=test_model(clf)

    

    clf = linear_model.Ridge()

    results["Ridge"]=test_model(clf)

    

    clf = linear_model.BayesianRidge()

    results["Bayesian Ridge"]=test_model(clf)

    

    clf = linear_model.HuberRegressor()

    results["Hubber"]=test_model(clf)

    

    clf = linear_model.Lasso(alpha=1e-4)

    results["Lasso"]=test_model(clf)

    

    clf = BaggingRegressor()

    results["Bagging"]=test_model(clf)

    

    clf = RandomForestRegressor()

    results["RandomForest"]=test_model(clf)

    

    clf = AdaBoostRegressor()

    results["AdaBoost"]=test_model(clf)

    

    clf = svm.SVR()

    results["SVM RBF"]=test_model(clf)

    

    clf = svm.SVR(kernel="linear")

    results["SVM Linear"]=test_model(clf)

    

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["R Square Score"] 

    #results=results.sort(columns=["R Square Score"],ascending=False)

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0.5,1])

    return results



lets_try(train,labels)
# Split traing and test

train = dataPCA[:1460]

test = dataPCA[1460:]



lets_try(train,labels)
cv = KFold(n_splits=5,shuffle=True,random_state=45)



parameters = {'alpha': [1000,100,10],

              'epsilon' : [1.2,1.25,1.50],

              'tol' : [1e-10]}



clf = linear_model.HuberRegressor()

r2 = make_scorer(r2_score)

grid_obj = GridSearchCV(clf, parameters, cv=cv,scoring=r2)

grid_fit = grid_obj.fit(train, labels)

best_clf = grid_fit.best_estimator_ 



best_clf.fit(train,labels)
# Shape the labels

labels_nl = labels

labels_nl = labels_nl.reshape(-1,1)
tf.reset_default_graph()

r2 = tflearn.R2()

net = tflearn.input_data(shape=[None, train.shape[1]])

net = tflearn.fully_connected(net, 30, activation='linear')

net = tflearn.fully_connected(net, 10, activation='linear')

net = tflearn.fully_connected(net, 1, activation='linear')

sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.01, decay_step=100)

net = tflearn.regression(net, optimizer=sgd,loss='mean_square',metric=r2)

model = tflearn.DNN(net)
model.fit(train, labels_nl,show_metric=True,validation_set=0.2,shuffle=True,n_epoch=50)
# Make predictions



predictions_huber = best_clf.predict(test)

predictions_huber = np.exp(predictions_huber)

predictions_huber = predictions_huber.reshape(-1,)

predictions_DNN = model.predict(test)

predictions_DNN = np.exp(predictions_DNN)

predictions_DNN = predictions_DNN.reshape(-1,)



sub = pd.DataFrame({

        "Id": ids,

        "SalePrice": predictions_huber

    })



sub.to_csv("prices_submission.csv", index=False)

#print(sub)