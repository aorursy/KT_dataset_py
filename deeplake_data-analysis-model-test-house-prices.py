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

#check histogram and see if better to log-transform

seaborn.distplot(labels);
train.head()
# Count the number of rows in train and total

train_sz = train.shape[0]

data_sz = data.shape[0]

# Count the number of NaNs each column has. Display columns having more than 30% NAN

DROP_NAN_PCT = 0.3

nans=pd.isnull(data).sum()

nans[nans > data_sz * DROP_NAN_PCT]
# REMOVE_NOISE_BY_NAN_COLUMNS

data=data.drop("Id", 1)

data=data.drop("Alley", 1)

data=data.drop("Fence", 1)

data=data.drop("MiscFeature", 1)

data=data.drop("PoolQC", 1)

data=data.drop("FireplaceQu", 1)

# Count the column types

data.dtypes.value_counts()

data.head()
# CHECK_CORRELATION_AND_KEY_FACTORS

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 12))

k = 30  #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

seaborn.set(font_scale=1)

hm = seaborn.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot - check that key factors shows positive relationship with SalePrice

seaborn.set()

seaborn.pairplot(train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']], size = 2.5)

plt.show();
# LOG_TRANSFORMATION

all_columns = data.columns.values

non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 

                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", 

                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", 

                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 

                   "ScreenPorch","PoolArea", "MiscVal"]



#'OverallQual', 'FullBath', 'YearBuilt' are categorical cases 

categorical = [value for value in all_columns if value not in non_categorical]



# Convert Categorical Variable Into Dummy Variables 

data = pd.get_dummies(data)

# Like to re-visit, as I'm not 100% comfortable to set NAN to most_frequent...

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

data = imp.fit_transform(data)
# Log transformation

data = np.log(data)

labels = np.log(labels)

# Change -inf to 0 again

data[data==-np.inf]=0
pca = PCA(whiten=True).fit(data)

np.cumsum(pca.explained_variance_ratio_)

#shows the variance is explained by N factors 
# per result above, get to know that 36 factors will be good enough

pca = PCA(n_components=36,whiten=True)

pca = pca.fit(data)

dataPCA = pca.transform(data)
# data

train = data[:1460]

test = data[1460:]
def apply_models(train,labels):

    results={}

    def train_get_score(clf):        

        cv = KFold(n_splits=5,shuffle=True,random_state=45)

        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=make_scorer(r2_score))

        return [r2_val_score.mean()]



    results["Linear"]=train_get_score(linear_model.LinearRegression())

    results["Ridge"]=train_get_score(linear_model.Ridge())

    results["Bayesian Ridge"]=train_get_score(linear_model.BayesianRidge())

    results["Hubber"]=train_get_score(linear_model.HuberRegressor())

    results["Lasso"]=train_get_score(linear_model.Lasso(alpha=1e-4))

    results["RandomForest"]=train_get_score(RandomForestRegressor())

    results["SVM RBF"]=train_get_score(svm.SVR())

    results["SVM Linear"]=train_get_score(svm.SVR(kernel="linear"))

    

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["R Square Score"] 

    #results=results.sort(columns=["R Square Score"],ascending=False)

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0.5,1])

    return results



apply_models(train,labels)
# dataPCA

train = dataPCA[:1460]

test = dataPCA[1460:]

apply_models(train,labels)
def gridSearch_predict(clf, train, labels):

    cv = KFold(n_splits=5,shuffle=True,random_state=45)

    parameters = {'alpha': [1000,100,10],'epsilon' : [1.2,1.25,1.50],'tol' : [1e-10]}

    grid_obj = GridSearchCV(clf, parameters, cv=cv,scoring=make_scorer(r2_score))

    predict_model = grid_obj.fit(train, labels).best_estimator_

    predict_model.fit(train,labels)

    return predict_model



predict_model = gridSearch_predict(linear_model.HuberRegressor(),train,labels)
# Make predictions

predictions = np.exp(predict_model.predict(test))

sub = pd.DataFrame({"Id": ids, "SalePrice": predictions})

print(sub)

sub.to_csv("prices_submission.csv", index=False)
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

model_DNN = tflearn.DNN(net)
model_DNN.fit(train, labels_nl,show_metric=True,validation_set=0.2,shuffle=True,n_epoch=50)
# Make predictions

predictions_DNN = np.exp(model_DNN.predict(test))

predictions_DNN = predictions_DNN.reshape(-1,)

sub = pd.DataFrame({"Id": ids, "SalePrice": predictions_DNN})

print(sub)

#sub.to_csv("prices_submission.csv", index=False)