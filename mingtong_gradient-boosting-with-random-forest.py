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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.shape, test.shape
#train.describe()

#train.info()
#train["SalePrice"].hist();
#prices = np.log1p(train["SalePrice"])

#prices.hist()
train["SalePrice"] = np.log1p(train["SalePrice"])

y_train = train["SalePrice"]
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
from scipy.stats import skew

from scipy.stats.stats import pearsonr

#log transform skewed numeric features:

numeric_feats = train.dtypes[train.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index
'SalePrice' in skewed_feats
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#Convert categorical variable into dummy/indicator variables

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

all_data.shape
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

X_train.shape, X_test.shape
#X_train.info
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.4, random_state=0)

#X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import cross_val_score

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV



#reg = linear_model.Ridge(alpha = .5)

#reg.fit(X_train, y_train) 

#y_pred = reg.predict(X_test)

#mean_squared_error(y_test, y_pred)  
#mean_squared_error(y_test, y_pred, multioutput='raw_values')
#reg = linear_model.RidgeCV(alphas = [0.01, 0.1, 1.0, 10])

#reg.fit(X_train, y_train)

#reg.alpha_   
#reg = linear_model.RidgeCV(alphas = [2.0, 5., 10, 20., 50.])

#reg.fit(X_train, y_train)

#reg.alpha_   
#reg = linear_model.RidgeCV(alphas = [6., 8., 10., 12., 14., 17.], fit_intercept=True, \

 #                          normalize=False, scoring=None, cv=8, gcv_mode=None, store_cv_values=False)

#reg.fit(X_train, y_train)
#print(reg.alpha_)

#y_pred = reg.predict(X_test)

#y_pred = np.expm1(y_pred)
#output = pd.DataFrame(test['Id'])
#output['SalePrice'] = y_pred
#output.to_csv('RidgeRegression_firsttry_12232016.csv', index=False)
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.

clf = LassoCV()



# Set a minimum threshold, using default threshold 1e-100 ~ 1e-4 first

sfm = SelectFromModel(clf, threshold=1e-100)

sfm.fit(X_train, y_train)

X_transform = sfm.transform(X_train)

n_features = X_transform.shape[1]

print(n_features)
print(n_features)

print(X_train.shape)

X_index = np.linspace(0, 287, 288)

xv, yv = np.meshgrid(X_index, range(1))

print(xv.shape)

selected_features = sfm.transform(xv)

print(selected_features)
X_selected_tr = X_train.as_matrix()

print(X_selected_tr.shape)

X_selected_tr = X_selected_tr[:, [3, 5, 6, 8, 22, 24, 26, 27, 28]]
X_transform - X_selected_tr #Same even though features return type is not integer




# Reset the threshold till the number of features equals two.

# Note that the attribute can be set directly instead of repeatedly

# fitting the metatransformer.

#while n_features > 2:

#    sfm.threshold += 0.1

#    X_transform = sfm.transform(X)

#    n_features = X_transform.shape[1]



# Plot the selected two features from X.

#import matplotlib.pyplot as plt

#plt.title(

#    "Feature selected from LASSO "

#    "threshold %0.3f." % sfm.threshold)

#feature1 = X_transform[:, 3]

##feature2 = X_transform[:, 1]

#plt.plot(feature1, 'r.')

#plt.xlabel("Feature number 1")

#plt.ylabel("Feature value")

#plt.ylim([np.min(feature2), np.max(feature2)])

#plt.show()
from sklearn.linear_model import LassoCV

reg = linear_model.LassoCV(alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

reg.fit(X_transform, y_train)

print(reg.alpha_)
X_test_selected = X_test.as_matrix()[:, [3, 5, 6, 8, 22, 24, 26, 27, 28]]

#y_pred = reg.predict(X_test_selected)

#y_pred = np.expm1(y_pred)
#output = pd.DataFrame(test['Id'])

#output['SalePrice'] = y_pred

#output.to_csv('Lasso_selected9_03132017.csv', index=False)
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.pipeline import make_pipeline



# 建立模型

from sklearn.ensemble import RandomForestRegressor



#forest = RandomForestClassifier(oob_score=True, n_estimators=10000)

forest = GradientBoostingRegressor(RandomForestRegressor(random_state=0, n_estimators=100))



forest.fit(X_train, y_train)

feature_importance = forest.feature_importances_

# 调整特征重要性的数值范围

import matplotlib.pyplot as plt

feature_importance = 100.0 * (feature_importance / feature_importance.max())

plt.plot(feature_importance)
fi_threshold = 1



# 取特征重要性靠前的变量

important_idx = np.where(feature_importance > fi_threshold)[0]





# 取特征名称

print(important_idx)

#important_features = xv[important_idx]

#print "n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):n",

#        important_features
y_pred = forest.predict(X_test)

#y_pred = regr.predict(X_test_selected)

y_pred = np.expm1(y_pred)



output = pd.DataFrame(test['Id'])

output['SalePrice'] = y_pred

output.to_csv('randomforest0319.csv', index=False)