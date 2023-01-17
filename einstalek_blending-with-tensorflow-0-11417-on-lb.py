# import some libraries

import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd

from scipy import stats

from scipy.stats import norm, skew 



from sklearn.preprocessing import LabelEncoder



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
def submit(ids, price):

    """

    Writes predicted prices to .csv file.

    

    Arguments:

        ids -- ID values

        price -- predicted price

    """

    subm = pd.DataFrame({'Id': ids,

                        'SalePrice': price})

    subm.to_csv('submission.csv', index=False)
# read train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Save the ID column and then drop it from datasets

train_ID = train['Id']

test_ID = test['Id']



train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# Deleting outliers

train = train.drop(train[(train['GrLivArea'] > 4000) & ((train['SalePrice'] < 300000))].index)



# Take log(1+x) of target variable

train["SalePrice"] = np.log1p(train["SalePrice"])



# Concat train and test

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
# Inplacing missing values

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")



all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

    

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



# Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



# Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# Applying LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# The skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})



skewness = skewness[abs(skewness) > 0.75]



# Applying Box Cox transformation

from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)

    all_data[feat] += 1
# Get dummy categorical features

all_data = pd.get_dummies(all_data)

print(all_data.shape)



train = all_data[:ntrain]

test = all_data[ntrain:]
# Import libraries with regression models

from sklearn.linear_model import ElasticNet,  BayesianRidge, LassoLarsIC, LassoCV

from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

import xgboost as xgb

import lightgbm as lgb



import tensorflow as tf
# So these are 6 models with tuned parameters

lasso = make_pipeline(RobustScaler(), LassoCV(eps =1e-8))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 

                             learning_rate=0.05, max_depth=6, 

                             min_child_weight=1.5, n_estimators=7200,

                             reg_alpha=0.9, reg_lambda=0.6,

                             subsample=0.2,seed=42, silent=1) #0.12105 LB

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
def create_placeholders(n_x, m):

    """

    Creates placeholdes for P and Y.

    

    Arguments:

        n_x -- size of one sample element

        m -- number of regression models

    Returns:

        P -- placeholder for P

        Y -- placeholder for Y

        """

    P = tf.placeholder(tf.float32, name="Preds", shape=[n_x, m])

    Y = tf.placeholder(tf.float32, name="Price", shape=[n_x, 1])

    return P, Y



def compute_cost(P, A, Y, lmbda=0.8):

    """

    Computes cost between predicted prices and actual ones.

    

    Arguments:

        P -- matrix of stacked predictions

        A -- vector of parameters

        Y -- actual prices

        lmbda -- regularazation parameter

    Returns:

        loss -- mean squared error + L1-regularazation

    """

    prediction = tf.matmul(P, A) / tf.reduce_sum(A) # this is formula for WAP

    

    # L1-regularazation has shown better score on LB than L2

    loss = tf.reduce_mean(tf.squared_difference(prediction, Y)) + lmbda*tf.reduce_mean(tf.abs(A))

    return loss



def initialize_parameters(m):

    """

    Initializes parameters A with ones.

    

    Arguments:

        m -- number of models

    Returns:

        A -- vector of parameters

    """

    A = tf.get_variable("Params", dtype=tf.float32, 

                        initializer=tf.constant(np.ones((m,1)).astype(np.float32)))

    return A



def tuning(preds, actual_price, num_iterations=100):

    """

    Implements gradient descent optimizations for WAP.

    

    Arguments:

        pred -- matrix of stacked predictions P

        actual_price -- actual price Y

        num_iterations -- number of iterations

    Returns:

        parameters -- vector A for WAP

    """

    np.random.seed(21)

    tf.reset_default_graph()

    

    (n_x, m) = preds.shape

    costs = []

    # create placeholders for P and Y

    P,  Y = create_placeholders(n_x, m)

    # initialize A

    A = initialize_parameters(m)

    # define loss as a function of A

    loss = compute_cost(P, A, Y)

    # Implement Gradient Descent optimization to minimize loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        # initialize global variables

        sess.run(init)

        for i in range(num_iterations):

            _ , current_cost = sess.run([optimizer, loss], feed_dict={P: preds,

                                                                      Y:actual_price})

            costs.append(current_cost)

        parameters = sess.run(A)

    plt.figure(figsize=(10, 6))

    plt.plot(costs)

    plt.xlabel("iterations")

    plt.ylabel("cost")

    plt.grid(True)

    return parameters
models = (ENet, GBoost, KRR, lasso, model_lgb, model_xgb)

p = []

# Let's train models and stack their predictions in one matrix

for model in models:

    model.fit(train, y_train)

    p.append(model.predict(train))
p = np.array(p)

# transpose p to get P

preds = p.T

actual_price = y_train.reshape(y_train.shape[0], -1)



# And finally let's tune parameters!

params = tuning(preds, actual_price, 700)
# And now let's compute WAP on test dataset

p = []

for model in models:

    p.append(model.predict(test))

p = np.array(p)

preds = p.T

WAP = np.squeeze(np.dot(preds, params) / np.sum(params))
WAP
# Save results to .csv file

submit(test_ID, np.exp(WAP))

# This gave me 0.11417 on LB 