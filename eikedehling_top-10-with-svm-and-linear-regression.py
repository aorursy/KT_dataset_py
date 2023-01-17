import numpy as np

import pandas

from scipy.stats import skew

from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.svm import SVR



train = pandas.read_csv("../input/train.csv")

test = pandas.read_csv("../input/test.csv")
all_data = pandas.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                          test.loc[:,'MSSubClass':'SaleCondition']))



#log transform the price:

train["SalePrice"] = np.log1p(train["SalePrice"])



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



all_data = pandas.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())



#log transform skewed numeric features:

skewness = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))

left_skewed_feats = skewness[skewness > 0.5].index

right_skewed_feats = skewness[skewness < -0.5].index

all_data[left_skewed_feats] = np.log1p(all_data[left_skewed_feats])

#all_data[right_skewed_feats] = np.exp(all_data[right_skewed_feats])



scaler = RobustScaler()

all_data[numeric_feats] = scaler.fit_transform(all_data[numeric_feats])



X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train['SalePrice']
alphas = [0.0005, 0.00075, 0.001, 0.00125, 0.0015]

scores = [

     np.sqrt(-cross_val_score(ElasticNet(alpha), X_train, y, scoring="neg_mean_squared_error", cv=5)).mean()

     for alpha in alphas

]

scores = pandas.Series(scores, index=alphas)

scores.plot(title = "Alphas vs error (Lowest error is best)")
gsc = GridSearchCV(

    estimator=SVR(kernel='rbf'),

    param_grid={

        'C': range(1, 4),

        'epsilon': (0.03, 0.04, 0.05, 0.06, 0.07),

    },

    cv=5

)

grid_result = gsc.fit(X_train, y)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.scatter([row['C'] for row in grid_result.cv_results_['params']],

           [row['epsilon'] for row in grid_result.cv_results_['params']],

           grid_result.cv_results_['mean_test_score'],

           c='b', marker='^')



ax.set_xlabel('C')

ax.set_ylabel('Epsilon')

ax.set_zlabel('Score')
linear_model = ElasticNet(alpha=0.001)

linear_model.fit(X_train, y)



svr_model = SVR(kernel='rbf', C=2, epsilon=0.05)

svr_model.fit(X_train, y)
test['SalePrice'] = np.expm1((linear_model.predict(X_test) +

                              svr_model.predict(X_test)) / 2.0)



test.to_csv('submission.csv', index=False, columns=['Id', 'SalePrice'])