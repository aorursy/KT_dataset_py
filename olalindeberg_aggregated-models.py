import numpy as np # Linear algebra

import pandas as pd # Common library for handling data objects

from sklearn.linear_model import LinearRegression, SGDRegressor # sklearn have lots of nice and easy models

import seaborn as sns

import matplotlib.pyplot as plt
# Return simulated data in for first key in parameters as data_'key', e.g. data_A

n_obs = 20000

parameters_distribution = {

    "A":

         {"age": (25, 5)

         },  

    "B":

        {"age": [30, 5]

         }, 

    "C":

        {"age": [35, 5]

         },   

}



for hospital, param in parameters_distribution.items():

    age = np.random.normal(*param['age'], size=n_obs).astype(int)

    stay = 50 + age * 0.4 + np.random.normal(0, 1, size=n_obs)

    data_hospital = 'data_' + hospital 

    vars()[data_hospital] = pd.DataFrame({'age': age, "stay": stay})
sns.regplot(x='age', y='stay', data=data_C)
data_A['hospital'] = 'A'

data_B['hospital'] = 'B'

data_C['hospital'] = 'C'

data_all = pd.concat([data_A, data_B, data_C])
sns.regplot(x='age', y='stay', data=data_all)




X_all = data_all[['age']].values

y_all = data_all['stay'].values



clf = LinearRegression()

model_linear = clf.fit(X_all, y_all)



print('intercept:' + str(model_linear.intercept_), 'betas:' + str(model_linear.coef_))
X_A = data_A[['age']].values

y_A = data_A['stay'].values

X_B = data_B[['age']].values

y_B = data_B['stay'].values

X_C = data_C[['age']].values

y_C = data_C['stay'].values



model_sgd  = SGDRegressor(alpha=0.0001, epsilon=0.1, eta0=0.01, fit_intercept=True,

       l1_ratio=0.15, learning_rate='invscaling', loss='squared_loss',

       max_iter=500, penalty=None, power_t=0.25, random_state=None,

       shuffle=False, verbose=0, warm_start=True)



model_sgd.n_iter = np.ceil(10**6 / len(y_A))

model_sgd.fit(X_A, y_A)

model_sgd.fit(X_B, y_B)

model_sgd.fit(X_C, y_C)



print('intercept:' + str(model_sgd.intercept_), 'betas:' + str(model_sgd.coef_))
# Making up some outcome to make a point

stay_A = 62

stay_B = 64

stay_C = 63

stay_average = (stay_A + stay_B + stay_C)/3



# Predicting the real results

pred_A = model_sgd.predict(X_A).mean()

pred_B = model_sgd.predict(X_B).mean()

pred_C = model_sgd.predict(X_C).mean()
df_results = pd.DataFrame(

    {'A': [stay_A, pred_A], 

     "B": [stay_B, pred_B],

     "C": [stay_C, pred_C],

     "stay_average": [stay_average, stay_average]

    },

    index=['Actual', 'Pred']

)
sns.scatterplot(data=df_results[['A', 'B', 'C']])

plt.plot([0, 1], [stay_average, stay_average], linewidth=2)