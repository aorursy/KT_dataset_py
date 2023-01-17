import numpy as np

import pandas as pd

from random import uniform

from scipy.optimize import dual_annealing

import matplotlib.pyplot as plt

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
df = pd.read_csv("../input/insurance.csv")
display(df.shape)

display(df.info())

display(df.head())

display(df["region"].value_counts())
sns.pairplot(df)

plt.show()
df["region"] = pd.Categorical(df["region"])

df["code"] = df["region"].cat.codes

sex_male = pd.get_dummies(df["sex"], drop_first=True)

smoke_yes = pd.get_dummies(df["smoker"], drop_first=True)

df_concat = pd.concat([df, sex_male, smoke_yes], axis=1)
sns.pairplot(df_concat)

plt.show()
X = df_concat[["age", "bmi", "children", "male", "yes", "code"]]

y = df_concat["charges"]

display(X.shape)

display(X.head())

display(y.shape)

display(y.head())
model = sm.OLS(y, sm.add_constant(X))

result = model.fit()

display(result.summary())
print(result.params)
def ackley2d(x):

    total = result.params[0]

    s1 = np.sum([x[i]*result.params[i+1] for i in range(len(x))])

    total += s1

    # or

    # total = 20.0

    # s1 = np.sum(x**2) / 2

    # total -= 20.0 * np.exp(-0.2 * s1**0.5)

    # s2 = np.sum(np.cos(2*np.pi*x))

    # total -= np.exp(s2 / 2)

    return total
bounds = [(X[i].min(), X[i].max()) for i in X.columns]
bounds
n_trial = 10
x = np.zeros(6)

for i in range(n_trial):

    # Initial value

    x[0] = uniform(bounds[0][0], bounds[0][1])

    x[1] = uniform(bounds[1][0], bounds[1][1])

    x[2] = uniform(bounds[2][0], bounds[2][1])

    x[3] = uniform(bounds[3][0], bounds[3][1])

    x[4] = uniform(bounds[4][0], bounds[4][1])

    x[5] = uniform(bounds[5][0], bounds[5][1])

#    x[6] = uniform(bounds[6][0], bounds[6][1])

#    x[7] = uniform(bounds[7][0], bounds[7][1])

#    x[8] = uniform(bounds[8][0], bounds[8][1])

    print(x)  # debug

    

    # Dual annealing optimization

    ret = dual_annealing(ackley2d, bounds, x0=x, maxiter=500)

    print('x:', ret.x)

    print('f(x):', ret.fun)