import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import lightgbm as lgb



%matplotlib inline

sns.set(font_scale=1.3)
# Generate data

n = 100_000

t = np.linspace(-2 * np.pi, 2 * np.pi, int(np.sqrt(n)))

x, z = np.meshgrid(t, t)

y = np.sin(x**2 + z**2) / (x**2 + z**2) + np.cos(x) * 0.5 - 0.5



# Turn to DataFrame

data = pd.DataFrame({

    'x': x.flatten(), 

    'z': z.flatten(),

    'y': y.flatten()

})



fig, ax = plt.subplots(1, 2, figsize=(11, 5))



# Response distribution

sns.distplot(data.y, ax=ax[0])

ax[0].set(title="Distribution of y")



# Response profile

def nice_heatmap(data, v, ax):

    sns.heatmap(data.pivot('z', 'x', v), 

                xticklabels=False, yticklabels=False, 

                cmap='coolwarm', vmin=-1, vmax=1, ax=ax)

    ax.set_title("Heatmap of " + v)

    return None



nice_heatmap(data, v='y', ax=ax[1])

fig.tight_layout()
X_train, X_test, y_train, y_test = train_test_split(

    data[["x", "z"]], data["y"], 

    test_size=0.33, random_state=63

)



print("All shapes:")

for dat in (X_train, X_test, y_train, y_test):

    print(dat.shape)
# Parameters

params = {

    'objective': 'regression',

    'num_leaves': 63,

    'metric': 'l2_root',

    'learning_rate': 0.3,

    'bagging_fraction': 1,

    'min_sum_hessian_in_leaf': 0.01

}



# Data interface

lgb_train = lgb.Dataset(X_train, label=y_train)

                 

# Fitting the model

if False:

    # Find good parameter set by cross-validation

    gbm = lgb.cv(params,

                 lgb_train,

                 num_boost_round=20000,

                 early_stopping_rounds=1000,

                 stratified=False,

                 nfold=5,

                 verbose_eval=1000, 

                 show_stdv=False)

else: 

    # Fit with parameters

    gbm = lgb.train(params,

                    lgb_train,

                    num_boost_round=5000)

# Add predictions to test data

data_eval = pd.DataFrame(np.c_[X_test, y_test], columns=['x', 'z', 'y'])

data_eval["predictions"] = gbm.predict(X_test)

data_eval["residuals"] = data_eval["y"] - data_eval["predictions"]



# Plot the results

fig, ax = plt.subplots(1, 3, figsize=(21, 6))

for i, v in enumerate(['y', 'predictions', 'residuals']):

    nice_heatmap(data_eval, v=v, ax=ax[i])

fig.tight_layout()