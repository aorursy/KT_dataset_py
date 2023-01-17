from sklearn import datasets

boston_dataset = datasets.load_boston()
boston_dataset.keys()
print(boston_dataset['DESCR'])
import pandas as pd

boston_df = pd.DataFrame(data=boston_dataset['data'], columns=boston_dataset['feature_names'])

# Adding the target variable

boston_df['MEDV'] = boston_dataset['target']

boston_df.head()
boston_df.tail()
len(boston_df)
boston_df.isnull().sum()
boston_df.describe().T
boston_df.corr()
cols = ['RM', 'AGE', 'TAX', 'LSTAT', 'MEDV']
# First, we need to set up our plotting environment



# Importing plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns

# Using a jupyter magic function to plot without the need to call plt.show()

%matplotlib inline



# Setting up plot appearance

%config InlineBackend.figure_format='retina'

sns.set()

plt.rcParams['figure.figsize'] = (9, 6)

plt.rcParams['axes.labelpad'] = 10

sns.set_style("darkgrid")
boston_corr_hm = sns.heatmap(boston_df[cols].corr(), cmap=sns.cubehelix_palette(20, light=0.95, dark=0.15))

boston_corr_hm.xaxis.tick_top() # move labels to the top



plt.savefig('boston-housing-exploratory-corr-heatmap.png', bbox_inches='tight', dpi=300)
sns.pairplot(boston_df[cols], plot_kws={'alpha': 0.6}, diag_kws={'bins': 30})



plt.savefig('boston-housing-exploratory-pairplot.png', bbox_inches='tight', dpi=300)
# At first, we start fitting a linear model both to RM x MEDV and LSTAT x MEDV

fig, ax = plt.subplots(1, 2)

sns.regplot('RM', 'MEDV', boston_df, ax=ax[0], scatter_kws={'alpha': 0.4})

sns.regplot('LSTAT', 'MEDV', boston_df, ax=ax[1], scatter_kws={'alpha': 0.4})

plt.savefig('boston-housing-exploratory-linear-reg-models.png', bbox_inches='tight', dpi=300)
# Residual Plots



fig, ax = plt.subplots(1, 2)

ax[0] = sns.residplot('RM', 'MEDV', boston_df, ax=ax[0], scatter_kws={'alpha': 0.4})

ax[0].set_ylabel('MDEV residuals $(y-\hat{y})$')

ax[1] = sns.residplot('LSTAT', 'MEDV', boston_df, ax=ax[1], scatter_kws={'alpha': 0.4})

ax[1].set_ylabel('')

plt.savefig('boston-housing-exploratory-residuals.png', bbox_inches='tight', dpi=300)
# Mean Square Error Calculation



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



def get_mse(df, feature, target='MEDV'):

    # Get x, y to model

    y = df[target].values

    x = df[feature].values.reshape(-1,1)

    print('{} ~ {}'.format(target, feature))

    

    # Build and fit the model

    lm = LinearRegression()

    lm.fit(x, y)

    msg = 'model: y = {:.3f} + {:.3f}x'.format(lm.intercept_, lm.coef_[0])

    print(msg)

    

    # Predict and determine MSE

    y_pred = lm.predict(x)

    error = mean_squared_error(y, y_pred)

    print('mse = {:.2f}'.format(error))

    print()
get_mse(boston_df, 'RM')

get_mse(boston_df, 'LSTAT')
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



poly = PolynomialFeatures(degree=3)



y = boston_df['MEDV'].values

x = boston_df['LSTAT'].values.reshape(-1,1) # format expected by scikit-learn



x_poly = poly.fit_transform(x)

linear_pred_model = LinearRegression()

linear_pred_model.fit(x_poly, y)

x_0 = linear_pred_model.intercept_ + linear_pred_model.coef_[0] # intercept

x_1, x_2, x_3 = linear_pred_model.coef_[1:]       # other coefficients

msg = 'model: y = {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f}x^3'.format(x_0, x_1, x_2, x_3)

print(msg) # model description
# Plotting the Samples and the Polynomial Model

import numpy as np



fig, ax = plt.subplots()



# Plot the samples

ax.scatter(x.flatten(), y, alpha=0.6)



# Plot the polynomial model

x_ = np.linspace(2, 38, 50).reshape(-1, 1)

x_poly = poly.fit_transform(x_)

y_ = linear_pred_model.predict(x_poly)

ax.plot(x_, y_, color='red', alpha=0.8)

ax.set_xlabel('LSTAT')

ax.set_ylabel('MEDV')

plt.savefig('boston-housing-exploratory-3rd-poly.png', bbox_inches='tight', dpi=300)
# Mean Square Error Calculation

from sklearn.metrics import mean_squared_error



x_poly = poly.fit_transform(x)

y_pred = linear_pred_model.predict(x_poly)

resid_MEDV = y - y_pred

ms_error = mean_squared_error(y, y_pred)

print('mse = {:.3f}'.format(ms_error))
fig, ax = plt.subplots(figsize=(5, 7))

ax.scatter(x, resid_MEDV, alpha=0.6)

ax.set_xlabel('LSTAT')

ax.set_ylabel('MEDV Residual $(y-\hat{y})$')

plt.axhline(0, color='black', ls='dotted')

plt.savefig('boston-housing-exploratory-3rd-poly-residuals.png', bbox_inches='tight', dpi=300)
# Plot Cumulative Distribution Function (CDF) of the 'AGE' feature



sns.distplot(boston_df.AGE.values, bins=100, hist_kws={'cumulative': True}, kde_kws={'lw': 0})

plt.xlabel('AGE')

plt.ylabel('CDF')

plt.axhline(0.33, color='red')

plt.axhline(0.66, color='red')

plt.xlim(0, boston_df.AGE.max())

plt.savefig('boston-housing-exploratory-age-cdf.png', bbox_inches='tight', dpi=300)
# Categorize AGE into 3 bins



def get_age_category(x):

    if x < 50:

        return 'Relatively New'

    elif 50 <= x < 85:

        return 'Relatively Old'

    else:

        return 'Very Old'



boston_df['AGE_category'] = boston_df.AGE.apply(get_age_category)



# Check the segmented counts

boston_df.groupby('AGE_category').size()
# How is MEDV distributed for each age category (including data points)



sns.violinplot(x='MEDV', y='AGE_category', data=boston_df, order=['Relatively New', 'Relatively Old', 'Very Old'], inner='point')

plt.xlim(0, 55)

plt.savefig('boston-housing-exploratory-age-medv-violin.png', bbox_inches='tight', dpi=300)