# Data Analysis #

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Model stuff #

from sklearn.model_selection import train_test_split



from sklearn.linear_model import RidgeCV

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNetCV

from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

plt.style.use("seaborn")

plt.rcParams['figure.figsize'] = (12,5)
# Import data #

dpath = '../input/'

diamonddf = pd.read_csv(dpath + "diamonds.csv")
diamonddf.head()
diamonddf.info()
diamonddf.drop("Unnamed: 0", axis = 1, inplace = True) # drop weird column
diamonddf.isnull().sum()
diamonddf.dtypes
# Are there any weird values? #

diamonddf.describe(include=['O'])
# Quantitative description #

diamonddf.describe()
numcols = diamonddf.select_dtypes(include = ['float64','int64']).columns.tolist()
colors = sns.color_palette("deep")

fig,axes = plt.subplots(3,3, figsize = (12,8)) # up to 9 quant vars

sns.distplot(diamonddf["carat"], color = colors[0], ax = axes[0,0])

sns.distplot(diamonddf["depth"], color = colors[1], ax = axes[0,1])

sns.distplot(diamonddf["table"], color = colors[2], ax = axes[0,2])

sns.distplot(diamonddf["price"], color = colors[3], ax = axes[1,0])

sns.distplot(diamonddf["x"], color = colors[4], ax = axes[1,1])

sns.distplot(diamonddf["y"], color = colors[0], ax = axes[1,2])

sns.distplot(diamonddf["z"], color = colors[1], ax = axes[2,0])

plt.suptitle("Distribution of Quantitative Data", size = 16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
colors = sns.color_palette("deep")

fig,axes = plt.subplots(3,3, figsize = (12,8)) # up to 9 quant vars

sns.boxplot(y= diamonddf["carat"], color = colors[0], ax = axes[0,0])

sns.boxplot(y = diamonddf["depth"], color = colors[1], ax = axes[0,1])

sns.boxplot(y = diamonddf["table"], color = colors[2], ax = axes[0,2])

sns.boxplot(y = diamonddf["price"], color = colors[3], ax = axes[1,0])

sns.boxplot(y = diamonddf["x"], color = colors[4], ax = axes[1,1])

sns.boxplot(y = diamonddf["y"], color = colors[0], ax = axes[1,2])

sns.boxplot(y = diamonddf["z"], color = colors[1], ax = axes[2,0])

plt.suptitle("Distribution of Quantitative Data (boxplots)", size = 16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
# diamonds that are probably errors

zero_df = diamonddf[(diamonddf['x'] == 0) |

           (diamonddf['y'] == 0) |

           (diamonddf['z'] == 0)]

zero_df.head()
zero_df.shape
# Drop the rows with zero as any x, y, or z

diamonddf.drop(zero_df.index, inplace = True)
cat_vars = diamonddf.select_dtypes(include = 'object').columns.tolist()

fig, axes = plt.subplots(1,3, figsize = (12,5))

i = 0

for var_name in cat_vars:

    diamonddf[var_name].value_counts().sort_values().plot(kind = 'barh', color = 'C0', ax = axes[i])

    axes[i].set_title(var_name)

    i += 1

plt.tight_layout()

plt.show()
train_df, test_df = train_test_split(diamonddf, test_size=0.2, random_state=12)
Y_test = test_df['price']

X_test = test_df.drop('price', axis = 1)
print("Total dataset size: {}".format(diamonddf.shape))

print("Training set size (80%): {}".format(train_df.shape))

print("Test set size (20%): {}".format(test_df.shape))
diamonds = train_df.copy()
# Pair plot#

sns.pairplot(diamonds)

plt.show()
ol1 = diamonds[diamonds['z'] > 20].index

ol2 = diamonds[diamonds['y'] > 20].index



fig, axes = plt.subplots(1,3, figsize = (12,4))

sns.scatterplot(x = diamonds['carat'], y = diamonds['z'], ax = axes[0]) 

axes[0].annotate(ol1[0], (diamonds['carat'].loc[ol1], diamonds['z'].loc[ol1]), size = 12)



sns.scatterplot(x = diamonds['x'], y = diamonds['y'], ax = axes[1])

axes[1].annotate(ol2[0], (diamonds['x'].loc[ol2], diamonds['y'].loc[ol2]), size = 12)



sns.scatterplot(x = diamonds['y'], y = diamonds['z'], ax = axes[2])

axes[2].annotate(ol1[0], (diamonds['y'].loc[ol1], diamonds['z'].loc[ol1]), size = 12)

axes[2].annotate(ol2[0], (diamonds['y'].loc[ol2] - 4, diamonds['z'].loc[ol2] + 1), size = 12)



plt.suptitle("Outliers in 3 sample plots", size = 14)

plt.show()
diamonds[diamonds['z'] > 20]
diamonds[diamonds['y'] > 20]
diamonds['z'].describe()
cond = (diamonds['y'] > 20) | (diamonds['z'] > 20) 

diamonds.drop(diamonds[cond].index, inplace = True)
fig, axes = plt.subplots(1,3, figsize = (12,4))

sns.scatterplot(x = diamonds['carat'], y = diamonds['z'], ax = axes[0]) 



sns.scatterplot(x = diamonds['x'], y = diamonds['y'], ax = axes[1])



sns.scatterplot(x = diamonds['y'], y = diamonds['z'], ax = axes[2])



plt.suptitle("3 Sample Plots without Outliers", size = 16)

plt.show()
sns.heatmap(diamonds.corr(), cmap = "RdBu_r", square = True, annot=True, cbar=True)

plt.title("Correlation Between Variables")

plt.show()
# Drop x, y, and z #

diamonds.drop(['x','y','z'], axis = 1, inplace = True)
clar_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

color_order = sorted(diamonds['color'].unique().tolist(), reverse = True)
fig, axes = plt.subplots(1,2, figsize = (12,5))

sns.boxplot(x = "cut", y = "price", data = diamonds, order = cut_order, ax = axes[0], palette = 'Blues')

sns.boxplot(x = 'clarity', y = 'price', data = diamonds, order = clar_order, ax = axes[1], palette = 'Blues')

plt.suptitle("Diamond Price by Cut and Clarity", size = 14)

plt.show()
fig, axes = plt.subplots(1,2, figsize = (12,5))

sns.scatterplot(x = 'carat', y = 'price', hue = "cut", palette = 'Blues', hue_order = cut_order,

                size = 10, data = diamonds, ax = axes[0])

sns.scatterplot(x = 'carat', y = 'price', hue = "clarity", palette = 'Blues', hue_order = clar_order,

                size = 10, data = diamonds, ax = axes[1])

plt.suptitle("Diamond Price vs. 2 predictors", size = 14)

plt.show()
fig, axes = plt.subplots(1,2, figsize = (12,5))

sns.boxplot(x = "color", y = "price", data = diamonds, order = color_order, palette = 'Blues', ax = axes[0])

sns.scatterplot(x = 'carat', y = 'price', hue = "color", palette = 'Blues', hue_order = color_order,

                size = 10, data = diamonds, ax = axes[1])

plt.suptitle("Diamond Price by Color", size = 14)

plt.show()
fig, axes = plt.subplots(2, 3, figsize = (12,6))

sns.kdeplot(np.log(diamonds['price']), shade=True , color='r', ax = axes[0,0])

axes[0,0].set_title("Log transform")

sns.kdeplot(np.sqrt(diamonds['price']), shade=True , color='b', ax = axes[0,1])

axes[0,1].set_title("Square root transform")

sns.kdeplot((diamonds['price']**(1/3)), shade=True , color='coral', ax = axes[0,2])

axes[0,2].set_title("Cube root transform")

sns.boxplot(y = np.log(diamonds['price']), ax = axes[1,0], color = 'coral')

sns.boxplot(y = np.sqrt(diamonds['price']), ax = axes[1,1], color = 'coral')

sns.boxplot(y = (diamonds['price']**(1/3)), ax = axes[1,2], color = 'coral')

plt.tight_layout()

plt.show()
def error_metrics(y_true, y_pred):

    mean_abs = "Mean Absolute Error: {}".format(mean_absolute_error(y_true, y_pred))

    mean_squared = "Mean Square Error: {}".format(mean_squared_error(y_true, y_pred))

    r2 = "r2 score: {}".format(r2_score(y_true, y_pred))

    return mean_abs, mean_squared, r2
# Remove the label #

X_train = diamonds.drop('price', axis = 1)

Y_train = diamonds['price'].copy()
def cat_mapper(categories):

    "create a dictionary that maps integers to the ordered categories"

    i = 0

    mapped = {}

    for cat in categories:

        mapped[cat] = i

        i += 1

    return mapped
cat_mapper(color_order)
cat_mapper(cut_order)
cat_mapper(clar_order)
X_train[cat_vars].head()
X_train_mapped = X_train.copy()

X_train_mapped['cut'] = X_train_mapped['cut'].map(cat_mapper(cut_order))

X_train_mapped['color'] = X_train_mapped['color'].map(cat_mapper(color_order))

X_train_mapped['clarity'] = X_train_mapped['clarity'].map(cat_mapper(clar_order))
minmaxscaler = MinMaxScaler()

numcols = ['carat','depth','table']
X_train_mapped[numcols] = minmaxscaler.fit_transform(X_train_mapped[numcols])
X_train_mapped.head()
# Adjusting the test datasets #

X_test.drop(['x','y','z'], axis = 1, inplace = True)

X_test['cut'] = X_test['cut'].map(cat_mapper(cut_order))

X_test['color'] = X_test['color'].map(cat_mapper(color_order))

X_test['clarity'] = X_test['clarity'].map(cat_mapper(clar_order))
X_test[numcols] = minmaxscaler.transform(X_test[numcols])
alphas = [.01,.1,1,10,100,1000,10000]
ridge = RidgeCV(alphas = alphas, cv = 5)

ridge_fit = ridge.fit(X_train_mapped, Y_train)
yhat_ridge = ridge_fit.predict(X_test)
sns.distplot(Y_test - yhat_ridge)

plt.title("Distribution of Errors (Ridge Regression)")

plt.show()
x = np.linspace(0, 30000, 1000)

sns.scatterplot(x = Y_test, y = yhat_ridge)

plt.plot(x,x, color = 'red', linestyle = 'dashed')

plt.xlim(-100, 36000)

plt.ylim(-100, 36000)

plt.title("Actual vs. Predicted (Ridge Regression)")

plt.show()
# Ridge error metrics #

error_metrics(Y_test, yhat_ridge)
lasso = LassoCV(cv=5, random_state=12, alphas = alphas)

lasso_fit = lasso.fit(X_train_mapped, Y_train)

yhat_lasso = lasso_fit.predict(X_test)
error_metrics(Y_test, yhat_lasso)
sns.distplot(Y_test - yhat_lasso)

plt.title("Distribution of Errors (LASSO Regression)")

plt.show()
sns.scatterplot(x = Y_test, y = yhat_lasso)

plt.plot(x,x, color = 'red', linestyle = 'dashed')

plt.xlim(-100, 36000)

plt.ylim(-100, 36000)

plt.title("Actual vs. Predicted (LASSO Regression)")

plt.show()
elasticnet = ElasticNetCV(cv=5, random_state=12,

                          l1_ratio = 0.9,

                          alphas = alphas)

elastic_fit = elasticnet.fit(X_train_mapped, Y_train)

yhat_elastic = elastic_fit.predict(X_test)
error_metrics(Y_test, yhat_elastic)
randomforest = RandomForestRegressor(max_depth=5, 

                                     random_state=12, 

                                     n_estimators = 1000)

                                     
rf_fit = randomforest.fit(X_train_mapped, Y_train)

yhat_rf = rf_fit.predict(X_test)
error_metrics(Y_test, yhat_rf)
sns.distplot(Y_test - yhat_rf)

plt.title("Distribution of Errors (Random Forest Regression)")

plt.show()
sns.scatterplot(x = Y_test, y = yhat_rf)

plt.plot(x,x, color = 'red', linestyle = 'dashed')

plt.xlim(-100,20000)

plt.ylim(-100, 20000)

plt.title("Actual vs. Predicted (LASSO Regression)", size = 14)

plt.show()