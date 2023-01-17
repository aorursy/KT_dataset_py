import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

% matplotlib inline
# load data
df = pd.read_csv('../input/Automobile_data.csv')
labels = list(df.columns.values)
print(labels)
df.head()
df.describe()
df.info()
object_labels = list(df.select_dtypes(include=['object']).columns.values)
# list of columns with misses
# 'num-of-doors' is categorial, all others are numerical
for col in df.columns:
    if '?' in df[col].unique():
        print(col)
# fill numerical values with average
from sklearn.preprocessing import Imputer
df = df.replace('?', 'NaN')
imp = Imputer(missing_values='NaN', strategy='mean' )
df[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']] = imp.fit_transform(df[['normalized-losses',
                                                                                                      'bore','stroke','horsepower','peak-rpm','price']])
df.head()
df['num-of-doors'].value_counts()
# replace misses in num of doors with the most likely value
df = df.replace('NaN', 'four')
# model distribution among producers
plt.figure(figsize=(10, 5))
sns.countplot(x='make', data=df)
plt.xticks(rotation='vertical')
plt.title('Manufacturers distribution in dataset')
plt.show()
sns.lmplot('city-mpg', 'curb-weight', df, hue="make", fit_reg=False);
sns.lmplot('highway-mpg',"curb-weight", df, hue="make",fit_reg=False);
# Categorial columns list
category_columns = [col for col in df.columns if df.dtypes[col] == 'object']
print(category_columns)
# Categorial columns values distribution
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
for col, ax in zip(category_columns[1:], axs.ravel()):
    sns.countplot(x=col, data=df, ax=ax)
# Generate new features
df['volume'] = df['height'] * df['width'] * df['length']
df['density'] = df['curb-weight'] / df['volume']
df['power-per-volume'] = df['horsepower'] / df['engine-size']
# There also can be generated other parameters describing engine parameters
plt.figure(figsize=(15, 15))
ax = sns.heatmap(df.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)
plt.title('Cross correlation between numerical')
plt.show()
# Replace categorial columns by dummy encoded
Data_category = pd.get_dummies(df[category_columns], drop_first=True)
df_1 = pd.concat([df, Data_category], axis=1)
df_1.drop(category_columns, axis=1, inplace=True)
df_1.head(2)
# Take a look at received zoo of features
from pandas.plotting import scatter_matrix

fig, ax = plt.subplots(figsize=(22,20))
corr = df_1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Cross correlation between numerical')
plt.show()
sns.lmplot(x= 'curb-weight' , y='price', data=df)
sns.lmplot(x= 'engine-size' , y='price', data=df)
sns.lmplot(x= 'engine-size' , y='price', data=df)
sns.lmplot(x= 'engine-size' , y='price',hue = 'fuel-type', data=df)
sns.lmplot(x= 'engine-size' , y='price',hue = 'aspiration', data=df)
sns.lmplot(x= 'highway-mpg' , y='price', data=df)
sns.lmplot(x= 'highway-mpg' , y='price', hue = 'engine-type', data=df)
sns.lmplot(x= 'highway-mpg' , y='price', hue = 'drive-wheels', data=df)
# Generate new variables usung dummies
df_1['engine-size_gas'] = df_1['engine-size'] * df_1['fuel-type_gas']

df_1['highway-mpg_rwd'] = df_1['highway-mpg'] * df_1['drive-wheels_rwd']
df_1['highway-mpg_fwd'] = df_1['highway-mpg'] * df_1['drive-wheels_fwd']
# replace '-' in features names with '_'
df_2 = df_1.copy()
names = []
for name in df_2.columns:
    names.append(name.replace('-', '_'))

df_2.columns = names
import statsmodels.formula.api as smf

lm0 = smf.ols(formula= 'price ~ curb_weight + engine_size + width + highway_mpg + bore + horsepower' , data =df_2).fit()

print(lm0.summary())
lm1 = smf.ols(formula= 'price ~ curb_weight + engine_size + width + height + highway_mpg + bore + stroke + horsepower + peak_rpm + compression_ratio + density + power_per_volume' , data =df_2).fit()

print(lm1.summary())
# Add combination of dummies and numerical features to regression model
lm3 = smf.ols(formula= 'price ~ curb_weight + engine_size + width + height + highway_mpg + bore + stroke + horsepower + peak_rpm + compression_ratio + density + power_per_volume + engine_size_gas + highway_mpg_rwd + highway_mpg_fwd' , data =df_2).fit()

print(lm3.summary())

# Normalize data and split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
nor = Normalizer()
df_3 = nor.fit_transform(df_2)

col = []
for i in df_1.columns:
    col.append(i.replace('-', '_'))
    
df_3 = pd.DataFrame(df_3 , columns  = col)

Y_1 = df_3['price']
X_1 = df_3.drop('price',axis =1)

x_train_1, x_test_1, y_train_1,  y_test_1 = train_test_split(X_1, Y_1,train_size=0.8, test_size=0.2, random_state=42)
# lastly let's just run simple regression on all variables
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


regressor = LinearRegression()
lm_2 = regressor.fit(x_train_1, y_train_1)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pred_train_y = regressor.predict(x_train_1)
pred_test_y = regressor.predict(x_test_1)
cv_score = np.mean(cross_val_score(lm_2, X_1, Y_1, cv=cv, scoring='r2'))

print ('%8s %8s %8s %8s' % ('Train R^2','Test R^2', 'CV R^2', 'MSE'))
print('%.6f %.6f %.6f %.6f'%  (lm_2.score(x_train_1,y_train_1), lm_2.score(x_test_1,y_test_1), cv_score,
                               np.mean((pred_test_y -y_test_1)**2)))
