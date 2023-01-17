# Import the various libraries we'll be using in this notebook

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import statsmodels.api as sm



# Graphics libraries:

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

df = pd.read_csv('../input/train.csv')



print('In the Ames Home Prices training set, there are: ')

print('    ', df.shape[0], 'rows of data')

print('with  ', df.shape[1] , 'features for each row')

df.columns
df['SalePrice'].describe()
sns.distplot(df['SalePrice']);

print('Kurtosis = ', df['SalePrice'].kurtosis())

print('Skew     = ', df['SalePrice'].skew())


sns.set(style="white")



# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.15, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .75})
df.corr()['SalePrice'].sort_values(ascending=False)
print(df.head(10)['OverallQual'])

df['OverallQual'].describe()
data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)

fig = plt.subplots(figsize=(8, 6))

fig = sns.swarmplot(x="OverallQual", y="SalePrice", data=df, color=".25")

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
ax = sns.regplot(x="GrLivArea", y="SalePrice", data=df)

X = df["GrLivArea"]

y = df["SalePrice"]



# Note the difference in argument order

model = sm.OLS(y, X).fit()

predictions = model.predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
test_df = pd.read_csv('../input/test.csv')



test_df['SalePrice'] = 118.0691*test_df['GrLivArea']

test_df.to_csv('submission.csv', columns=['Id', 'SalePrice'], index=False)