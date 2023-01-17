import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
data = pd.read_csv('../input/train.csv')

data.tail(20)
data.describe()
data.info()
dummy_type = pd.get_dummies(data['type'])

# print sample

dummy_type.sample(2)

# concat

data = pd.concat([data, dummy_type], axis=1)

data = data.rename(columns = {0:'typeO', 1: 'typec'})

print(data.sample(2))
data = data.drop('Total Bags', axis=1)



    

data.head()
data.describe()
# Compute the correlation matrix

corr = data.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
df1 = data[(data['Total Volume'] <= 1.083006e+04) & (data['type'] == 0)]

df2 = data[(data['Total Volume'] > 1.083006e+04) & (data['type'] ==0 )]

df3 = data[ (data['Total Volume'] > 1.083006e+04) & (data['type'] == 1)]
#rgr1

X = df1.drop(['AveragePrice','type'],axis=1)

Y= df1['AveragePrice']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
k_fold = KFold( n_splits=3, shuffle=True, random_state=0)

rgr1 = RandomForestRegressor(n_estimators=60)

mse = metrics.make_scorer(metrics.mean_squared_error)

print (cross_val_score(rgr1, X, Y, cv=k_fold, n_jobs=1, scoring = mse))
rgr1.fit(x_train,y_train)

pred=rgr1.predict(x_test)

print('MSE:', metrics.mean_squared_error(y_test, pred))
#rgr2

X = df2.drop(['AveragePrice','type'],axis=1)

Y= df2['AveragePrice']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
rgr2 = RandomForestRegressor(n_estimators=60)

print (cross_val_score(rgr2, X, Y, cv=k_fold, n_jobs=1, scoring = mse))
rgr2.fit(x_train,y_train)

pred=rgr2.predict(x_test)

print('MSE:', metrics.mean_squared_error(y_test, pred))
#rgr3

X = df3.drop(['AveragePrice','type'],axis=1)

Y= df3['AveragePrice']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
rgr3 = RandomForestRegressor(n_estimators=50)

print (cross_val_score(rgr3, X, Y, cv=k_fold, n_jobs=1, scoring = mse))
rgr3.fit(x_train,y_train)

pred=rgr3.predict(x_test)

print('MSE:', metrics.mean_squared_error(y_test, pred))
test = pd.read_csv('../input/test.csv')

test.head()
test = test.drop('Total Bags', axis=1)

test.head()
dummy_type = pd.get_dummies(test['type'])

# print sample

dummy_type.sample(2)

# concat

test = pd.concat([test, dummy_type], axis=1)

test = test.rename(columns = {0:'typeO', 1: 'typec'})

print(test.sample(2))
test.info()
tf1 = test[(test['Total Volume'] <= 1.083006e+04) & (test['type'] == 0)]

tf2 = test[(test['Total Volume'] > 1.083006e+04) & (test['type'] ==0 )]

tf3 = test[ (test['Total Volume'] > 1.083006e+04) & (test['type'] == 1)]

tf1 = tf1.drop('type',axis=1)

tf2 = tf2.drop('type',axis=1)

tf3 = tf3.drop('type',axis=1)
s1 = pd.DataFrame(tf1['id'])

s2 = pd.DataFrame(tf2['id'])

s3 = pd.DataFrame(tf3['id'])
s1['AveragePrice'] = rgr1.predict(tf1)

s2['AveragePrice'] = rgr2.predict(tf2)

s3['AveragePrice'] = rgr3.predict(tf3)
sol = [s1, s2, s3]

solution = pd.concat(sol, ignore_index=True)

solution
#solution.to_csv('sol_rgr_3models_totvol25%&typebased_WOtypebagreg1&2.csv', index=False)