import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import math

%matplotlib inline
data = pd.read_csv('train.csv')
data["type"].unique()
data.head()
data
data.describe()
corr = data.corr()
print(corr['AveragePrice'])
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot =True)
plt.show()
data.corr()
#print(data.columns)
#new_data = data.drop(["Total Bags"], axis = 1)
#print(new_data.columns)
new_data1 = data.drop(columns = [ "Total Bags", "year","id"])
X_data = new_data1.drop(["AveragePrice"], axis = 1)
y_data = data["AveragePrice"]
print(X_data)
#print(y_data)
from sklearn.model_selection import train_test_split

#X_train,X_val,y_train,y_val = train_test_split(X_data, y_data, test_size = 0.33)

#new_data1.describe()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()             #Instantiate the scaler
scaled_X_train = scaler.fit_transform(X_data)     #Fit and transform the data


""""from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
lr = LinearRegression()

scaled_X_val = scaler.transform(X_val)

lr.fit(scaled_X_train,y_train)
pred = lr.predict(scaled_X_val)

from sklearn.model_selection import cross_val_score
scorer = make_scorer(r2_score)
scores = cross_val_score(lr, scaled_X_train, y_train, cv = 5, scoring = scorer)
print(scores.mean())
#print(math.sqrt(mean_squared_error(y_val,pred)))"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(n_jobs= 4)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

#scaled_X_val = scaler.transform(X_val)

parameters = {'n_estimators' : [100,160,250,85,75], 'max_features': [2,5,7,6,3]}
scorer = make_scorer(r2_score)
grid_obj = GridSearchCV(regr,parameters,scoring=scorer, n_jobs=6)
grid_fit = grid_obj.fit(scaled_X_train,y_data)
best_clf = grid_fit.best_estimator_
#unoptimized_predictions = (regr.fit(scaled_X_train, y_data)).predict(scaled_X_val)      
#optimized_predictions = best_clf.predict(scaled_X_val)       

#acc_unop = r2_score(y_val, unoptimized_predictions)*100       
#acc_op = r2_score(y_val, optimized_predictions)*100         

#print(acc_op)

test = pd.read_csv("test.csv")
new_test = test.drop(columns = [ "Total Bags","year","id"])
scaled_test = scaler.transform(new_test)

pred = best_clf.predict(scaled_test)

pre = pd.Series(data = pred)

newd = pd.concat([test['id'],pre],axis=1)
newd.rename(index = str, columns = {0:'AveragePrice'}, inplace = True)


newd.to_csv("res.csv", index = False)