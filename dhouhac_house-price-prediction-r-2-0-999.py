# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/kc_house_data.csv')
train.info()
train.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train.hist(bins=30, figsize=(14,12))
plt.show()
train.describe()
correlations = train.corr()

def heatmap_gen(corr):
    sns.set(style='white')
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    plt.figure(figsize=(9,8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr * 100, annot=True, fmt='.0f', mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    

heatmap_gen(correlations)
sns.pairplot(train[['sqft_living', 'grade', 'bathrooms', 'price']])
plt.show()
sns.boxplot(x='grade', y='price', data=train)
plt.show()
train.plot.scatter(x='grade', y='price', s=train.sqft_living * 0.05,alpha=0.2, c='waterfront',colormap='plasma',edgecolors='grey', figsize=(10,10))
plt.show()
train.isnull().any()
for feature in ['bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'waterfront']:
    print(feature, train[feature].sort_values().unique())
train[train.bathrooms == 0]
sns.violinplot(x='bedrooms', data=train)
plt.show()
train[train.bedrooms > 30]
train_cp = train.copy()
train_cp = train_cp[train_cp.bedrooms < 30]
train_cp['diff_renov_built'] = train_cp.yr_renovated - train_cp.yr_built
train_cp[(train_cp.diff_renov_built < 0) & (train_cp.yr_renovated !=0)]
train_cp['isrenovated'] = (train_cp.yr_renovated != 0).astype(int)
train_cp['hasbasement'] = (train_cp.sqft_basement != 0).astype(int)
train_cp['yr_sold'] = train_cp['date'].str[:4].astype(int)
train_cp['age'] = train_cp.yr_sold - train_cp.yr_built
train_cp['yr_after_renov'] = train_cp.yr_sold - train_cp.yr_renovated - train_cp.yr_built * (1 - train_cp.isrenovated)
train_cp.plot.scatter(x='bedrooms', y='bathrooms', alpha=0.1, s=train_cp.price*0.0001, figsize=(10,10))
plt.show()
train_cp.plot.scatter(x='long', y='lat', c='price', colormap='jet', alpha=0.3, edgecolor='grey', figsize=(10,10))
plt.show()
train_cp['lat>47.5'] = (train_cp.lat >= 47.5).astype(int)
train_cp.head()
train_cp = train_cp.drop(['date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'yr_sold'], axis=1)
train_cp = train_cp.set_index('id')
from sklearn.model_selection import train_test_split
y = train_cp.price
X = train_cp.drop('price', axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
pipelines = {
    'elsnet' : make_pipeline(StandardScaler(), ElasticNet(random_state=123)),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123)),
}
elsnet_hyper = {
'elasticnet__alpha': [0.05, 0.1, 0.5, 1, 5, 10],
'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

rf_hyper = {
'randomforestregressor__n_estimators' : [100, 200],
'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}

gb_hyper = {
'gradientboostingregressor__n_estimators': [100, 200],
'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
'gradientboostingregressor__max_depth': [1, 3, 5]
}

hyper={
    'elsnet': elsnet_hyper,
    'rf': rf_hyper,
    'gb': gb_hyper
}
from sklearn.model_selection import GridSearchCV
fitted_models = {}
for name , pipeline in pipelines.items():
    model = GridSearchCV(pipeline , hyper[name], cv=10)
    model.fit(X_train , y_train)
    fitted_models[name] = model
    print(name)
for name, model in fitted_models.items():
    print(name, model.best_score_)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
for name , model in fitted_models.items():
    pred = model.predict(X_test)
    print( name )
    print( '------------' )
    print( 'R^2:', r2_score(y_test , pred ))
    print( 'MAE:', mean_absolute_error(y_test , pred))
    print()
fitted_models['rf'].best_estimator_
