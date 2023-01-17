import pandas as pd
import numpy as np
import os
%matplotlib inline

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
# change this if needed
PATH_TO_DATA = '../input'
full_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'housing.csv'))
print(full_df.shape)
full_df.head()
%%time
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(full_df,shuffle = True, test_size = 0.25, random_state=17)
train_df=train_df.copy()
test_df=test_df.copy()
print(train_df.shape)
print(test_df.shape)
train_df.describe()
train_df.info()
train_df[pd.isnull(train_df).any(axis=1)].head(10)
numerical_features=list(train_df.columns)
numerical_features.remove('ocean_proximity')
numerical_features.remove('median_house_value')
print(numerical_features)
train_df['median_house_value'].hist()
max_target=train_df['median_house_value'].max()
print("The largest median value:",max_target)
print("The # of values, equal to the largest:", sum(train_df['median_house_value']==max_target))
print("The % of values, equal to the largest:", sum(train_df['median_house_value']==max_target)/train_df.shape[0])
min_target=train_df['median_house_value'].min()
print("The smallest median value:",min_target)
print("The # of values, equal to the smallest:", sum(train_df['median_house_value']==min_target))
print("The % of values, equal to the smallest:", sum(train_df['median_house_value']==min_target)/train_df.shape[0])
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

qqplot(train_df['median_house_value'], line='s')
pyplot.show()
from scipy.stats import normaltest

stat, p = normaltest(train_df['median_house_value'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
target_log=np.log1p(train_df['median_house_value'])
qqplot(target_log, line='s')
pyplot.show()
stat, p = normaltest(target_log)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
train_df['median_house_value_log']=np.log1p(train_df['median_house_value'])
test_df['median_house_value_log']=np.log1p(test_df['median_house_value'])
train_df[numerical_features].hist(bins=50, figsize=(10, 10))
skewed_features=['households','median_income','population', 'total_bedrooms', 'total_rooms']
log_numerical_features=[]
for f in skewed_features:
    train_df[f + '_log']=np.log1p(train_df[f])
    test_df[f + '_log']=np.log1p(test_df[f])
    log_numerical_features.append(f + '_log')
train_df[log_numerical_features].hist(bins=50, figsize=(10, 10))
max_house_age=train_df['housing_median_age'].max()
print("The largest value:",max_house_age)
print("The # of values, equal to the largest:", sum(train_df['housing_median_age']==max_house_age))
print("The % of values, equal to the largest:", sum(train_df['housing_median_age']==max_house_age)/train_df.shape[0])
train_df['age_clipped']=train_df['housing_median_age']==max_house_age
test_df['age_clipped']=test_df['housing_median_age']==max_house_age
import matplotlib.pyplot as plt
import seaborn as sns

corr_y = pd.DataFrame(train_df).corr()
plt.rcParams['figure.figsize'] = (20, 16)  # Размер картинок
sns.heatmap(corr_y, 
            xticklabels=corr_y.columns.values,
            yticklabels=corr_y.columns.values, annot=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin = LinearRegression()

# we will train our model based on all numerical non-target features with not NaN total_bedrooms
appropriate_columns = train_df.drop(['median_house_value','median_house_value_log',
                            'ocean_proximity', 'total_bedrooms_log'],axis=1)
train_data=appropriate_columns[~pd.isnull(train_df).any(axis=1)]

# model will be validated on 25% of train dataset 
# theoretically we can use even our test_df dataset (as we don't use target) for this task, but we will not
temp_train, temp_valid = train_test_split(train_data,shuffle = True, test_size = 0.25, random_state=17)

lin.fit(temp_train.drop(['total_bedrooms'],axis=1), temp_train['total_bedrooms'])
np.sqrt(mean_squared_error(lin.predict(temp_valid.drop(['total_bedrooms'],axis=1)),
                           temp_valid['total_bedrooms']))
np.sqrt(mean_squared_error(np.ones(len(temp_valid['total_bedrooms']))*temp_train['total_bedrooms'].mean(),
                           temp_valid['total_bedrooms']))
lin.fit(train_data.drop(['total_bedrooms'],axis=1), train_data['total_bedrooms'])

train_df['total_bedrooms_is_nan']=pd.isnull(train_df).any(axis=1).astype(int)
test_df['total_bedrooms_is_nan']=pd.isnull(test_df).any(axis=1).astype(int)

train_df['total_bedrooms'].loc[pd.isnull(train_df).any(axis=1)]=\
lin.predict(train_df.drop(['median_house_value','median_house_value_log','total_bedrooms','total_bedrooms_log',
               'ocean_proximity','total_bedrooms_is_nan'],axis=1)[pd.isnull(train_df).any(axis=1)])

test_df['total_bedrooms'].loc[pd.isnull(test_df).any(axis=1)]=\
lin.predict(test_df.drop(['median_house_value','median_house_value_log','total_bedrooms','total_bedrooms_log',
               'ocean_proximity','total_bedrooms_is_nan'],axis=1)[pd.isnull(test_df).any(axis=1)])

#linear regression can lead to negative predictions, let's change it
test_df['total_bedrooms']=test_df['total_bedrooms'].apply(lambda x: max(x,0))
train_df['total_bedrooms']=train_df['total_bedrooms'].apply(lambda x: max(x,0))
train_df['total_bedrooms_log']=np.log1p(train_df['total_bedrooms'])
test_df['total_bedrooms_log']=np.log1p(test_df['total_bedrooms'])
print(train_df.info())
print(test_df.info())
sns.set()
sns.pairplot(train_df[log_numerical_features+['median_house_value_log']])
sns.set()
local_coord=[-122, 41] # the point near which we want to look at our variables
euc_dist_th = 2 # distance treshhold

euclid_distance=train_df[['latitude','longitude']].apply(lambda x:
                                                         np.sqrt((x['longitude']-local_coord[0])**2+
                                                                 (x['latitude']-local_coord[1])**2), axis=1)

# indicate wethere the point is within treshhold or not
indicator=pd.Series(euclid_distance<=euc_dist_th, name='indicator')

print("Data points within treshhold:", sum(indicator))

# a small map to visualize th eregion for analysis
sns.lmplot('longitude', 'latitude', data=pd.concat([train_df,indicator], axis=1), hue='indicator', markers ='.', fit_reg=False, height=5)

# pairplot
sns.pairplot(train_df[log_numerical_features+['median_house_value_log']][indicator])
sns.lmplot('longitude', 'latitude', data=train_df,markers ='.', hue='ocean_proximity', fit_reg=False, height=5)
plt.show()
value_count=train_df['ocean_proximity'].value_counts()
value_count
plt.figure(figsize=(12,5))


sns.barplot(value_count.index, value_count.values)
plt.title('Ocean Proximity')
plt.ylabel('Number of Occurrences')
plt.xlabel('Ocean Proximity')

plt.figure(figsize=(12,5))
plt.title('House Value depending on Ocean Proximity')
sns.boxplot(x="ocean_proximity", y="median_house_value_log", data=train_df)
ocean_proximity_dummies = pd.get_dummies(pd.concat([train_df['ocean_proximity'],test_df['ocean_proximity']]),
                                         drop_first=True)
dummies_names=list(ocean_proximity_dummies.columns)
train_df=pd.concat([train_df,ocean_proximity_dummies[:train_df.shape[0]]], axis=1 )
test_df=pd.concat([test_df,ocean_proximity_dummies[train_df.shape[0]:]], axis=1 )

train_df=train_df.drop(['ocean_proximity'], axis=1)
test_df=test_df.drop(['ocean_proximity'], axis=1)
train_df.head()
train_df[['longitude','latitude']].describe()
from matplotlib.colors import LinearSegmentedColormap

plt.figure(figsize=(10,10))

cmap = LinearSegmentedColormap.from_list(name='name', colors=['green','yellow','red'])

f, ax = plt.subplots()
points = ax.scatter(train_df['longitude'], train_df['latitude'], c=train_df['median_house_value_log'],
                    s=10, cmap=cmap)
f.colorbar(points)
sf_coord=[-122.4194, 37.7749]
la_coord=[-118.2437, 34.0522]

train_df['distance_to_SF']=np.sqrt((train_df['longitude']-sf_coord[0])**2+(train_df['latitude']-sf_coord[1])**2)
test_df['distance_to_SF']=np.sqrt((test_df['longitude']-sf_coord[0])**2+(test_df['latitude']-sf_coord[1])**2)

train_df['distance_to_LA']=np.sqrt((train_df['longitude']-la_coord[0])**2+(train_df['latitude']-la_coord[1])**2)
test_df['distance_to_LA']=np.sqrt((test_df['longitude']-la_coord[0])**2+(test_df['latitude']-la_coord[1])**2)
from sklearn.preprocessing import StandardScaler

features_to_scale=numerical_features+log_numerical_features+['distance_to_SF','distance_to_LA']

scaler = StandardScaler()

X_train_scaled=pd.DataFrame(scaler.fit_transform(train_df[features_to_scale]),
                            columns=features_to_scale, index=train_df.index)
X_test_scaled=pd.DataFrame(scaler.transform(test_df[features_to_scale]),
                           columns=features_to_scale, index=test_df.index)
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10, random_state=17, shuffle=True)
from sklearn.linear_model import Ridge

model=Ridge(alpha=1)
X=train_df[numerical_features+dummies_names]
y=train_df['median_house_value']
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
# using scaled data
X=pd.concat([train_df[dummies_names], X_train_scaled[numerical_features]], axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
# adding NaN indicating feature
X=pd.concat([train_df[dummies_names+['total_bedrooms_is_nan']],
             X_train_scaled[numerical_features]], axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
# adding house age cliiping indicating feature
X=pd.concat([train_df[dummies_names+['age_clipped']],
             X_train_scaled[numerical_features]], axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
# adding log features
X=pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled[numerical_features+log_numerical_features]],
            axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
# adding city distance features
X=pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled],
            axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
sns.set()
sns.pairplot(train_df[['distance_to_SF','distance_to_LA','median_house_value_log']])
new_features_train_df=pd.DataFrame(index=train_df.index)
new_features_test_df=pd.DataFrame(index=test_df.index)


new_features_train_df['1/distance_to_SF']=1/(train_df['distance_to_SF']+0.001)
new_features_train_df['1/distance_to_LA']=1/(train_df['distance_to_LA']+0.001)
new_features_train_df['log_distance_to_SF']=np.log1p(train_df['distance_to_SF'])
new_features_train_df['log_distance_to_LA']=np.log1p(train_df['distance_to_LA'])

new_features_test_df['1/distance_to_SF']=1/(test_df['distance_to_SF']+0.001)
new_features_test_df['1/distance_to_LA']=1/(test_df['distance_to_LA']+0.001)
new_features_test_df['log_distance_to_SF']=np.log1p(test_df['distance_to_SF'])
new_features_test_df['log_distance_to_LA']=np.log1p(test_df['distance_to_LA'])
new_features_train_df['rooms/person']=train_df['total_rooms']/train_df['population']
new_features_train_df['rooms/household']=train_df['total_rooms']/train_df['households']

new_features_test_df['rooms/person']=test_df['total_rooms']/test_df['population']
new_features_test_df['rooms/household']=test_df['total_rooms']/test_df['households']


new_features_train_df['bedrooms/person']=train_df['total_bedrooms']/train_df['population']
new_features_train_df['bedrooms/household']=train_df['total_bedrooms']/train_df['households']

new_features_test_df['bedrooms/person']=test_df['total_bedrooms']/test_df['population']
new_features_test_df['bedrooms/household']=test_df['total_bedrooms']/test_df['households']
new_features_train_df['bedroom/rooms']=train_df['total_bedrooms']/train_df['total_rooms']
new_features_test_df['bedroom/rooms']=test_df['total_bedrooms']/test_df['total_rooms']
new_features_train_df['average_size_of_household']=train_df['population']/train_df['households']
new_features_test_df['average_size_of_household']=test_df['population']/test_df['households']
new_features_train_df=pd.DataFrame(scaler.fit_transform(new_features_train_df),
                            columns=new_features_train_df.columns, index=new_features_train_df.index)

new_features_test_df=pd.DataFrame(scaler.transform(new_features_test_df),
                            columns=new_features_test_df.columns, index=new_features_test_df.index)
new_features_train_df.head()
new_features_test_df.head()
# computing current best score

X=pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled],
            axis=1, ignore_index = True)

cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
best_score = np.sqrt(-cv_scores.mean())
print("Best score: ", best_score)

# list of the new good features
new_features_list=[]

for feature in new_features_train_df.columns:
    new_features_list.append(feature)
    X=pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled,
                 new_features_train_df[new_features_list]
                ],
                axis=1, ignore_index = True)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    score = np.sqrt(-cv_scores.mean())
    if score >= best_score:
        new_features_list.remove(feature)
        print(feature, ' is not a good feature')
    else:
        print(feature, ' is a good feature')
        print('New best score: ', score)
        best_score=score
X=pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled,
             new_features_train_df[new_features_list]
            ],
            axis=1).reset_index(drop=True)
y=train_df['median_house_value'].reset_index(drop=True)
from sklearn.metrics import mean_squared_error

def cross_val_score_with_log(model=model, X=X,y=y,kf=kf, use_log=False):

    X_temp=np.array(X)

    # if use_log parameter is true we will predict log(y+1)
    if use_log:
        y_temp=np.log1p(y)
    else:
        y_temp=np.array(y)
    
    cv_scores=[]
    for train_index, test_index in kf.split(X_temp,y_temp):

        prediction = model.fit(X_temp[train_index], y_temp[train_index]).predict(X_temp[test_index])
        
        # if use_log parameter is true we should come back to the initial targer
        if use_log:
            prediction=np.expm1(prediction)
        cv_scores.append(-mean_squared_error(y[test_index],prediction))

    return np.sqrt(-np.mean(cv_scores))
cross_val_score_with_log(X=X,y=y,kf=kf, use_log=False)
cross_val_score_with_log(X=X,y=y,kf=kf, use_log=True)
from sklearn.model_selection import validation_curve

Cs=np.logspace(-5, 4, 10)
train_scores, valid_scores = validation_curve(model, X, y, "alpha", 
                                              Cs, cv=kf, scoring='neg_mean_squared_error')

plt.plot(Cs, np.sqrt(-train_scores.mean(axis=1)), 'ro-')

plt.fill_between(x=Cs, y1=np.sqrt(-train_scores.max(axis=1)), 
                 y2=np.sqrt(-train_scores.min(axis=1)), alpha=0.1, color = "red")


plt.plot(Cs, np.sqrt(-valid_scores.mean(axis=1)), 'bo-')

plt.fill_between(x=Cs, y1=np.sqrt(-valid_scores.max(axis=1)), 
                 y2=np.sqrt(-valid_scores.min(axis=1)), alpha=0.1, color = "blue")

plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title('Regularization Parameter Tuning')

plt.show()
Cs[np.sqrt(-valid_scores.mean(axis=1)).argmin()]
from sklearn.model_selection import learning_curve

model=Ridge(alpha=1.0)

train_sizes, train_scores, valid_scores = learning_curve(model, X, y, train_sizes=list(range(50,10001,100)),
                                                         scoring='neg_mean_squared_error', cv=5)

plt.plot(train_sizes, np.sqrt(-train_scores.mean(axis=1)), 'ro-')

plt.fill_between(x=train_sizes, y1=np.sqrt(-train_scores.max(axis=1)), 
                 y2=np.sqrt(-train_scores.min(axis=1)), alpha=0.1, color = "red")

plt.plot(train_sizes, np.sqrt(-valid_scores.mean(axis=1)), 'bo-')

plt.fill_between(x=train_sizes, y1=np.sqrt(-valid_scores.max(axis=1)), 
                 y2=np.sqrt(-valid_scores.min(axis=1)), alpha=0.1, color = "blue")

plt.xlabel('Train size')
plt.ylabel('RMSE')
plt.title('Regularization Parameter Tuning')

plt.show()
X.columns
features_for_trees=['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'age_clipped',
       'longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'distance_to_SF', 'distance_to_LA','bedroom/rooms']       
%%time
from sklearn.ensemble import RandomForestRegressor

X_trees=X[features_for_trees]

model_rf=RandomForestRegressor(n_estimators=100, random_state=17)
cv_scores = cross_val_score(model_rf, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

print(np.sqrt(-cv_scores.mean()))
from sklearn.model_selection import GridSearchCV

param_grid={'n_estimators': [100],
            'max_depth':  [22, 23, 24, 25],
            'max_features': [5,6,7,8]}

gs=GridSearchCV(model_rf, param_grid, scoring='neg_mean_squared_error', fit_params=None, n_jobs=-1, cv=kf, verbose=1)

gs.fit(X_trees,y)
print(np.sqrt(-gs.best_score_))
gs.best_params_
best_depth=gs.best_params_['max_depth']
best_features=gs.best_params_['max_features']
%%time
model_rf=RandomForestRegressor(n_estimators=100, max_depth=best_depth, max_features=best_features, random_state=17)
cv_scores = cross_val_score(model_rf, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

print(np.sqrt(-cv_scores.mean()))
model_rf=RandomForestRegressor(n_estimators=200,  max_depth=best_depth, max_features=best_features, random_state=17)
Cs=list(range(20,201,20))
train_scores, valid_scores = validation_curve(model_rf, X_trees, y, "n_estimators", 
                                              Cs, cv=kf, scoring='neg_mean_squared_error')

plt.plot(Cs, np.sqrt(-train_scores.mean(axis=1)), 'ro-')

plt.fill_between(x=Cs, y1=np.sqrt(-train_scores.max(axis=1)), 
                 y2=np.sqrt(-train_scores.min(axis=1)), alpha=0.1, color = "red")


plt.plot(Cs, np.sqrt(-valid_scores.mean(axis=1)), 'bo-')

plt.fill_between(x=Cs, y1=np.sqrt(-valid_scores.max(axis=1)), 
                 y2=np.sqrt(-valid_scores.min(axis=1)), alpha=0.1, color = "blue")

plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title('Regularization Parameter Tuning')

plt.show()
# uncomment to install if you have not yet
#!pip install lightgbm
%%time
from lightgbm.sklearn import LGBMRegressor

model_gb=LGBMRegressor()
cv_scores = cross_val_score(model_gb, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=1)

print(np.sqrt(-cv_scores.mean()))
gs
# model complexity optimization
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_grid={'max_depth':  randint(6,11),
            'num_leaves': randint(7,127),
            'reg_lambda': np.logspace(-3,0,100),
            'random_state': [17]}

gs=RandomizedSearchCV(model_gb, param_grid, n_iter = 50, scoring='neg_mean_squared_error', fit_params=None, 
                n_jobs=-1, cv=kf, verbose=1, random_state=17)

gs.fit(X_trees,y)
np.sqrt(-gs.best_score_)
gs.best_params_
# model convergency optimization

param_grid={'n_estimators': [500],
            'learning_rate': np.logspace(-4, 0, 100),
            'max_depth':  [10],
            'num_leaves': [72],
            'reg_lambda': [0.0010722672220103231],
            'random_state': [17]}

gs=RandomizedSearchCV(model_gb, param_grid, n_iter = 20, scoring='neg_mean_squared_error', fit_params=None, 
                n_jobs=-1, cv=kf, verbose=1, random_state=17)

gs.fit(X_trees,y)
np.sqrt(-gs.best_score_)
gs.best_params_
results_df=pd.DataFrame(columns=['model','CV_results', 'holdout_results'])
# hold-out features and target 
X_ho=pd.concat([test_df[dummies_names+['age_clipped']], X_test_scaled,
             new_features_test_df[new_features_list]],axis=1).reset_index(drop=True)
y_ho=test_df['median_house_value'].reset_index(drop=True)

X_trees_ho=X_ho[features_for_trees]
%%time

#linear model
model=Ridge(alpha=1.0)

cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
score_cv=np.sqrt(-np.mean(cv_scores.mean()))


prediction_ho = model.fit(X, y).predict(X_ho)
score_ho=np.sqrt(mean_squared_error(y_ho,prediction_ho))

results_df.loc[results_df.shape[0]]=['Linear Regression',  score_cv,  score_ho]
%%time

#Random Forest
model_rf=RandomForestRegressor(n_estimators=200,  max_depth=23, max_features=5, random_state=17)

cv_scores = cross_val_score(model_rf, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
score_cv=np.sqrt(-np.mean(cv_scores.mean()))


prediction_ho = model_rf.fit(X_trees, y).predict(X_trees_ho)
score_ho=np.sqrt(mean_squared_error(y_ho,prediction_ho))

results_df.loc[results_df.shape[0]]=['Random Forest',  score_cv,  score_ho]
%%time

#Gradient boosting
model_gb=LGBMRegressor(reg_lambda=0.0010722672220103231, max_depth=10,
                       n_estimators=500, num_leaves=72, random_state=17, learning_rate=0.06734150657750829)
cv_scores = cross_val_score(model_gb, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
score_cv=np.sqrt(-np.mean(cv_scores.mean()))

prediction_ho = model_gb.fit(X_trees, y).predict(X_trees_ho)
score_ho=np.sqrt(mean_squared_error(y_ho,prediction_ho))

results_df.loc[results_df.shape[0]]=['Gradient boosting',  score_cv,  score_ho]
results_df