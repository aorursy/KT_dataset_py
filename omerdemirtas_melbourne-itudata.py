import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection as ms
import geopandas as gpd
df = pd.read_csv('Melbourne_housing_FULL.csv')

df.head()
df.shape
df.dtypes
df.isnull().sum() / len(df) * 100 # percentage of missing values
ax = sns.heatmap(abs(df.corr()))
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world[world.name == 'Australia'].plot(
    color='white', edgecolor='black')

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longtitude, df.Lattitude))
minx, miny, maxx, maxy = gdf.total_bounds
ax.set_xlabel('Longtitude')
ax.set_ylabel('Lattitude')
gdf.plot(ax=ax, color='red')

plt.show()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world[world.name == 'Australia'].plot(
    color='white', edgecolor='black')

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longtitude, df.Lattitude))
ax.set_xlim(144, 146)
ax.set_ylim(-39, -37)
ax.set_xlabel('Longtitude')
ax.set_ylabel('Lattitude')
gdf.plot(ax=ax, color='red')

plt.show()
df['Date'] = pd.to_datetime(df['Date'])
df['SaleYear'] = df['Date'].dt.year
sns.countplot('SaleYear', data=df)
df.isna().any()
cat = ['Type','Method', 'Regionname','SellerG', 'CouncilArea', 'Postcode'] 
num = ['Rooms', 'Distance', 'Bathroom', 'Bedroom2', 'Car', 'Landsize', 'BuildingArea'] #'Propertycount', 'SaleYear', 'YearBuilt'
num2 =  ['Rooms', 'Distance', 'Bathroom', 'Bedroom2', 'Car', 'BuildingArea','Lattitude','Longtitude', 'Landsize' ] #'Propertycount', 'SaleYear', 'YearBuilt'
target = 'Price'
df.dropna(inplace=True)
df.drop(['Date'], axis='columns', inplace=True)
df.reset_index()
df.reset_index().isna().any()
df_cat = df[cat]
df_num = df[num]

y = df[target]

df_cat
fig, axes = plt.subplots(nrows=7,ncols=1, figsize=(12,50))

i=0

for col in df_num.columns:

    sns.boxplot(df_num[col], ax=axes[i])
    
    i += 1
outliers = []

for col in df_num.columns:
    outliers.append(df_num[col].quantile(0.75) + 1.5 * (df_num[col].quantile(0.75) - df_num[col].quantile(0.25)))
    
#outliers[7] = (df_num['YearBuilt'].quantile(0.25) - 1.5 * (df_num['YearBuilt'].quantile(0.75) - df_num['YearBuilt'].quantile(0.25)))

outliers_r = list(zip(outliers, df_num.columns))


df = df.drop(df[df['Landsize'] == 0].index)
df = df.drop(df[df['BuildingArea'] == 0].index)

for whisker, col in outliers_r:
    if col == 'YearBuilt':
        df.drop(df[df[col] < whisker].index, inplace=True)
    else:
        df.drop(df[df[col] > whisker].index, inplace=True)
        

df['Rooms'] = df['Bedroom2'] + df['Bathroom']
df['SoldAge'] = df['SaleYear'] - df['YearBuilt']
df = df[df['SoldAge'] < 50]
df = df.drop(df[df['BuildingArea'] > df['Landsize']].index)
df_cat = df[cat]
df_num = df[num2]

y = df[target]

df_catOHEnc = pd.get_dummies(df_cat)
X = pd.concat([df_num, df_catOHEnc], axis=1).reset_index()
X.drop(['index'], axis='columns', inplace=True)

print(X.isna().sum().sum())
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world[world.name == 'Australia'].plot(
    color='white', edgecolor='black')

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longtitude, df.Lattitude))
minx, miny, maxx, maxy = gdf.total_bounds
ax.set_xlabel('Longtitude')
ax.set_ylabel('Lattitude')
gdf.plot(ax=ax, color='red')

plt.show()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world[world.name == 'Australia'].plot(
    color='white', edgecolor='black')

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longtitude, df.Lattitude))
minx, miny, maxx, maxy = gdf.total_bounds
ax.set_xlabel('Longtitude')
ax.set_ylabel('Lattitude')
ax.set_xlim(144, 146)
ax.set_ylim(-39, -37)
gdf.plot(ax=ax, color='red')

plt.show()
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def predict_gnb(X_train,X_test,y_train,y_test):
    
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    
    
    y_pred = clf.predict(X_test)  
    y_true = y_test.values
    score = clf.score(X_test, y_test)
    
    return (y_true,y_pred, score)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def predict_dt(X_train,X_test,y_train,y_test):
    
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    
    y_true = y_test.values
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    
    return (y_true,y_pred, score)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def predict_knn(X_train,X_test,y_train,y_test,k=5):
    
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    
   
    y_pred = clf.predict(X_test)
    
    y_true = y_test.values
    score = clf.score(X_test, y_test)
    
    return (y_true,y_pred, score)

from sklearn.ensemble import RandomForestRegressor
def predict_rfr(X_train, X_test, y_train, y_test):
    rfmodel = RandomForestRegressor(n_estimators = 100, random_state = 7)
    rfmodel.fit(X_train, y_train)
    y_pred  = rfmodel.predict(X_test)
    y_true = y_test.values
    score = rfmodel.score(X_test,y_test)
    return (y_true, y_pred, score)
from sklearn.neighbors import KNeighborsRegressor

def predict_knreg(X_train, X_test, y_train, y_test):
    knmodel = KNeighborsRegressor(n_neighbors = 7)
    knmodel.fit(X_train, y_train)
    y_pred = knmodel.predict(X_test)
    y_true = y_test.values
    score = knmodel.score(X_test,y_test)
    return (y_true, y_pred, score)
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export 
from sklearn.model_selection import GridSearchCV

def predict_dtr(X_train, X_test, y_train, y_test):
    param_grid = {'max_depth': np.arange(3,20)}
    tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10)
    tree.fit(X_train,y_train)
    tree_final = DecisionTreeRegressor(max_depth=7)
    tree_final.fit(X_train,y_train)
    y_pred = tree_final.predict(X_test)
    y_true = y_test.values
    score = tree_final.score(X_test, y_test)
    
    return (y_true, y_pred, score)
    #print(tree.best_score_)
from sklearn import ensemble

def predict_ensemble(X_train, X_test, y_train, y_test):
    clf = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 10, min_samples_split = 2,
              learning_rate = 0.1, loss = 'ls')
    clf.fit(X_train,y_train )
    y_pred = clf.predict(X_test)
    y_true = y_test.values
    score = clf.score(X_test,y_test)
    return (y_true, y_pred,score)
    

from xgboost import XGBRegressor
def predict_xgb(X_train, X_test, y_train, y_test, booster= 'gbtree', n_jobs=2):
    xgb = XGBRegressor(booster = booster , n_jobs = n_jobs, n_estimators=1000, learning_rate=0.05, nthread=10)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_true = y_test.values
    score = xgb.score(X_test, y_test)
    
    return (y_true, y_pred, score)
def sigma(x):
    result = x[0] - x[1]
    result = result ** 2
    result = (result.sum() / len(x[0]))**(1/2)
    
    return result
print('K Neighbour Classifier')
print('RMSE', sigma(predict_knn(X_train,X_test,y_train,y_test,k=1)))
print('Score', predict_knn(X_train,X_test,y_train,y_test,k=1)[2])

print('K Neighbour Classifier')
print('RMSE', sigma(predict_knn(X_train,X_test,y_train,y_test,k=2)))
print('Score', predict_knn(X_train,X_test,y_train,y_test,k=2)[2])
print('K Neighbour Classifier')
print('RMSE', sigma(predict_knn(X_train,X_test,y_train,y_test,k=3)))
print('Score', predict_knn(X_train,X_test,y_train,y_test,k=3)[2])
print('Gaussian Naive Bayes')
print('RMSE', sigma(predict_gnb(X_train,X_test,y_train,y_test)))
print('Score', predict_gnb(X_train,X_test,y_train,y_test)[2])
print('Decision Tree Classifier')
print('RMSE', sigma(predict_dt(X_train,X_test,y_train,y_test)))
print('Score', predict_dt(X_train,X_test,y_train,y_test)[2])
print('K Neighbors Regressor')
print('RMSE', sigma(predict_knreg(X_train, X_test, y_train, y_test)))
print('Score', predict_knreg(X_train, X_test, y_train, y_test)[2])
print('Decision Tree Regressor')
print('RMSE', sigma(predict_dtr(X_train, X_test, y_train, y_test)))
print('Score', predict_dtr(X_train, X_test, y_train, y_test)[2])
print('Random Forest Regressor')
print('RMSE', sigma(predict_rfr(X_train, X_test, y_train, y_test)))
print('Score', predict_rfr(X_train, X_test, y_train, y_test)[2])
print('ensemble')
print('RMSE', sigma(predict_ensemble(X_train, X_test, y_train, y_test)))
print('Score', predict_ensemble(X_train, X_test, y_train, y_test)[2] )
print('XGBoost')
print('RMSE', sigma(predict_xgb(X_train, X_test, y_train, y_test, booster= 'gbtree', n_jobs=2)))
print('Score', predict_xgb(X_train, X_test, y_train, y_test, booster= 'gbtree', n_jobs=2)[2])