### Les Modules de Travail

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

#from pandas.plotting import scatter_matrix

import seaborn as sns
Dataset=pd.read_csv('../input/california-housing-prices/housing.csv')
Dataset.head()
Dataset.info()
df=Dataset.copy()
print(df.columns)
print(df.shape)
print(df.dtypes.value_counts())

df.dtypes.value_counts().plot.pie()
plt.figure(figsize=(20,10))

sns.heatmap(df.isna(), cbar=False)
df['ocean_proximity'].value_counts()
df['ocean_proximity'].value_counts(normalize=True)
#Conversion de la variable qualitative en categories

df=df.astype({'ocean_proximity':'category'})
print(df.dtypes.value_counts())

df.dtypes.value_counts().plot.pie()
df['rooms_per_household']=df['total_rooms']/df['households']
df['bedrooms_per_household']=df['total_bedrooms']/df['households']
df['population_per_household']=df['population']/df['households']
df.head()
df=df.drop(['longitude','latitude' ], axis=1)  
df.describe()
for col in df.select_dtypes('float'):

    #print(col)

    plt.figure()

    sns.distplot(df[col])
df['ocean_proximity'].value_counts().plot.pie()
df['ocean_proximity'].value_counts().plot(kind='bar')
sns.pairplot(df.select_dtypes('float'))
sns.heatmap(df.select_dtypes('float').corr())
corr = df.select_dtypes('float').corr()

plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1, annot=True, annot_kws={"size": 8}, square=True);
sns.clustermap(df.select_dtypes('float').corr())
df.describe()['median_house_value']
df['median_house_value'].hist(bins= 50)
plt.figure(figsize=(10,6))

sns.boxplot(data=df,x='ocean_proximity',y='median_house_value',palette='viridis')

plt.plot()
plt.figure(figsize=(10,6))



sns.stripplot(data=df,x='ocean_proximity',y='median_house_value',jitter=0.3)
from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(df, test_size=0.1, random_state=42)
X_train=trainset.drop(['median_house_value'],axis=1)

y_train=trainset['median_house_value']

print(trainset.shape)
X_test=testset.drop(['median_house_value'],axis=1)

y_test=testset['median_house_value']

print(testset.shape)
X_train.head()
X_test.head()
y_test
y_train
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder,PolynomialFeatures

from sklearn.compose import make_column_selector as selector

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

import numpy as np
X_train.columns
print(X_test.dtypes.value_counts())

X_test.dtypes.value_counts().plot.pie()
numerical_features = list(X_train.select_dtypes(include=['float64']))

categorical_features = list(X_train.select_dtypes(include=['category']))
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_features),('cat', categorical_transformer, categorical_features)])
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error
model = Pipeline(steps=[('preprocessor', preprocessor),('LinearRegression', LinearRegression())])

model.fit(X_train, y_train)

model.score(X_test, y_test)

print(model.score(X_test, y_test))

y_pred= model.predict(X_test)

y_pred= y_pred.reshape(-1,1)



print('MAE:', mean_absolute_error(y_test, y_pred))

print('MSE:', mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
TreeDecision =Pipeline(steps=[('preprocessor', preprocessor),('Decision', DecisionTreeRegressor(random_state=0))])

RandomForest = Pipeline(steps=[('preprocessor', preprocessor),('RandomForest', RandomForestRegressor(n_estimators=10,random_state=0))]) 

SVCmodel = Pipeline(steps=[('preprocessor', preprocessor),('SVR', SVR(kernel='rbf'))])

GradientRegressor =Pipeline(steps=[('preprocessor', preprocessor),('GradientRegressor', GradientBoostingRegressor(random_state=0))])

KNN = Pipeline(steps=[('preprocessor', preprocessor),('KNN', KNeighborsRegressor(n_neighbors=5))])

#SGDRegressor = Pipeline(steps=[('preprocessor', preprocessor),('SGDRegressor', SGDRegressor())])
dict_of_models = {'TreeDecision' : TreeDecision,

                  'RandomForest':RandomForest,

                  'SVCmodel':SVCmodel,

                  'GradientRegressor':GradientRegressor,

                  'KNN':KNN,

                  #'SGDRegressor':SGDRegressor

                  }
from sklearn.metrics import mean_squared_error,mean_absolute_error
def evaluation(model):

    model.fit(X_train, y_train)#apprentissage des données

    model.score(X_test, y_test) 

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    y_pred= model.predict(X_test) #prediction des valeurs

    y_pred= y_pred.reshape(-1,1)

    

    print('SCORE:',model.score(X_test, y_test))

    print('MAE:', mean_absolute_error(y_test, y_pred))

    print('MSE:', mean_squared_error(y_test, y_pred))

    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\n")

    

    plt.figure(figsize=(15,8))

    plt.scatter(X_test.iloc[:,1],y_test)

    plt.scatter(X_test.iloc[:,1],y_pred,c='r')

    plt.show()
for name, model in dict_of_models.items():

    print(name)

    evaluation(model)
param_grid = [{'n_estimators':[3,10,30],'max_features':[2,4,6,8],'max_depth':[6,8,10]}]
model_final=RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(model_final, param_grid, cv=5,scoring='neg_mean_squared_error')#initialisation

grid_search
X_train_prepared=preprocessor.fit_transform(X_train)
print(X_train_prepared.shape)

print(X_train_prepared[0])
grid_search.fit(X_train_prepared, y_train) #apprentissage du modèle
RandomForest = Pipeline(steps=[('preprocessor', preprocessor),('RandomForest', RandomForestRegressor(n_estimators=30,random_state=42,max_depth=10,max_features=8))])
RandomForest
RandomForest.fit(X_train, y_train)

RandomForest.score(X_test, y_test)

print(RandomForest.score(X_test, y_test))

y_pred= RandomForest.predict(X_test)

y_pred= y_pred.reshape(-1,1)



print('MAE:', mean_absolute_error(y_test, y_pred))

print('MSE:', mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
evaluation(RandomForest)
encoder = OneHotEncoder()
encoder.fit_transform(X_train[categorical_features])
encoded_name=encoder.get_feature_names()

encoded_name
list_features=np.append(numerical_features,encoded_name)

list_features=list(list_features)

list_features
df_features=pd.DataFrame(data=list_features,columns=['feature'])

df_features
df_features['importance']=RandomForest.steps[1][1].feature_importances_
df_features