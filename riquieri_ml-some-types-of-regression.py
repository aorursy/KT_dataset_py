import pandas as pd
csv_path = "../input/auto85/auto.csv"

df = pd.read_csv(csv_path, na_values = '?', header=None)

header = ['Symbolyng','Normalized-losses','Make','Fuel-Type','Aspiration','Num-of-doors','Body-style','Drive-wheels',
         'Engine-location', 'Whell-base', 'Lenght', 'Width', 'Weight','Curb-weight','Engine-type','Num-of-cylinders',
         'Engine-size','Fuel-system','Bore','Stroke','Compression-ratio','Horsepower','Peak-rpm','City-mpg','Highway-mpg',
         'Price']

df.columns=header 

pd.set_option('display.max_columns', None)
df.head()
is_null = df.isnull().sum().sort_values(ascending=False)
percent = ((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False))*100

missing_data = pd.concat([is_null, percent], axis=1, keys=['Total', 'Percent'])
missing_data
df['Normalized-losses']. fillna(df['Normalized-losses'].mean(), inplace=True)
df['Horsepower'].fillna(df['Horsepower'].mean(), inplace=True)
df.dropna(subset=['Price'], axis=0, inplace=True)
df.dropna(subset=['Stroke'], axis=0, inplace=True)
df.dropna(subset=['Bore'], axis=0, inplace=True)
df.dropna(subset=['Peak-rpm'], axis=0, inplace=True)
df.dropna(subset=['Num-of-doors'], axis=0, inplace=True)
is_null = df.isnull().sum().sort_values(ascending=False)
percent = ((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False))*100

missing_data = pd.concat([is_null, percent], axis=1, keys=['Total', 'Percent'])
missing_data
encode = pd.get_dummies(df[['Make','Fuel-Type','Aspiration','Num-of-doors','Body-style','Drive-wheels','Engine-location','Engine-type','Num-of-cylinders','Fuel-system']])
df = pd.concat([df, encode], axis=1)
df.drop(['Make','Fuel-Type','Aspiration','Num-of-doors','Body-style','Drive-wheels','Engine-location','Engine-type','Num-of-cylinders','Fuel-system'], axis=1, inplace=True)
df.head()
x = df.drop(['Price'],axis=1)
y = df['Price']
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(x_treino,y_treino)
result =linear_model.score(x_teste,y_teste)
print(result)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

linear_model = LinearRegression()
kfold = KFold(n_splits=5,shuffle=True)

result= cross_val_score(linear_model,x,y,cv=kfold)
print(result.mean())
from sklearn.linear_model import Ridge

ridge_model = Ridge()
ridge_model.fit(x_treino,y_treino)
result =ridge_model.score(x_teste,y_teste)
print(result)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

ridge_model = Ridge()
kfold = KFold(n_splits=5,shuffle=True)

result= cross_val_score(ridge_model,x,y,cv=kfold)
print(result.mean())
from sklearn.linear_model import Lasso

lasso_model = Lasso(max_iter=5000)
lasso_model.fit(x_treino,y_treino)
result =lasso_model.score(x_teste,y_teste)
print(result)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

lasso_model = Lasso(max_iter=5000)
kfold = KFold(n_splits=5,shuffle=True,)

result= cross_val_score(lasso_model,x,y,cv=kfold)
print(result.mean())
from sklearn.linear_model import ElasticNet

elastic_model= ElasticNet(alpha=750, l1_ratio=0.9)
elastic_model.fit(x_treino,y_treino)
result =elastic_model.score(x_teste,y_teste)
print(result)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

elastic_model = ElasticNet(max_iter=5000)
kfold = KFold(n_splits=5,shuffle=True,)

result= cross_val_score(elastic_model,x,y,cv=kfold)
print(result.mean())
from sklearn.tree import DecisionTreeRegressor

decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(x_treino,y_treino)
result =decision_tree_model.score(x_teste,y_teste)
print(result)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

decision_tree_model = DecisionTreeRegressor()
kfold = KFold(n_splits=5,shuffle=True,)

result= cross_val_score(decision_tree_model,x,y,cv=kfold)
print(result.mean())
