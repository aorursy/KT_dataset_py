import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from sklearn import neighbors
dataset = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")

dataset.head()
dataset.describe()
dataset.info()
dataset.isna().sum()
dataset = dataset[dataset.cuisines.isna() == False]
dataset.isna().sum()
dataset.columns
dataset.drop(["url","phone","address","listed_in(city)"], axis = 1, inplace=True)
dataset.columns
dataset.rename(columns ={'approx_cost(for two people)': 'avg_cost'}, inplace=True)

dataset.rename(columns ={'listed_in(type)': 'listed_type'}, inplace=True)
dataset.name.value_counts().head()
# Plot Restaurant Names vs No of locations

plt.figure(figsize = (10,5))

ax = dataset.name.value_counts()[:20].plot(kind = 'bar')

plt.xlabel("Restaurant Name")

plt.ylabel("No. of restaurants")

plt.title('Restaurant Names vs No of locations')
dataset.online_order.value_counts()
# Plot Online vs Offline Orders

plt.figure(figsize=(10,5))

ax = dataset.online_order.value_counts().plot(kind = 'bar')

plt.xlabel("Online/Offline Orders")

plt.ylabel("Count")

plt.title("Online/Offline Orders Count")
dataset.book_table.value_counts()
# Plot Book Table Facility Counts

plt.figure(figsize=(10,5))

ax = dataset.book_table.value_counts().plot(kind = 'bar')

plt.xlabel("Book Table Facility")

plt.ylabel("Count")

plt.title("Book Table Facility Counts")
dataset.location.value_counts().head()
# Plot location with highest no of restaurants

plt.figure(figsize=(10,10))

ax = dataset.location.value_counts()[:15].plot(kind = 'pie')

plt.title("location with highest no of restaurant counts")

plt.legend()
plt.figure(figsize=(10,10))

names = dataset.location.value_counts()[:15].index

values = dataset.location.value_counts()[:15].values

explode = [0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



plt.pie(values, explode=explode, autopct='%0.1f%%', shadow=True, labels = names)

plt.title("Percentage of restaurants present in that location")

plt.show()
dataset.rest_type.value_counts().head()
# Plot highest no of restaurant types

plt.figure(figsize=(10,10))

names = dataset.rest_type.value_counts()[:15].index

values = dataset.rest_type.value_counts()[:15].values

explode = [0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



plt.pie(values, explode=explode, autopct='%0.1f%%', shadow=True, labels = names)

plt.title("Percentage of restaurants types")

plt.show()
dataset.rate.value_counts().head()
dataset = dataset[dataset.rest_type.isna()==False]

dataset = dataset[dataset.rate.isna()==False]

dataset = dataset[dataset.rate != 'NEW']

dataset = dataset[dataset.rate != '-']

dataset['rate'] = dataset['rate'].apply(lambda r: r.replace('/5', ''))

dataset['rate'] = dataset['rate'].apply(lambda r: float(r))

dataset.rate.value_counts().head()
f,ax=plt.subplots(figsize=(18,8))

g = sns.pointplot(x=dataset["rest_type"], y=dataset["rate"], data=dataset)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.title('Restaurent type vs Rate', weight = 'bold')

plt.show()
dataset.avg_cost.value_counts().head()
plt.figure(figsize=(10,10))

name = dataset.avg_cost.value_counts()[:15].index

values = dataset.avg_cost.value_counts()[:15].values

explode = [0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



plt.pie(values, explode=explode, labels=name, autopct='%01.f%%', shadow=True)
dataset.dish_liked.value_counts().head(20)
dataset_dish_liked = dataset[dataset.dish_liked.notnull()]

dataset_dish_liked.dish_liked = dataset_dish_liked.dish_liked.apply(lambda x:x.lower().strip())



liked_dish_count=[]

for dishes in dataset_dish_liked.dish_liked:

    for dish in dishes.split(','):

        liked_dish_count.append(dish.strip())

        

pd.Series(liked_dish_count).value_counts().head()
plt.figure(figsize=(10,5))

ax = pd.Series(liked_dish_count).value_counts()[:20].plot(kind = 'bar')

plt.xlabel("Dishes")

plt.ylabel("Count")

plt.title("Most liked Dishes Count")
#Encode the input Variables

def Encode(dataset):

    for column in dataset.columns[~dataset.columns.isin(['rate', 'votes'])]:

        dataset[column] = dataset[column].factorize()[0]

    return dataset



dataset_copy = Encode(dataset.copy())
plt.figure(figsize=(20,10))

sns.heatmap(data=dataset_copy.corr(), cmap="seismic")

plt.show()
dataset.drop(["dish_liked","reviews_list","menu_item"], axis = 1, inplace = True)
dataset.head()
dataset.isna().sum()
dataset.shape
dataset['avg_cost'] = dataset['avg_cost'].str.replace(',','')

dataset['avg_cost'] = dataset['avg_cost'].astype('float64')

dataset.info()
dataset_not_na_avg_cost = dataset.groupby("listed_type")['avg_cost'].transform('mean')

dataset['avg_cost'].fillna(dataset_not_na_avg_cost, inplace =True)

dataset.info()
dataset.head()
dataset['rest_type'] = dataset['rest_type'].str.replace(',','')

dataset['rest_type'] = dataset['rest_type'].astype('str').apply(lambda x: ' '.join(sorted(x.split())))
dataset['cuisines'] = dataset['cuisines'].str.replace(',','')

dataset['cuisines'] = dataset['cuisines'].astype('str').apply(lambda x: ' '.join(sorted(x.split())))
dataset.head()
dataset['online_order'] = pd.get_dummies(dataset['online_order'])

dataset['book_table'] = pd.get_dummies(dataset['book_table'])
dataset.head()
dataset_location = pd.get_dummies(dataset['location'])

dataset_rest_type = pd.get_dummies(dataset['rest_type'])

dataset_cuisines = pd.get_dummies(dataset['cuisines'])

dataset_listedin_type = pd.get_dummies(dataset['listed_type'])
dataset_final = pd.concat([dataset, dataset_cuisines, dataset_location, dataset_listedin_type, dataset_rest_type], axis=1)
dataset_final.head()
dataset_final.drop(["location","rest_type","cuisines","listed_type"], axis = 1, inplace=True)

dataset_final.head()
X = dataset_final.drop(['name','rate'], axis = 1)

y = dataset_final['rate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
ridge = Ridge()

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
lasso = Lasso()

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
random_forest = RandomForestRegressor(random_state=0, n_estimators=100)

random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
decision_tree = DecisionTreeRegressor()

decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
svr = SVR()

svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
from sklearn import neighbors

params = {'n_neighbors':[2,3,4,5,6,7,8,9]}



knn = neighbors.KNeighborsRegressor()



model = GridSearchCV(knn, params, cv=5)

model.fit(X_train, y_train)

model.best_params_
knn=KNeighborsRegressor(n_neighbors = 2)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
etree=ExtraTreesRegressor(n_estimators = 100)

etree.fit(X_train, y_train)
y_pred = etree.predict(X_test)

print(r2_score(y_test, y_pred, multioutput='uniform_average'))
etree_df = pd.DataFrame(y_test, columns=['Actual'])

etree_df['Predicted'] = etree.predict(X_test)

etree_df.corr()
etree_df.head(40)
plt.figure(figsize=(20,10))

plt.plot(range(len(etree_df['Actual'].head(500))), etree_df['Actual'].head(500), color = "red")

plt.plot(range(len(etree_df['Predicted'].head(500))), etree_df['Predicted'].head(500), color = "blue")

plt.xlabel("Range")

plt.ylabel("Rating")

plt.title("Restaurant Rating Actual vs Predicted")

plt.legend()
feature_importance_df = pd.DataFrame(X_train.columns, columns=["Feature"])

feature_importance_df["Importance"] = etree.feature_importances_

feature_importance_df.sort_values('Importance', ascending=False, inplace=True)

feature_importance_df = feature_importance_df.head(20)

feature_importance_df
plt.figure(figsize=(15,5))

ax = feature_importance_df['Feature']

plt.bar(range(feature_importance_df.shape[0]), feature_importance_df['Importance']*100)

plt.xticks(range(feature_importance_df.shape[0]), feature_importance_df['Feature'], rotation = 20)

plt.xlabel("Features")

plt.ylabel("Importance")

plt.title("Plot Feature Importances")