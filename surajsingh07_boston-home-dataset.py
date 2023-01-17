# Libraries for Data Analysis

import numpy as np

import pandas as pd

# Libraries for Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Loading dataset from sklearn

from sklearn.datasets import load_boston

# Making the instance of the class

boston = load_boston()

print(boston.keys())

print(boston.DESCR)
# Creating the feature dataframe

df_feat = pd.DataFrame(boston.data, columns=boston.feature_names)

# Creating the target dataframe which is 'price'

df_target = pd.DataFrame(boston.target, columns=['PRICE'])

# Concatinating the two dataframe into one for data exploration

df = pd.concat([df_feat, df_target], axis=1)

df.head()
df.info()
df.describe()
# Checking the distribution of varibales

df.hist(figsize=(20,15), edgecolor='black')

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(20,8))

matrix = np.triu(df.corr())

sns.heatmap(df.corr(), mask=matrix, annot=True, cmap='viridis',cbar_kws = dict(use_gridspec=False,location="top"), ax=axs[0])

axs[0].set_title('Correlation matrix')

axs[1].barh(df.columns, df.corrwith(df.PRICE))

axs[1].set_title('features correlation with price')

plt.show()
sns.set_style('whitegrid')

sns.jointplot(x='RM', y='PRICE', data=df, color='#007b7f', edgecolor='black')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('PRICE',axis=1), df.PRICE, test_size=0.20)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(df_scaled[:,:-1], df_scaled[:,-1], test_size=0.20)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor
knn = KNeighborsRegressor().fit(X_train_scaled, y_train_scaled)

knn_score = knn.score(X_test_scaled, y_test_scaled)



lr = LinearRegression().fit(X_train, y_train)

lr_score = lr.score(X_test, y_test)



ridge = Ridge().fit(X_train, y_train)

ridge_score = ridge.score(X_test, y_test)



lasso = Lasso().fit(X_train, y_train)

lasso_score = lasso.score(X_test, y_test)



tree = DecisionTreeRegressor().fit(X_train, y_train)

tree_score = tree.score(X_test, y_test)



forest = RandomForestRegressor().fit(X_train, y_train)

forest_score = forest.score(X_test, y_test)



boost = GradientBoostingRegressor().fit(X_train, y_train)

boost_score = boost.score(X_test, y_test)



svm = SVR().fit(X_train_scaled, y_train_scaled)

svm_score = svm.score(X_test_scaled, y_test_scaled)



mlp = MLPRegressor().fit(X_train_scaled, y_train_scaled)

mlp_score = mlp.score(X_test_scaled, y_test_scaled)



baseline_score = pd.DataFrame({'model':['KNN','Linear Regression', 'Ridge', 'Lasso', 'Decision Tree','Random Forest',

                                     'Gradient Boost','Kernel SVM','Neural Network'], 'Score':[knn_score, lr_score,

                                                                                              ridge_score, lasso_score,

                                                                                              tree_score, forest_score,

                                                                                              boost_score, svm_score,

                                                                                              mlp_score]})

baseline_score.sort_values(by='Score', ascending=False)
train_accuracy = []

test_accuracy = []

for i in range(1,11):

    knn = KNeighborsRegressor(n_neighbors=i)

    knn.fit(X_train_scaled, y_train_scaled)

    train_accuracy.append(knn.score(X_train_scaled, y_train_scaled))

    test_accuracy.append(knn.score(X_test_scaled, y_test_scaled))

    



score = pd.DataFrame({'n_neighbors':range(1,11),'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('n_neighbors')

score.transpose()
knn = KNeighborsRegressor(n_neighbors=2).fit(X_train_scaled, y_train_scaled)

knn_score = knn.score(X_test_scaled, y_test_scaled)
train_accuracy = []

test_accuracy = []

for i in [0.001, 0.01, 0.1, 1, 100]:

    ridge = Ridge(alpha=i).fit(X_train, y_train)

    train_accuracy.append(ridge.score(X_train, y_train))

    test_accuracy.append(ridge.score(X_test, y_test))

    

pd.DataFrame({'alpha':[0.001,0.01,0.1,1,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('alpha').transpose()
ridge = Ridge(alpha=0.001).fit(X_train,y_train)

ridge_score = ridge.score(X_test, y_test)
train_accuracy = []

test_accuracy = []

for i in [0.001, 0.01, 0.1, 1, 100]:

    lasso = Lasso(alpha=i).fit(X_train, y_train)

    train_accuracy.append(lasso.score(X_train, y_train))

    test_accuracy.append(lasso.score(X_test, y_test))

    

pd.DataFrame({'alpha':[0.001,0.01,0.1,1,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('alpha').transpose()
lasso = Lasso(alpha=0.001).fit(X_train,y_train)

lasso_score = lasso.score(X_test, y_test)
train_accuracy = []

test_accuracy = []

for i in [1,2,3,10,100]:

    tree = DecisionTreeRegressor(max_depth=i).fit(X_train, y_train)

    train_accuracy.append(tree.score(X_train, y_train))

    test_accuracy.append(tree.score(X_test, y_test))

    

pd.DataFrame({'max_depth':[1,2,3,10,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('max_depth').transpose()

tree = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)

tree_score = tree.score(X_test, y_test)
train_accuracy = []

test_accuracy = []

for i in [5, 20, 50, 75, 100]:

    forest = RandomForestRegressor(n_estimators=i, random_state=43).fit(X_train, y_train)

    train_accuracy.append(forest.score(X_train, y_train))

    test_accuracy.append(forest.score(X_test, y_test))

    

pd.DataFrame({'n_estimator':[5,20,50,75,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('n_estimator').transpose()
forest = RandomForestRegressor(n_estimators=50, random_state=43).fit(X_train, y_train)

forest_score = forest.score(X_test, y_test)
train_accuracy = []

test_accuracy = []

for i in [0.001,0.01,0.1,1]:

    boost = GradientBoostingRegressor(learning_rate=i).fit(X_train, y_train)

    train_accuracy.append(boost.score(X_train, y_train))

    test_accuracy.append(boost.score(X_test, y_test))

    

pd.DataFrame({'learning_rate':[0.001,0.01,0.1,1], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('learning_rate').transpose()

boost = GradientBoostingRegressor(learning_rate=0.1).fit(X_train, y_train)

boost_score = boost.score(X_test, y_test)
train_accuracy = []

test_accuracy = []

for i in [1, 10 ,100 ,1000]:

    svm = SVR(C=i).fit(X_train_scaled, y_train_scaled)

    train_accuracy.append(svm.score(X_train_scaled, y_train_scaled))

    test_accuracy.append(svm.score(X_test_scaled, y_test_scaled))

    

pd.DataFrame({'C':[1,10,100,1000], 'train_accuracy':train_accuracy,'test_accuracy':test_accuracy}).set_index('C').transpose()
svm = SVR(C=10).fit(X_train_scaled, y_train_scaled)

svm_score = svm.score(X_test_scaled, y_test_scaled)
train_accuracy = []

test_accuracy = []

for i in [[10],[10,10], [20,20]]:

    mlp = MLPRegressor(hidden_layer_sizes=i).fit(X_train_scaled, y_train_scaled)

    train_accuracy.append(mlp.score(X_train_scaled, y_train_scaled))

    test_accuracy.append(mlp.score(X_test_scaled, y_test_scaled))

    

pd.DataFrame({'hidden_layers':['10','10,10','20,20'], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('hidden_layers').transpose()
mlp = MLPRegressor(hidden_layer_sizes=[20,20]).fit(X_train_scaled, y_train_scaled)

mlp_score = mlp.score(X_test_scaled, y_test_scaled)
tuning_score = pd.DataFrame([knn_score, lr_score, ridge_score , lasso_score, tree_score, forest_score, boost_score,

                            svm_score, mlp_score], columns=['performance after tuning'])

scores = pd.concat([baseline_score, tuning_score],axis=1)

scores