import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter("ignore")
train = pd.read_csv("../input/train.csv")
train.head()
X_train_1 = train.iloc[:, 1:-1]
Y_train_1 = train.iloc[:, -1]
Y_train_log = np.log(Y_train_1 + 1)

X_train_1['room_per_household'] = (X_train_1['total_rooms'] + X_train_1['total_bedrooms'])/X_train_1['households']
X_train_1['room_per_population'] = (X_train_1['total_rooms'] + X_train_1['total_bedrooms'])/X_train_1['population']
X_train_1.head()
X_train_1.shape
figure, axes = plt.subplots(nrows=8, ncols=1, figsize=(10,60))
for i in range(0, 8):
    axes[i].hist(X_train_1.iloc[:, i+2], range=(0, X_train_1.iloc[:, i+2].quantile(.99)), bins=25)
    axes[i].set_title(X_train_1.columns[i+2])
plt.show()
figure, axes = plt.subplots(nrows=8, ncols=1, figsize=(20,60))
for i in range(0, 8):
    axes[i].scatter(X_train_1.iloc[:, i+2], Y_train_1)
    axes[i].set_title(X_train_1.columns[i+2])
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X_train_1, Y_train_log, test_size = .25)
#Normalizacao
std_scaler = preprocessing.StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

#Normalização desconsiderando outliers
robust_std_scaler = preprocessing.RobustScaler()
X_train_robust_std = robust_std_scaler.fit_transform(X_train)
X_test_robust_std = robust_std_scaler.transform(X_test)

#Transformacao polinomial com normalizacao
pol = preprocessing.PolynomialFeatures(degree=3, interaction_only=True)
X_train_pol = pol.fit_transform(X_train)
X_test_pol = pol.transform(X_test)

scaler = preprocessing.StandardScaler()
X_train_pol = scaler.fit_transform(X_train_pol)
X_test_pol = scaler.transform(X_test_pol)

#Transformando para intervalo [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_min_max = min_max_scaler.fit_transform(X_train)
X_test_min_max = min_max_scaler.transform(X_test)

#Juntando todas as tranformacoes para poder iterar no treino
X_train_t = {0: X_train_std, 1: X_train_robust_std, 2: X_train_pol, 3:X_train_min_max}
X_test_t = {0: X_test_std, 1: X_test_robust_std, 2: X_test_pol, 3:X_test_min_max}
tranformation = ['Normalizacao', 'Normalizacao sem outliers', 'Polinomial', 'Min Max']
knn_models = {}
knn_cv_scores = {}

print("**KNN MODEL**")
for i in range (0, 4):
    knn_models[i] = []
    knn_cv_scores[i] = []
    print('Tranformacao: ' + str(tranformation[i]))
    for k in range (1, 15):
        knn = KNeighborsRegressor(n_neighbors=k ,n_jobs=-1)
        knn_cv_scores[i].append(cross_val_score(knn, X_train_t[i], Y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error'))
        print('Val Score K=' + str(k) + ': ' + str(knn_cv_scores[i][k-1].mean()))
ridge = linear_model.RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000], cv=3)
ridge_models = {}
ridge_cv_scores = {}

print("**RIDGE MODEL**")
for i in range (0, 4):
    ridge_cv_scores[i] = cross_val_score(ridge, X_train_t[i], Y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    print('Tranformacao: ' + str(tranformation[i]) + ', Val Score (mean): ' + str(ridge_cv_scores[i].mean()))
lasso = linear_model.LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000], cv=3)
lasso_models = {}
lasso_cv_scores = {}

print("**LASSO MODEL**")
for i in range (0, 4):
    lasso_cv_scores[i] = cross_val_score(lasso, X_train_t[i], Y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    print('Tranformacao: ' + str(tranformation[i]) + ', Val Score (mean): ' + str(lasso_cv_scores[i].mean()))
print("**RANDOM FOREST MODEL**")
for k in range(2, 10):
    random_forest = RandomForestRegressor(n_estimators=500, max_features=k, n_jobs=-1)
    random_forest_models = {}
    random_forest_cv_scores = {}
    for i in range (0, 4):
        random_forest_models[i] = random_forest
        random_forest_cv_scores[i] = cross_val_score(random_forest, X_train_t[i], Y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        print('Tranformacao: ' + str(tranformation[i]) + ', Val Score (mean),  ' + 'Features=' + str(k) + ': ' + str(random_forest_cv_scores[i].mean()))
std_scaler = preprocessing.StandardScaler()
X_train= std_scaler.fit_transform(X_train_1)

random_forest = RandomForestRegressor(n_estimators=500, max_features=7, n_jobs=-1)
random_forest.fit(X_train, Y_train_log)
test = pd.read_csv('../input/test.csv')
test['room_per_household'] = (test['total_rooms'] + test['total_bedrooms'])/test['households']
test['room_per_population'] = (test['total_rooms'] + test['total_bedrooms'])/test['population']
test.head()
X_test = std_scaler.transform(test.iloc[:, 1:])
Y_predict = np.exp(random_forest.predict(X_test) + 1)