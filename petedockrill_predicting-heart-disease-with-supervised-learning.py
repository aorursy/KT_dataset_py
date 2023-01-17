import numpy as np 

import pandas as pd 

import sklearn as sl

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style("whitegrid")

%matplotlib inline
full_data = pd.read_csv('../input/heart-disease-dataset/heart.csv')



num_features = ['age','trestbps','chol','restecg','thalach','oldpeak']

cat_features = ['sex','cp','fbs','exang','ca','thal','slope','restecg']

full_data = full_data[['sex','cp','fbs','exang','slope','restecg','ca','thal','age','trestbps','chol','thalach','oldpeak','target']]

MF_pairplot = sns.pairplot(full_data[['age','trestbps','chol','thalach','oldpeak','target']], hue='target', kind ='reg', height = 4)

MF_pairplot


full_data = full_data.dropna()

y = full_data['target']

X = full_data.drop('target', axis = 1)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state =42) 



print(y_train.shape)

print(y_test.shape)

from sklearn.compose import ColumnTransformer 

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

cat_column_tran = ColumnTransformer([('ohe',OneHotEncoder(sparse=False), slice(0,8,1))], remainder='passthrough')

num_column_tran = ColumnTransformer([('ss',StandardScaler(),slice(9,13,1))], remainder='passthrough')

preprocess = Pipeline([('Cat_tran',cat_column_tran),('Num_tran',num_column_tran),('PCA',PCA(0.95))])



processed_data = preprocess.fit_transform(X_train)
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV

params_KNN = dict(n_neighbors = range(1,10))

grid_search_KNN = GridSearchCV(KNN, param_grid = params_KNN, cv =4, scoring='recall')

grid_search_KNN.fit(X_train,y_train)
KNN_best_k = grid_search_KNN.best_params_['n_neighbors']

print("For a k-Nearest Neighbors model, the optimal value of k is "+str(KNN_best_k))

KNN_df = pd.DataFrame(grid_search_KNN.cv_results_)

fig_KNN = plt.figure(figsize=(12,9))

plt.plot(KNN_df['param_n_neighbors'],KNN_df['mean_test_score'],'b-o')

plt.xlim(0,10)

plt.ylim(0.5,1.0)

plt.xlabel('k')

plt.ylabel('Mean recall over 4 cross-validation sets')
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(max_features = 1)
params_DT = dict(max_depth = range(2,30))

grid_search_DT = GridSearchCV(DT, param_grid = params_DT, cv = 4, scoring='recall')

grid_search_DT.fit(X_train,y_train)
DT_best_layers = grid_search_DT.best_params_['max_depth']

print("For a Decision Tree model, the optimal number of layers is "+str(DT_best_layers))

DT_df = pd.DataFrame(grid_search_DT.cv_results_)

fig = plt.figure(figsize=(12,9))

plt.plot(DT_df['param_max_depth'],DT_df['mean_test_score'],'g-o')

plt.xlim(0,30)

plt.ylim(0.65,1.0)

plt.xlabel('Maximum layers')

plt.ylabel('Mean recall over 4 cross-validation sets')
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(max_depth = DT_best_layers, max_features = 1)
params_RF = dict(n_estimators = range(1,50))

grid_search_RF = GridSearchCV(RF, param_grid = params_RF, cv = 4, scoring='recall')

grid_search_RF.fit(X_train,y_train)
RF_best_estimators = grid_search_RF.best_params_['n_estimators']

print("For a Random Forest, the optimal number of estimators is "+str(RF_best_estimators))

RF_df = pd.DataFrame(grid_search_RF.cv_results_)

fig = plt.figure(figsize=(30,9))

plt.plot(RF_df['param_n_estimators'],RF_df['mean_test_score'],'r-o')

plt.xlim(0,50)

plt.ylim(0.65,1.0)

plt.xlabel('Number of estimators')

plt.ylabel('Mean recall over 4 cross-validation sets')
from sklearn.metrics import recall_score, precision_score, accuracy_score



KNN_final = grid_search_KNN.best_estimator_

KNN_final.fit(processed_data, y_train)



DT_final = grid_search_DT.best_estimator_

DT_final.fit(processed_data, y_train)



RF_final = grid_search_RF.best_estimator_

RF_final.fit(processed_data, y_train)



pipelines = [KNN_final, DT_final, RF_final]



best_recall = 0.0

best_classifier = 0.0

best_pipeline = ""



pipe_dict = {0:'k-Nearest Neighbours',1:'Decision Tree',2:'Random Forest'}



for i,model in enumerate(pipelines):

    X_test_trans = preprocess.transform(X_test)

    y_pred = model.predict(X_test_trans)

    print("{} test recall: {}".format(pipe_dict[i],recall_score(y_pred, y_test) ))

    print("{} test precision: {}".format(pipe_dict[i],precision_score(y_pred, y_test) ))

    print("{} test accuracy: {}".format(pipe_dict[i],accuracy_score(y_pred, y_test) ))

    if recall_score(y_pred,y_test)>best_recall:

        best_recall = recall_score(y_pred,y_test)

        best_pipeline = model 

        best_classifer = i



print("Classifier with best recall: {}".format(pipe_dict[best_classifier]))
