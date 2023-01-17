# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, precision_score

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

#from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#from sklearn.naive_bayes import GaussianNB

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



import xgboost as xgb

from xgboost import XGBClassifier

#from xgboost import XGBRegressor





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



path_to_file = '../input/particle-identification-from-detector-responses/pid-5M.csv'

df = pd.read_csv(path_to_file)
#classifiers = [

#    KNeighborsClassifier(3),

#    SVC(kernel="linear", C=0.025),

#    SVC(gamma=2, C=1),

#    GaussianProcessClassifier(1.0 * RBF(1.0)),

#    DecisionTreeClassifier(max_depth=5),

#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

#    MLPClassifier(alpha=1, max_iter=1000),

#    AdaBoostClassifier(),

#    GaussianNB(),

#    QuadraticDiscriminantAnalysis()]

df.head(5)
#for the moment reduce the size -> speed up 

df = df[0:100000]

#df = df[0:20000]



df_pi = df[df.id == abs(211)]

df_ka = df[df.id == abs(321)]

df_pr = df[df.id == abs(2212)]





df.head(10)


fig, ax1 = plt.subplots(figsize=(15,10))

sns.scatterplot(x=df_pi.p, y=df_pi.beta , data=df, ax = ax1, label = "Pions");

sns.scatterplot(x=df_ka.p, y=df_ka.beta , data=df, ax = ax1, label = "Kaons");

sns.scatterplot(x=df_pr.p, y=df_pr.beta , data=df, ax = ax1, label = "Protons");

ax1.set_xlabel(r'Particle momentum $p$',fontsize=22)

ax1.set_ylabel(r'Relativistic velocity $\beta$', fontsize=22)

ax1.tick_params(axis='both', which='major', labelsize=15)



plt.legend(loc="lower right", fontsize=20)



plt.show()

X_full = df

X = X_full.drop(columns=['id'],axis = 1)

y = X_full.id

# training and validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
def get_acc(y,pred):

    corr = 0

    for i in range(len(y)):

        if abs(y.iloc[i] - pred[i]) < 0.1*y.iloc[i]:

            corr+=1

    return corr/len(y)
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "n_estimators": [10, 100]

             }





# run grid search





DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

model_ADC = AdaBoostClassifier(base_estimator = DTC)

grid_ADC  = GridSearchCV(model_ADC, param_grid=param_grid, scoring = 'roc_auc')





model_ADC.fit(X_train,y_train) 

pred_ADC  = model_ADC.predict(X_test)



acc_ADC   = accuracy_score(y_test,pred_ADC)

mse_ADC   = mean_squared_error(y_test, pred_ADC)

pre_ADC   = precision_score(y_test, pred_ADC, average='macro')







print(acc_ADC)

print(mse_ADC)

print(pre_ADC)





# Create the parameter grid based on the results of random search 

#param_grid_XGB = {

#    'max_depth': [3,5,9],

#    'max_features': [3,4,5],

#    'min_samples_leaf': [2, 3,4],

#    'min_samples_split': [6, 8, 10],

#    'n_estimators': [75, 100, 200]

#}



#model_XGB = XGBClassifier()

#grid_search_XGB = GridSearchCV(estimator = model_XGB, param_grid = param_grid_XGB, cv = 3, n_jobs = -1, verbose = 2)

#grid_search_XGB.fit(X_train, y_train)

#best_grid = grid_search_RF.best_estimator_
    

model_XGB    = XGBClassifier()

model_XGB.fit(X_train, y_train)

pred_XGB     = model_XGB.predict(X_test)



data_dmatrix = xgb.DMatrix(data=X,label=y)

params       = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.05,'max_depth': 5, 'alpha': 10}

XGB_cla      = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

acc_XGB      = accuracy_score(y_test,pred_XGB)



mae_XGB      = mean_absolute_error(y_test, pred_XGB)

mse_XGB      = mean_squared_error(y_test, pred_XGB)

pre_XGB      = precision_score(y_test, pred_XGB, average='macro')







print(acc_XGB)

print(mse_XGB)

print(pre_XGB)





fig, ax = plt.subplots(figsize=(20, 30))

#xgb.plot_importance(XGB_cla, ax=ax)

xgb.plot_tree(XGB_cla, ax = ax)

model_KNN  = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')

model_KNN.fit(X_train, y_train)

model_KNN.predict(X_test)

pred_KNN  = model_KNN.predict(X_test)

acc_KNN   = accuracy_score(y_test, pred_KNN)





mae_KNN   = mean_absolute_error(y_test, pred_KNN)

mse_KNN   = mean_squared_error(y_test, pred_KNN)

pre_KNN   = precision_score(y_test, pred_KNN, average='macro')





print(acc_KNN)

print(mse_KNN)

print(pre_KNN)





model_MLP = MLPClassifier(alpha=1, max_iter=100)

model_MLP.fit(X_train, y_train)

model_MLP.predict(X_test)

pred_MLP = model_MLP.predict(X_test)

acc_MLP  = accuracy_score(y_test, pred_MLP)

mae_MLP  = mean_absolute_error(y_test, pred_MLP)

mse_MLP  = mean_squared_error(y_test, pred_MLP)

pre_MLP0 = precision_score(y_test, pred_MLP, average='macro', zero_division = 0)

pre_MLP1 = precision_score(y_test, pred_MLP, average='macro', zero_division = 1)







print(acc_MLP)

print(mse_MLP)

print(pre_MLP0)

print(pre_MLP1)

model_SVC = SVC(kernel="linear", C=0.025)

model_SVC.fit(X_train, y_train)

model_SVC.predict(X_test)

pred_SVC = model_KNN.predict(X_test)

acc_SVC  = accuracy_score(y_test, pred_SVC)

mse_SVC  = mean_squared_error(y_test, pred_SVC)

pre_SVC  = precision_score(y_test, pred_SVC, average='macro')





print(acc_SVC)

print(mse_SVC)

print(pre_SVC)
#GaussianProcessClassifier



#super slow -> comment out

#model_GPC = GaussianProcessClassifier(1.0 * RBF(1.0))

#model_GPC.fit(X_train, y_train)

#model_GPC.predict(X_test)

#pred_GPC = model_GPC.predict(X_test)

#acc_GPC  = accuracy_score(y_test,pred_GPC)

#print(acc_GPC)

model_DT = DecisionTreeClassifier(max_depth=5)

model_DT.fit(X_train, y_train)

model_DT.predict(X_test)

pred_DT = model_DT.predict(X_test)

acc_DT  = accuracy_score(y_test, pred_DT)

mse_DT  = mean_squared_error(y_test, pred_DT)

pre_DT  = precision_score(y_test, pred_DT, average='macro')





print(mse_DT)

print(acc_DT)

print(pre_DT)



#Grid search for hyperparameter optimisation

#commented out otherwise it would take too long



# Create the parameter grid based on the results of random search 

#param_grid_RF = {

#    'bootstrap': [True],

#    'max_depth': [80, 100, 120],

#    'max_depth': [3, 5, 7, 10, 20],

#    'max_features': [3,4],

#    'min_samples_leaf': [2, 3, 4],

#    'min_samples_split': [6, 8, 10],

#    'n_estimators': [100]

#}



#model_RF = RandomForestClassifier(random_state=0)

#grid_search_RF = GridSearchCV(estimator = model_RF, param_grid = param_grid_RF, cv = 3, n_jobs = -1, verbose = 2)

#grid_search_RF.fit(X_train, y_train)

#best_grid = grid_search_RF.best_estimator_

#best_grid






model_RF = Pipeline([

        ('classifier', RandomForestClassifier(n_estimators =100, min_samples_leaf=2, min_samples_split=8, max_depth=10, max_features=3, random_state=0))

])



model_RF.fit(X_train, y_train)

pred_RF = model_RF.predict(X_test)

acc_RF  = accuracy_score(y_test, pred_RF)

mse_RF  = mean_squared_error(y_test, pred_RF)

pre_RF  = precision_score(y_test, pred_RF, average='macro')







# feature importance

tree_feature_importances = (

    model_RF.named_steps['classifier'].feature_importances_)

sorted_idx = tree_feature_importances.argsort()



fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 10

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size

y_ticks = np.arange(0, len(X.columns))

fig, ax = plt.subplots()

ax.barh(y_ticks, tree_feature_importances[sorted_idx])

ax.set_yticklabels(X.columns[sorted_idx])

ax.set_yticks(y_ticks)

ax.set_title("Random Forest Feature Importances")

plt.show()





print(acc_RF)

print(mse_RF)

print(pre_RF)
print("AdaBoost            : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_ADC * 100.0, pre_ADC * 100.0, mse_ADC))

print("Decision Tree       : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_DT * 100.0, pre_DT * 100.0, mse_DT))

print("XGBoost             : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_XGB * 100.0, pre_XGB * 100.0, mse_XGB))

print("KNearNeigh          : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_KNN * 100.0, pre_KNN * 100.0, mse_KNN))

print("MLP                 : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_MLP * 100.0, pre_MLP1 * 100.0, mse_MLP))

print("Random forest       : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_RF * 100.0, pre_RF * 100.0, mse_RF))

print("SVC                 : Accuracy: %.2f%%, Precision: %.2f%%, mean sq. error : %.2f" % (acc_SVC * 100.0, pre_SVC * 100.0, mse_SVC))





labels = ['', 'AdaBoost','Dec. Tree','XGBoost','k near. neighb.', 'Mult. Lay. Perc.', 'Random For.', 'SVC']





figRes, axRes = plt.subplots(3, sharex=True, figsize=(15,10))

axRes[0].set_ylabel('Accuracy', fontsize=22)

axRes[0].tick_params(axis='both', which='major', labelsize=15)

axRes[0].plot([acc_ADC, acc_DT, acc_XGB, acc_KNN, acc_MLP, acc_RF, acc_SVC])

axRes[1].set_ylabel('Precision', fontsize=22)

axRes[1].tick_params(axis='both', which='major', labelsize=15)

axRes[1].plot([pre_ADC, pre_DT, pre_XGB, pre_KNN, pre_MLP1, pre_RF, pre_SVC])

axRes[2].set_xlabel(r'ML algorithm',fontsize=22)

axRes[2].set_ylabel(r'Mean sq. error', fontsize=22)

axRes[2].tick_params(axis='both', which='major', labelsize=15)

axRes[2].set_xticklabels(labels)

axRes[2].plot([mse_ADC, mse_DT, mse_XGB, mse_KNN, mse_MLP, mse_RF, mse_SVC])
