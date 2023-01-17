import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# Data loading



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
heart = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

heart.head()
heart.shape
heart["DEATH_EVENT"].value_counts()



# deaths = 13

# alive = 299
heart["DEATH_EVENT"].isnull().sum()
sns.barplot(x = "DEATH_EVENT" , y = "age" , data = heart)

sns.set(style = "whitegrid")
sns.barplot(x = "DEATH_EVENT" , y = "serum_creatinine" , data = heart)

sns.set(style = "whitegrid")
sns.barplot(x = "DEATH_EVENT" , y = "creatinine_phosphokinase" , data = heart)

sns.set(style = "whitegrid")
g = sns.FacetGrid(heart,hue = "DEATH_EVENT",height = 5)

g.map(sns.distplot,"anaemia")

g.add_legend()
g = sns.FacetGrid(heart,hue = "DEATH_EVENT",height = 5)

g.map(sns.distplot,"platelets")

g.add_legend()
g = sns.FacetGrid(heart,hue = "DEATH_EVENT",height = 5)

g.map(sns.distplot,"high_blood_pressure")

g.add_legend()
g = sns.FacetGrid(heart,hue = "DEATH_EVENT",height = 5)

g.map(sns.distplot,"serum_sodium")

g.add_legend()
'''by above analysis we can say that serum_creatinine,platelets,high_blood_pressure,creatinine_phosphokinase,anaemia

cause changes to death event'''
heart.corr()  # correlation between any two features
plt.subplots(figsize = (10,10))

sns.heatmap(heart.corr() , annot = True , linewidths = 1 )
correlation = heart.corr()

correlation_target = abs(correlation)

correlation_target['DEATH_EVENT']
heart.head()
# importing libraries



from sklearn import model_selection

from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix 



# making new datasets with important features



x = heart.loc[:,{"high_blood_pressure","anaemia","age","ejection_fraction","serum_creatinine","time"}]

y = np.array(heart["DEATH_EVENT"])

heart['DEATH_EVENT'].value_counts()
# train and test

# train = 70% of total data

# test = 30% of total data



x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = 0.3,random_state = 0)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
model = []

f1_score = []

accuracy = []
# now for cross validation we will take data from train data and split it equally



# x_tr = 70% of total x_train

# x_cv = 30% of total x_train

# y_tr = 70% of total y_train

# y_cv = 30% of total y_train



from sklearn.neighbors import KNeighborsClassifier



x_tr,x_cv,y_tr,y_cv = model_selection.train_test_split(x_train,y_train,test_size = 0.3,random_state = 0)



print(x_tr.shape)

print(x_cv.shape)

print(y_tr.shape)

print(y_cv.shape)
# now main thing i.e, fitting and predicting



for i in range(1,30,2):

    knn = KNeighborsClassifier(n_neighbors = i )

    knn.fit(x_tr,y_tr)

    pred = knn.predict(x_cv)

    acc = accuracy_score(y_cv , pred ,normalize = True)*float(100)

    print(' cv accuracy for k = {0} is {1}' .format (i,acc))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_tr,y_tr)

predict = knn.predict(x_test)

print('test accuracy',accuracy_score(y_test , predict ,normalize = True)*float(100))



knn_normal_accuracy = accuracy_score(y_test , predict ,normalize = True)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,predict)

sns.heatmap(cm , annot = True )



# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



knn_normal_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(knn_normal_f1_score))





# putting datas in list



model.append('knn_normal')

f1_score.append(knn_normal_f1_score)

accuracy.append(knn_normal_accuracy)





knn = KNeighborsClassifier()



param_grid = {'n_neighbors': np.arange(1, 15)}







knn_gcv = GridSearchCV(knn, param_grid, cv=4)



knn_gcv.fit(x_train, y_train)



print("Best K Value is ",knn_gcv.best_params_)



print("test accuracy ",(knn_gcv.score(x_test,y_test))*float(100))



knn_grid_accuracy = knn_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,knn_gcv.predict(x_test))

sns.heatmap(cm , annot = True )



# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



knn_grid_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(knn_grid_f1_score))



# putting datas in list



model.append('knn_grid')

f1_score.append(knn_grid_f1_score)

accuracy.append(knn_grid_accuracy)
from sklearn.linear_model import LogisticRegression



lor = LogisticRegression(max_iter=1000)



params_lor = {'C':[0.00001,0.0001,0.001,0.1,1,10,100]}



lor_gcv = GridSearchCV(lor , param_grid = params_lor)



lor_gcv.fit(x_train, y_train)



print("Best C Value is ",lor_gcv.best_params_)



print("test accuracy ",(lor_gcv.score(x_test,y_test))*float(100))



logistic_regression_accuracy = lor_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,lor_gcv.predict(x_test))

sns.heatmap(cm , annot = True )





# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



logistic_regression_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(logistic_regression_f1_score))



# putting datas in list



model.append('logistic_regression')

f1_score.append(logistic_regression_f1_score)

accuracy.append(logistic_regression_accuracy)
from sklearn.naive_bayes import GaussianNB



naive = GaussianNB()



params_naive = {'var_smoothing':[0.00001,0.0001,0.001,0.1,1,10,100]}



naive_gcv = GridSearchCV(naive , param_grid = params_naive )



naive_gcv.fit(x_train, y_train)



print("Best var_smoothing Value is ",naive_gcv.best_params_)



print("test accuracy ",(naive_gcv.score(x_test,y_test))*float(100))



naive_bayes_accuracy = naive_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,naive_gcv.predict(x_test))

sns.heatmap(cm , annot = True )





# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



naive_bayes_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(naive_bayes_f1_score))







# putting datas in list



model.append('naive_bayes')

f1_score.append(naive_bayes_f1_score)

accuracy.append(naive_bayes_accuracy)
from sklearn.svm import SVC



svm = SVC()



params_svm = {'C':[0.00001,0.0001,0.001,0.1,1,10,100]}



svm_gcv = GridSearchCV(svm , param_grid = params_svm )



svm_gcv.fit(x_train,y_train)



print("Best C Value is ",svm_gcv.best_params_)



print("test accuracy ",(svm_gcv.score(x_test,y_test))*float(100))



svm_accuracy = svm_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,svm_gcv.predict(x_test))

sns.heatmap(cm , annot = True )





# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



svm_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(svm_f1_score))





# putting datas in list



model.append('svm')

f1_score.append(svm_f1_score)

accuracy.append(svm_accuracy)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()



params_dt = {'max_depth':np.arange(1,10)}



dt_gcv = GridSearchCV(dt , param_grid = params_dt)



dt_gcv.fit(x_train , y_train)



print("Best C Value is ",dt_gcv.best_params_)



print("test accuracy ",(dt_gcv.score(x_test,y_test))*float(100))



decision_tree_accuracy = dt_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,dt_gcv.predict(x_test))

sns.heatmap(cm , annot = True )



# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



decision_tree_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(decision_tree_f1_score))





# putting datas in list



model.append('decision_tree')

f1_score.append(decision_tree_f1_score)

accuracy.append(decision_tree_accuracy)
from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import RandomizedSearchCV



rf = RandomForestClassifier()



params_rf = {'n_estimators' : np.arange(1,100,10) }



rf_gcv = RandomizedSearchCV(rf , param_distributions = params_rf)



rf_gcv.fit(x_train,y_train)



print("Best n_estimators Value is ",rf_gcv.best_params_)





print("test accuracy ",(rf_gcv.score(x_test,y_test))*float(100))



random_forest_accuracy = rf_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,rf_gcv.predict(x_test))

sns.heatmap(cm , annot = True )





# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



random_forest_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(random_forest_f1_score))





# putting datas in list



model.append('random_forest')

f1_score.append(random_forest_f1_score)

accuracy.append(random_forest_accuracy)
from xgboost import XGBClassifier



xgb = XGBClassifier()



params_xgb = {'learning_rate':[0.00001,0.0001,0.001,0.01,0.1,1],'n_estimators':np.arange(1,50,10),'max_depth':np.arange(1,10)}





xgb_gcv = RandomizedSearchCV(xgb , param_distributions = params_xgb)



xgb_gcv.fit(x_train,y_train)



print("Best parameters values are ",xgb_gcv.best_params_)





print("test accuracy ",(xgb_gcv.score(x_test,y_test))*float(100))



xg_boost_accuracy = xgb_gcv.score(x_test,y_test)*float(100)



# to plot confusion matrix



cm = confusion_matrix(y_test,xgb_gcv.predict(x_test))

sns.heatmap(cm , annot = True )





# calculation of F1 score



TN = cm[0,0]

TP = cm[1,1]

FN = cm[0,1]

FP = cm[1,0]



Recall = TP/(TP+FN)

Precision = TP/(TP+FP)



xg_boost_f1_score = ((2 * Recall * Precision)/(Recall + Precision))



print('f1_score of the model is {}'.format(xg_boost_f1_score))





# putting datas in list



model.append('xg_boost')

f1_score.append(xg_boost_f1_score)

accuracy.append(xg_boost_accuracy)

# table for heart_failure_models



dict = {'model': model, 'accuracy': accuracy, 'f1_score': f1_score}   



heart_failure_prediction = pd.DataFrame(dict) 



heart_failure_prediction
# plotting to compare each models on accuracy



fig_dims = (13, 4)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = model, y = accuracy, ax=ax)

# plotting to compare each models on f1_score



fig_dims = (13, 4)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = model, y = f1_score, ax=ax)