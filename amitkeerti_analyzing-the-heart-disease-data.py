# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
######################## Stage 1 ###########################

df = pd.read_csv('../input/heart.csv')



#Print the top 10 rows of the data frame

df.head(10)
#Print the summary of the data frame. The summary shows that the data is clean and all the values are non-null.

df.info()
plt.figure(figsize=(14,14))

sns.heatmap(df.corr(), annot = True, linewidth = 0.8, cmap='coolwarm')
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



#Split the data into training set and testing set. 75 percent of the data will be used for training.

X = df.drop(columns=['target'])

Y = df.target



train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.25,random_state = None)



svm = SVC(kernel='linear',C =100)

#train SVM using training data that has NOT been scaled (or normalized)

svm.fit(train_X,train_Y)

#predict using the non-scaled test data on the learnt SVM.

test_Y_predict=svm.predict(test_X)



print("SVM with Non-normalized data: Accuracy = ", round(accuracy_score(test_Y_predict,test_Y) * 100, 2), "%")



#Note: Accuracy varies by a wide margin from 78% to 85% !

#Note: I tried the split between training and test set with different values: 80:20, 75:25, 70:30. Results were similar.
min_max = [[i, df[i].min(), df[i].max()] for i in list(df)]

min_max
figure, axis = plt.subplots(7,2, figsize=(100,100))

plt.subplots_adjust(hspace = 0.3)



sns.distplot(df.age, ax=axis[0,0])

sns.distplot(df.sex, ax=axis[0,1])

sns.distplot(df.cp, ax=axis[1,0])

sns.distplot(df.trestbps, ax=axis[1,1])

sns.distplot(df.chol, ax=axis[2,0])

sns.distplot(df.fbs, ax=axis[2,1])

sns.distplot(df.restecg, ax=axis[3,0])

sns.distplot(df.thalach, ax=axis[3,1])

sns.distplot(df.exang, ax=axis[4,0])

sns.distplot(df.oldpeak, ax=axis[4,1])

sns.distplot(df.slope, ax=axis[5,0])

sns.distplot(df.ca, ax=axis[5,1])

sns.distplot(df.thal, ax=axis[6,0])



plt.show()
from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()



#Scale the training data using the standard scalar

train_X_std_scalar = std_scaler.fit_transform(train_X)



#Scale the test data using the standard scalar

test_X_std_scalar = std_scaler.fit_transform(test_X)



#train SVM using training data that has NOT been scaled (or normalized)

svm_normalized = SVC(kernel='linear',C =100)

svm_normalized.fit(train_X_std_scalar,train_Y)



#predict using the non-scaled test data on the learnt SVM.

test_Y_predict_std_scalar=svm_normalized.predict(test_X_std_scalar)



#Print the accuracy of the svm v/s actual values.

print("SVM with Standard Scaler Normalized data: Accuracy = ", round(accuracy_score(test_Y_predict_std_scalar,test_Y) * 100, 2), "%")
from sklearn.preprocessing import MinMaxScaler



all_columns = df.columns.values.tolist()

#'age','trestbps','chol','thalach','oldpeak' are the continuous variables

continuous_columns = ['age','trestbps','chol','thalach','oldpeak']

#Rest all of them are catagorical_columns

categorical_columns = [column for column in all_columns if column not in continuous_columns]

categorical_columns.remove('target')



#apply MinMaxScaler only to the continuous variables that exhibit a normal distribution.

min_max_scalar = MinMaxScaler()



#Take a copy of the features.

X_min_max = X



#Scale only the continuous columns using min max scalar

X_min_max[continuous_columns] = min_max_scalar.fit_transform(X_min_max[continuous_columns])



#From the feature data, convert the Categorical data (ex: 0, 1, 2, 3) to indicator variables

X_categorical_data = pd.get_dummies(X_min_max, columns=categorical_columns)



#Split the data into training set and test set.

train_X_min_max,test_X_min_max,train_Y_min_max,test_Y_min_max = train_test_split(X_categorical_data,Y,test_size = 0.25,random_state = None)



#train SVM using training data that has NOT been scaled (or normalized)

svm_normalized_min_max = SVC(kernel='linear',C =100)

svm_normalized_min_max.fit(train_X_min_max,train_Y_min_max)



#predict using the non-scaled test data on the learnt SVM.

test_Y_predict_min_max_scaler=svm_normalized_min_max.predict(test_X_min_max)



#Print the accuracy of the svm v/s actual values.

print("SVM with Min Max Normalized data: Accuracy = ", round(accuracy_score(test_Y_predict_min_max_scaler,test_Y_min_max) * 100, 2), "%")
from sklearn.preprocessing import RobustScaler



#apply MinMaxScaler only to the continuous variables that exhibit a normal distribution.

robust_scalar = RobustScaler()



X_robust = X



#Scale only the continuous columns using robust scalar

X_robust[continuous_columns] = robust_scalar.fit_transform(X_robust[continuous_columns])



#From the feature data, convert the Categorical data (ex: 0, 1, 2, 3) to indicator variables

X_categorical_data = pd.get_dummies(X_robust, columns=categorical_columns)



#Split the data into training set and test set.

train_X_robust,test_X_robust,train_Y_robust,test_Y_robust = train_test_split(X_categorical_data,Y,test_size = 0.25,random_state = None)



#train SVM using training data that has NOT been scaled (or normalized)

svm_normalized_robust = SVC(kernel='linear',C =100)

svm_normalized_robust.fit(train_X_robust,train_Y_robust)



#predict using the non-scaled test data on the learnt SVM.

test_Y_predict_robust_scaler=svm_normalized_robust.predict(test_X_robust)



#Print the accuracy of the svm v/s actual values.

print("SVM with Robust Scaler : Accuracy = ", round(accuracy_score(test_Y_predict_robust_scaler,test_Y_robust) * 100, 2), "%")

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix





gs_tuning_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],

                     'C': [0.01, 0.1, 1, 10, 100]},

                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

clf = GridSearchCV(SVC(), gs_tuning_parameters, cv=5)

clf.fit(train_X_min_max, train_Y_min_max)



print('---- Choosing the best parameters for SVM on the data scaled using MinMaxScalar ----')

print()

print('The best parameters are:', clf.best_params_)

print()



test_Y_predict_GridSearchCV = clf.predict(test_X_min_max)



print("SVM with best params from GridSearchCV: Accuracy = ", round(accuracy_score(test_Y_predict_GridSearchCV, test_Y_min_max) * 100, 2), "%")

print("SVM without using GridSearchCV: Accuracy = ", round(accuracy_score(test_Y_predict_min_max_scaler,test_Y_min_max) * 100, 2), "%")

print()

print("Classification Report:")

print(classification_report(test_Y_min_max, test_Y_predict_GridSearchCV))





print("The above classification report can be understood using the below confusion matrix")

print()

confusion_df = pd.DataFrame(confusion_matrix(test_Y_min_max,test_Y_predict_GridSearchCV),

             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],

             index = ["Class " + str(class_name) for class_name in [0,1]])



print(confusion_df)

print()
train_X_feat_optimized = train_X_min_max.drop(columns=['exang_0', 'exang_1', 'oldpeak', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4'])

test_X_feat_optimized = test_X_min_max.drop(columns=['exang_0', 'exang_1', 'oldpeak', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4'])



gs_tuning_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],

                     'C': [0.01, 0.1, 1, 10, 100]},

                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

clf_fs_heatmap = GridSearchCV(SVC(), gs_tuning_parameters, cv=5)

clf_fs_heatmap.fit(train_X_feat_optimized, train_Y_min_max)



print('---- Performance comparison after feature selection from heatmap on data scaled using MinMaxScalar ----')

print()

print('The best parameters are:', clf_fs_heatmap.best_params_)

print()

test_Y_predict_feat_optimized = clf_fs_heatmap.predict(test_X_feat_optimized)

print("SVM with feature optimized from heatmap: Accuracy = ", round(accuracy_score(test_Y_predict_feat_optimized, test_Y_min_max) * 100, 2), "%")

print("SVM with best params from GridSearchCV: Accuracy = ", round(accuracy_score(test_Y_predict_GridSearchCV, test_Y_min_max) * 100, 2), "%")

print("SVM without using GridSearchCV: Accuracy = ", round(accuracy_score(test_Y_predict_min_max_scaler,test_Y_min_max) * 100, 2), "%")

print()
from sklearn.feature_selection import SelectKBest, f_classif





bestfeatures = SelectKBest(score_func=f_classif, k=10)

fit = bestfeatures.fit(X,Y)



df_scores = pd.DataFrame(fit.scores_)

df_columns = pd.DataFrame(X.columns)



 

featureScores = pd.concat([df_columns,df_scores],axis=1)

#Give name to the columns

featureScores.columns = ['Features','Score']

#print the features with their score

#print(featureScores.nlargest(13,'Score'))  



#Annova method indicates that restecg, chol and fbs are least relevant features.

#Note that 'exang', 'oldpeak' and 'ca' which were least relevant features in heatmap are part of top  features in ANNOVA!!!





train_X_ANNOVA = train_X_min_max.drop(columns=['restecg_0', 'restecg_1', 'restecg_2', 'chol', 'fbs_0', 'fbs_1'])

test_X_ANNOVA = test_X_min_max.drop(columns=['restecg_0', 'restecg_1', 'restecg_2', 'chol', 'fbs_0', 'fbs_1'])



gs_tuning_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],

                     'C': [0.01, 0.1, 1, 10, 100]},

                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

clf_fs_ANNOVA = GridSearchCV(SVC(), gs_tuning_parameters, cv=5)

clf_fs_ANNOVA.fit(train_X_ANNOVA, train_Y_min_max)



print('---- Performance comparison after feature selection using ANNOVA on data scaled using MinMaxScalar ----')

print()

print('The best parameters are:', clf_fs_ANNOVA.best_params_)

print()

test_Y_predict_ANNOVA = clf_fs_ANNOVA.predict(test_X_ANNOVA)

print("SVM with feature optimized using ANNOVA : Accuracy = ", round(accuracy_score(test_Y_predict_ANNOVA, test_Y_min_max) * 100, 2), "%")

print("SVM with feature optimized from heatmap: Accuracy = ", round(accuracy_score(test_Y_predict_feat_optimized, test_Y_min_max) * 100, 2), "%")

print("SVM with best params from GridSearchCV: Accuracy = ", round(accuracy_score(test_Y_predict_GridSearchCV, test_Y_min_max) * 100, 2), "%")

print("SVM without using GridSearchCV: Accuracy = ", round(accuracy_score(test_Y_predict_min_max_scaler,test_Y_min_max) * 100, 2), "%")

print()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier





classifer_models = [SVC(kernel='linear',C =10),

                    LogisticRegression(max_iter=1000,solver='lbfgs'),

                    RandomForestClassifier(n_estimators=200,random_state=50),

                    KNeighborsClassifier(),

                    GaussianNB(),

                    DecisionTreeClassifier(),

                    ExtraTreeClassifier(),

                    GradientBoostingClassifier()

                   ]



classifer_modelnames = ['SVC',

                        'LogisticRegression',

                        'RandomForestClassifier',

                        'KNeighborsClassifier',

                        'GaussianNB',

                        'DecisionTreeClassifier',

                        'ExtraTreeClassifier',

                        'GradientBoostingClassifier' 

                       ]



print("Performance of different classification algorithms on scaled data:")

for index, model in enumerate(classifer_models):

    model.fit(train_X_min_max,train_Y_min_max)

    print("Classifier ", classifer_modelnames[index], ", Accuracy  = ", round(model.score(test_X_min_max,test_Y_min_max)*100,2),"%")
