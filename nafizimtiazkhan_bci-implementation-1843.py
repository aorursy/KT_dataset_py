%matplotlib inline

!pip install --upgrade pip

import pandas as pd

import pickle

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

import numpy as np

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

! pip install -q scikit-plot

from sklearn.tree import DecisionTreeRegressor

import scikitplot as skplt

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv('../input/bci-final/bci_final.csv')

data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.info())

features = list(data.columns.values)

print(features)

print(df.head())

intact=pd.read_csv('../input/bci-final/bci_final.csv')



corr = data.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.0) | (corr <= -0.0)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);

# Plottinf correlation above or below 0.5

corr = data.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);





import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
print(data.head())

reg_evaluation = pd.DataFrame({'Model': [],

                           'Details':[],

                           'RMSE(train)':[],

                           'R-squared (train)':[],

                           'Adj R-squared (train)':[],

                           'MAE (train)':[],

                           'RMSE (test)':[],

                           'R-squared (test)':[],

                           'Adj R-squared (test)':[],

                           'MAE(test)':[],

                           '10-Fold Cross Validation':[]})



reg_evaluation2 = pd.DataFrame({'Model': [],

                           'Test':[],

                           '1':[],

                           '2':[],

                           '3':[],

                           '4':[],

                           '5':[],

                           '6':[],

                           '7':[],

                           '8':[],

                           '9':[],

                           '10':[],

                           'Mean':[]})



classification_evaluation = pd.DataFrame({'Model': [],

                           'Accuracy(train)':[],

                           'Precision(train)':[],

                           'Recall(train)':[],

                           'F1_score(train)':[],

                           'Accuracy(test)':[],

                           'Precision(test)':[],

                           'Recalll(test)':[],

                           'F1_score(test)':[]})



classification_evaluation2 = pd.DataFrame({'Model': [],

                           'Test':[],

                           '1':[],

                           '2':[],

                           '3':[],

                           '4':[],

                           '5':[],

                           '6':[],

                           '7':[],

                           '8':[],

                           '9':[],

                           '10':[],

                           'Mean':[]})

print(data.info())
# from imblearn.over_sampling import SMOTE



# # for reproducibility purposes

# seed = 100

# # SMOTE number of neighbors

# k = 1



# #df = pd.read_csv('df_imbalanced.csv', encoding='utf-8', engine='python')

# # make a new df made of all the columns, except the target class

# X = df.loc[:, df.columns != 'state']

# y = df.state

# sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=seed)

# X_res, y_res = sm.fit_resample(X, y)



# # plt.title('base')

# # plt.xlabel('x')

# # plt.ylabel('y')

# # plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,

# #            s=25, edgecolor='k', cmap=plt.cm.coolwarm)

# # plt.show()



# df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)

# # rename the columns

# #df.columns = ['e1/do']+['e2/do']+['fu/fy']+['fmx/fndt']+['type']

# df.to_csv('df_smoted.csv', index=False, encoding='utf-8')

# df.head()

# data=df

# print(data.shape)
data.head()
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



from sklearn.svm import SVR

complex_model_1=SVR()

from sklearn.tree import DecisionTreeRegressor

complex_model_1= DecisionTreeRegressor(random_state=0)



complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['SVR','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('SVR_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('SVR_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('SVR_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('SVR_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('SVR_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('SVR_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

print(intact)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

from sklearn.metrics import accuracy_score

clf =svm.SVC(kernel='rbf',degree=100)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['SVM',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('SVM_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('SVM_test_actual.csv', y_test, delimiter=',', fmt='%s')







# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('SVM_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('SVM_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('SVM_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('SVM_entire_actual.csv', y, delimiter=',', fmt='%s')



data['state']=MLR_y_pred_entire_data





import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





p=y_train

q=y_test

y_train=y_train.replace([0,1], ["Unconcerned","Engaged"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1], ["Unconcerned","Engaged"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1], ["Unconcerned","Engaged"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Unconcerned","Engaged"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



from sklearn.ensemble import RandomForestRegressor

complex_model_1 = RandomForestRegressor(n_estimators= 136, min_samples_split= 2,

                                        min_samples_leaf= 1, max_features= 'auto',  bootstrap= 'True')



from sklearn.tree import DecisionTreeRegressor

complex_model_1= DecisionTreeRegressor(random_state=0)





complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['RF_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('RF_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('RF_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('RF_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('RF_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('RF_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('RF_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)




# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf =RandomForestClassifier()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['RF_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('RF_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('RF_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('RF_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('RF_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('RF_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('RF_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



import xgboost as xgb

from xgboost import plot_importance

# complex_model_1 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, gamma=0, subsample=0.75)



import xgboost as xgb

from xgboost import plot_importance

complex_model_1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=10)



complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['XgBoost_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('XgBoost_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('XgBoost_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('XgBoost_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('XgBoost_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('XgBoost_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('XgBoost_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)


# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf =xgb.XGBClassifier(random_state=700)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['XgBoost_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('XgBoost_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('XgBoost_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('XgBoost_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('XgBoost_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('XgBoost_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('XgBoost_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



import xgboost as xgb

from xgboost import plot_importance

from sklearn.ensemble import AdaBoostRegressor

complex_model_1 = AdaBoostRegressor()

complex_model_1= DecisionTreeRegressor(random_state=0)



complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['AdaBoost_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('AdaBoost_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('AdaBoost_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('AdaBoost_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('AdaBoost_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('AdaBoost_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('AdaBoost_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)


# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

#n_estimators=10000, 

clf = AdaBoostClassifier()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['AdaBoost_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('AdaBoost_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('AdaBoost_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('AdaBoost_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('AdaBoost_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('AdaBoost_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('AdaBoost_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



!pip install CatBoost

from catboost import CatBoostRegressor

# iterations=700,learning_rate=0.02,depth=12,eval_metric='RMSE',random_seed = 23,bagging_temperature = 0.2,od_type='Iter',

#                              metric_period = 75,

#                              od_wait=100

        

complex_model_1 = CatBoostRegressor()

complex_model_1= DecisionTreeRegressor(random_state=0)

complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['CB_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('CB_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('CB_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('CB_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('CB_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('CB_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('CB_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

!pip install catboost

from catboost import CatBoostClassifier



clf = CatBoostClassifier(

    iterations=1000, 

    learning_rate=0.1, 

    #verbose=5,

    #loss_function='CrossEntropy'

)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['CB_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('CB_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('CB_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('CB_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('CB_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('CB_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('CB_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



from sklearn.neural_network import MLPRegressor



complex_model_1 = MLPRegressor(random_state=42, max_iter=500)





complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['ANN_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('ANN_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('ANN_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('ANN_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('ANN_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('ANN_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('ANN_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

!pip install catboost

from catboost import CatBoostClassifier



from sklearn.neural_network import MLPClassifier

clf =MLPClassifier(solver='lbfgs', alpha=1e-5,

                     hidden_layer_sizes=(16, 16), random_state=100)



clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['ANN_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('ANN_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('ANN_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('ANN_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('ANN_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('ANN_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('ANN_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

complex_model_1= DecisionTreeRegressor(random_state=0)





complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['KNN_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('KNN_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('KNN_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('KNN_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('KNN_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('KNN_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('KNN_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
# df['bpm']=MLR_y_pred_entire_data

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

clf =KNeighborsClassifier(n_neighbors=1)



clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['KNN_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('KNN_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('KNN_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('KNN_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('KNN_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('KNN_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('KNN_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



import xgboost as xgb

from xgboost import plot_importance

from sklearn.tree import DecisionTreeRegressor

complex_model_1= DecisionTreeRegressor(random_state=0)



complex_model_1.fit(X_train, y_train)





complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['DT_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('DT_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('DT_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('DT_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('DT_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('DT_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('DT_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from sklearn import tree

clf = tree.DecisionTreeClassifier()



clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['DT_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('DT_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('DT_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('DT_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('DT_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('DT_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('DT_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



import xgboost as xgb

from xgboost import plot_importance

from sklearn.tree import DecisionTreeRegressor

complex_model_1= LinearRegression()

complex_model_1.fit(X_train, y_train)





complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['Linear_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('Linear_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('Linear_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('Linear_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('Linear_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('Linear_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('Linear_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['bpm']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from sklearn import tree

clf = GradientBoostingClassifier(random_state=1000, learning_rate=0.1,n_estimators=500)



clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['GB_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('GB_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('GB_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('GB_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('GB_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('GB_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('GB_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))
data=pd.read_csv('../input/bci-final/bci_final.csv')

print(data.head())

X = data.loc[:, data.columns != 'beat-per-15 sec']

y=data['beat-per-15 sec']

X.head()

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



import xgboost as xgb

from xgboost import plot_importance

from sklearn.tree import DecisionTreeRegressor

complex_model_1= LinearRegression()

complex_model_1.fit(X_train, y_train)





complex_model_1.fit(X_train, y_train)



pred = complex_model_1.predict(X_test)

rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))

r = reg_evaluation.shape[0]

reg_evaluation.loc[r] = ['Linear_reg','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]





# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('Linear_reg_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('Linear_reg_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('Linear_reg_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('Linear_reg_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('Linear_reg_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('Linear_reg_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



print(MLR_y_pred_entire_data)

data['beat-per-15 sec']=MLR_y_pred_entire_data

print(data)

reg_evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)

X = data.loc[:, data.columns != 'state']

y=data['state']

X.head()



from sklearn.naive_bayes import GaussianNB



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from sklearn import tree

clf =GaussianNB()



clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = classification_evaluation.shape[0]

classification_evaluation.loc[r] = ['NB_Classification',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]





complex_model_1=clf

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('NB_Classification_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('NB_Classification_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('NB_Classification_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('NB_Classification_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('NB_Classification_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('NB_Classification_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



print(importances)

classification_evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
import seaborn as sns

quantitative_features_list1 = data.columns.values

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)
# test=pd.read_csv('../input/experimental-design/experimental_design.csv')

# test.head()

# print("actual result")

# print(test['state'])

# lst=test['state']

# test= test.loc[:, test.columns != 'state']

# test = scaler.fit_transform(test)

# test_result = complex_model_1.predict(test)

# print("predicted result")

# print(test_result)
test=pd.read_csv('../input/experimental-design/experimental_design.csv')

ct=test

# print(test.head())

# print("actual result")

# print(test['state'])

lst=test['state']

test= test.loc[:, test.columns != 'state']

test = scaler.fit_transform(test)

test_result = complex_model_1.predict(test)

#print("predicted result")

#print(test_result)



        





ct['actual']=ct['state']

ct['predicted']=test_result

ct['beat-per-15 sec']=ct['beat-per-15 sec']/10

ct=ct.drop(columns=['alphaLow','alphaHigh','betaLow','betaHigh','state'])

ct.head()

ct.plot(kind='bar',figsize=(25, 10),alpha=1,fontsize=18,stacked=False)



actual=0

predicted=0

total=len(ct['actual'])

for i in range(len(ct['actual'])):

    if (ct['actual'][i]==1):

        actual=actual+1

    if (ct['predicted'][i]==1):

        predicted=predicted+1

print("Percentage of ENGAGED in actual data: {}".format((actual/total)*100))

print("Percentage of ENGAGED in predicted data: {}".format((predicted/total)*100))

print()

print("Percentage of UNCONCERNED in actual data: {}".format(((total-actual)/total)*100))

print("Percentage of UNCONCERNED in predicted data: {}".format(((total-predicted)/total)*100))