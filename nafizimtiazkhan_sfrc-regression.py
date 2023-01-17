



%matplotlib inline

import pandas as pd

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

from sklearn.metrics import mean_absolute_error

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv('../input/sfrc-dataset/sfrc_beams_with_outliers_revised (1).csv')

df.head()

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)



df= df[['a/d']+['p']+['fc']+['lf/df']+['Vf']+['F']+['Type']+['Vu']]

df["Type"] = df["Type"].astype('category')

df['Type']=df['Type'].cat.codes

df["Type"]=df["Type"].astype('float')





df['a/d'] = np.log(df['a/d'])

df['p'] = np.log(df['p'])

df['fc'] = np.log(np.sqrt(df['fc']))

df['lf/df'] = np.log(df['lf/df'])

df['Vf'] = np.log(df['Vf'])

df['F'] = np.log(df['F'])



df['Type'] = np.log(df['Type'])

df=df.round(decimals=6)

#df['Vu'] = np.log10(df['Vu'])

df.to_csv('test.csv')



df.head()



df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.mean())



data=df

X = data.loc[:, data.columns != 'Vu']

y=data['Vu']

X.head()
data.info()
y.head()
print(data.shape)
# corr = data.corr() 

# plt.figure(figsize=(12, 10))



# sns.heatmap(corr[(corr >= 0.0) | (corr <= -0.0)], 

#             cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

#             annot=True, annot_kws={"size": 8}, square=True);
# Plottinf correlation above or below 0.5







# corr = data.corr() # We already examined SalePrice correlations

# plt.figure(figsize=(12, 10))



# sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 

#             cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

#             annot=True, annot_kws={"size": 8}, square=True);
# import seaborn as sns

# quantitative_features_list1 = ['a/d', 'p', 'fc', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

# #quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

# data_plot_data=data_mod_num = data[quantitative_features_list1]

# sns.pairplot(data_plot_data)
evaluation = pd.DataFrame({'Model': [],

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



evaluation2 = pd.DataFrame({'Model': [],

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

def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)



features = list(data.columns.values)

print(features)

features=  ['a/d', 'p', 'fc', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

features2=  ['a/d', 'p', 'fc', 'lf/df', 'Vf', 'F', 'Type']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
print("for linear Regression")

complex_model_1 = linear_model.LinearRegression(normalize=True)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression-1','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['Linear Regression','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['Linear Regression','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['Linear Regression','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['Linear Regression','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['Linear Regression','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['Linear Regression','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['Linear Regression','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['Linear Regression','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('MLR_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('MLR_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('MLR_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('MLR_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('MLR_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('MLR_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



K=X



features = list(X.columns.values)

importances = complex_model_1.coef_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)



#equation for Linear Regression

print('Intercept: {}'.format(complex_model_1.intercept_))

print('Coefficients: {}'.format(complex_model_1.coef_))




# complex_model_R1 = linear_model.Ridge(alpha=0.01, random_state=20)

# complex_model_R1.fit(X_train, y_train)



# pred1 = complex_model_R1.predict(X_test)

# rmsecm1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred1)),'.3f'))

# rtrcm1 = float(format(complex_model_R1.score(X_train, y_train),'.3f'))





# artrcm1 = float(format(adjustedR2(complex_model_R1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm1 = float(format(complex_model_R1.score(X_test, y_test),'.3f'))

# artecm1 = float(format(adjustedR2(complex_model_R1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv1 = float(format(cross_val_score(complex_model_R1,X_train, y_train,cv=10).mean(),'.3f'))





# r = evaluation.shape[0]

# evaluation.loc[r] = ['Ridge Regression','alpha=1, all features',rmsecm1,rtrcm1,artrcm1,rtecm1,artecm1,cv1]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
print("For ridge regression")

complex_model_1 = linear_model.Ridge(alpha=0.01, random_state=20)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['ridge Regression-1','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['ridge Regression','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['ridge Regression','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['ridge Regression','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['ridge Regression','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['ridge Regression','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['ridge Regression','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['ridge Regression','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['ridge Regression','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]



# # Print the predicted and actual value for the test set

# Ridge_y_test_prediction= complex_model_R1.predict(X_test)

# np.savetxt('Ridge_test_prediction.csv', Ridge_y_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('Ridge_test_actual.csv', y_test, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

# np.savetxt('MLR_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('MLR_entire_actual.csv', y, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# Ridge_y_pred_entire_data = complex_model_R1.predict(X_standardized)

# np.savetxt('Ridge_entire_prediction.csv', Ridge_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('Ridge_entire_actual.csv', y, delimiter=',', fmt='%s')
# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('Ridge_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('Ridge_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('Ridge_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('Ridge_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('Ridge_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('Ridge_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.coef_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)



#equation for Linear Regression

print('Intercept: {}'.format(complex_model_1.intercept_))

print('Coefficients: {}'.format(complex_model_1.coef_))
# complex_model_L1 = linear_model.Lasso(alpha=0.00001)

# complex_model_L1.fit(X_train, y_train)



# pred1 = complex_model_L1.predict(X_test)

# rmsecm1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred1)),'.3f'))

# rtrcm1 = float(format(complex_model_L1.score(X_train, y_train),'.3f'))

# artrcm1 = float(format(adjustedR2(complex_model_L1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm1 = float(format(complex_model_L1.score(X_test, y_test),'.3f'))

# artecm1 = float(format(adjustedR2(complex_model_L1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv1 = float(format(cross_val_score(complex_model_L1,X_train, y_train,cv=10).mean(),'.3f'))



# cv2 = cross_val_score(complex_model_L1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')



# r = evaluation.shape[0]

# evaluation.loc[r] = ['Lasso Regression','alpha=1, all features',rmsecm1,rtrcm1,artrcm1,rtecm1,artecm1,cv1]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)

# print('RMSE for train')

# print(cv2)
complex_model_1 = linear_model.Lasso(alpha=0.00001)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['LASSO Regression-1','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['LASSO Regression','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['LASSO Regression','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['LASSO Regression','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['LASSO Regression','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['LASSO Regression','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['LASSO Regression','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['LASSO Regression','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['LASSO Regression','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('Lasso_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('Lasso_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('Lasso_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('Lasso_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('Lasso_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('Lasso_entire_actual.csv', y, delimiter=',', fmt='%s')





import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.coef_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)



#equation for Linear Regression

print('Intercept: {}'.format(complex_model_1.intercept_))

print('Coefficients: {}'.format(complex_model_1.coef_))


# knnreg = KNeighborsRegressor(n_neighbors=2)

# knnreg.fit(X_train, y_train)



# pred = knnreg.predict(X_test)

# rmseknn2 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrknn2 = float(format(knnreg.score(X_train, y_train),'.3f'))

# artrknn2 = float(format(adjustedR2(knnreg.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rteknn2 = float(format(knnreg.score(X_test, y_test),'.3f'))

# arteknn2 = float(format(adjustedR2(knnreg.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv2 = float(format(cross_val_score(knnreg,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# # evaluation.loc[r] = ['KNN Regression','k=1, all features',rmseknn1,rtrknn1,artrknn1,rteknn1,arteknn1,cv1]

# evaluation.loc[r+1] = ['KNN Regression','k=2, all features',rmseknn2,rtrknn2,artrknn2,rteknn2,arteknn2,cv2]

# # evaluation.loc[r+2] = ['KNN Regression','k=3, all features',rmseknn3,rtrknn3,artrknn3,rteknn3,arteknn3,cv3]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
complex_model_1 = KNeighborsRegressor(n_neighbors=2)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['kNN Regression','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['kNN Regression','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['kNN Regression','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['kNN Regression','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['kNN Regression','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['kNN Regression','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['kNN Regression','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['kNN Regression','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['kNN Regression','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# # Print the predicted and actual value for the test set

# knnreg_test_prediction= knnreg.predict(X_test)

# np.savetxt('KNN_test_prediction.csv', knnreg_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('KNN_test_actual.csv', y_test, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# knnreg_pred_entire_data = knnreg.predict(X_standardized)

# np.savetxt('KNN_entire_prediction.csv', knnreg_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('KNN_entire_actual.csv', y, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# knnreg_y_pred_entire_data = knnreg.predict(X_standardized)

# np.savetxt('KNN_entire_prediction.csv',knnreg_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('KNN_entire_actual.csv', y, delimiter=',', fmt='%s')
# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('KNN_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('KNN_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('KNN_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('KNN_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('KNN_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('KNN_entire_actual.csv', y, delimiter=',', fmt='%s')







import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





# features = list(X.columns.values)

# importances = complex_model_1.metric_params

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)



# from sklearn.svm import SVR

# SVR_model=SVR(kernel='rbf', C=21, degree=3)



# SVR_model.fit(X_train, y_train)



# pred = SVR_model.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(SVR_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(SVR_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(SVR_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(SVR_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(SVR_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['Support Vector Reg','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)

print("For SVR: ")

from sklearn.svm import SVR

complex_model_1 = SVR(kernel='rbf', C=21, degree=3)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['SVR Regression','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['SVR','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['SVR','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['SVR','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['SVR','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['SVR','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['SVR','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['SVR','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['SVR','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

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





# features = list(X.columns.values)

# importances = complex_model_1.metric_params

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)
    

# from sklearn.tree import DecisionTreeRegressor

# DT_model= DecisionTreeRegressor(random_state=27)

# #27 0.742

# DT_model.fit(X_train, y_train)



# pred = DT_model.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(DT_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(DT_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(DT_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(DT_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(DT_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['Decision Tree','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)   
print("For Decision Tree regression")

from sklearn.tree import DecisionTreeRegressor

complex_model_1 = DecisionTreeRegressor(random_state=27)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['DT','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['DT','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['DT','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['DT','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['DT','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['DT','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['DT','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['DT','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['DT','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# # Print the predicted and actual value for the test set

# DT_y_test_prediction= DT_model.predict(X_test)

# np.savetxt('DT_test_prediction.csv', DT_y_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('DT_test_actual.csv', y_test, delimiter=',', fmt='%s')



# # Print the predicted and actual value for the traing set

# DT_y_train_prediction= DT_model.predict(X_train)

# np.savetxt('DT_train_prediction.csv', DT_y_train_prediction, delimiter=',', fmt='%s')

# np.savetxt('DT_train_actual.csv', y_train, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# DT_y_pred_entire_data = DT_model.predict(X_standardized)

# np.savetxt('DT_entire_prediction.csv', DT_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('DT_entire_actual.csv', y, delimiter=',', fmt='%s')
# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('DT_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('DT_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('DT_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('DT_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('DT_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('DT_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
# from sklearn.ensemble import RandomForestRegressor

# RF_model= RandomForestRegressor(random_state=500, n_estimators=200)

# #500 200 0813

# RF_model.fit(X_train, y_train)



# pred = RF_model.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(RF_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(RF_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(RF_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(RF_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(RF_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['Random Forest','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
print("For Random Forest regression")

from sklearn.ensemble import RandomForestRegressor

complex_model_1 = RandomForestRegressor(random_state=500, n_estimators=200)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['RF','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['RF','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['RF','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['RF','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['RF','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['RF','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['RF','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['RF','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['RF','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('RF_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('RF_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('RF_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('RF_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('RF_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('RF_entire_actual.csv', y, delimiter=',', fmt='%s')





import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
RF_model=complex_model_1

from sklearn.inspection import permutation_importance

r = permutation_importance(RF_model, X_test, y_test,n_repeats=50,random_state=0)

#if r.importances_mean[i] - 2 * r.importances_std[i] > 0:

for i in r.importances_mean.argsort()[::-1]:

  print(f"{features[i]:<20}"

  f"{r.importances_mean[i]:.3f}"

  f" +/- {r.importances_std[i]:.3f}")


# Extract single tree

estimator = RF_model.estimators_[5]



from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = features2,

                class_names = features2,

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')




# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(100, 510, num = 11)]

# max_depth.append(None)

# # Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree

# bootstrap = [True, False]

# # Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}





# # Use the random grid to search for best hyperparameters

# # First create the base model to tune

# rf = RandomForestRegressor()

# # Random search of parameters, using 3 fold cross validation, 

# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model

# rf_random.fit(X_train, y_train)



# rf_random.best_params_

from sklearn.ensemble import RandomForestRegressor

RF_model_opt= RandomForestRegressor(n_estimators= 136, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto',  bootstrap= 'True')



RF_model_opt.fit(X_train, y_train)



# knnreg.fit(X_train, y_train)

# pred = complex_model_1.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test, pred)),'.3f'))

# rtrcm = float(format(complex_model_1.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(complex_model_1.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))





pred = RF_model_opt.predict(X_test)

rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rtrcm = float(format(RF_model_opt.score(X_train, y_train),'.3f'))

artrcm = float(format(adjustedR2(RF_model_opt.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

rtecm = float(format(RF_model_opt.score(X_test, y_test),'.3f'))

artecm = float(format(adjustedR2(RF_model_opt.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

cv = float(format(cross_val_score(RF_model_opt,X_train, y_train,cv=10).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Random Forest - opt','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
features = list(X.columns.values)



importances = RF_model_opt.feature_importances_

import numpy as np

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
importances
# import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)



# import xgboost as xgb

# from xgboost import plot_importance

# XGB_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.16, gamma=0, subsample=0.75,

#                            colsample_bytree=1, max_depth=7)

# # XGB_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.20, gamma=0, subsample=0.75,

# #                            colsample_bytree=1, max_depth=7)









# XGB_model.fit(X_train, y_train)



# pred = XGB_model.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(XGB_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(XGB_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(XGB_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(XGB_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(XGB_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['XG Boost','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
print("For XGBoost regression")

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import xgboost as xgb

from xgboost import plot_importance

complex_model_1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.16, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)



complex_model_1.fit(X_train, y_train)

testing=pd.read_csv('revised.csv');

pred2=complex_model_1.predict(X_test)







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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['XGBOOST Regression','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['XGBOOST Regression','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['XGBOOST Regression','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['XGBOOST Regression','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['XGBOOST Regression','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['XGBOOST Regression','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['XGBOOST Regression','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['XGBOOST Regression','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['XGBOOST Regression','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# # Print the predicted and actual value for the test set

# XG_y_test_prediction= XGB_model.predict(X_test)

# np.savetxt('XG_test_prediction.csv', XG_y_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('XG_test_actual.csv', y_test, delimiter=',', fmt='%s')



# # Print the predicted and actual value for the traing set

# XG_y_train_prediction= XGB_model.predict(X_train)

# np.savetxt('XG_train_prediction.csv', XG_y_train_prediction, delimiter=',', fmt='%s')

# np.savetxt('XG_train_actual.csv', y_train, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# XG_y_pred_entire_data = XGB_model.predict(X_standardized)

# np.savetxt('XG_entire_prediction.csv', XG_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('XG_entire_actual.csv', y, delimiter=',', fmt='%s')
# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('XgBOOST_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('XgBOOST_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('XgBOOST_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('XgBOOST_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('XgBOOST_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('XgBOOST_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

# #for tuning parameters

# #from sklearn.grid_search import GridSearchCV   #Performing grid search

# from sklearn.model_selection import GridSearchCV



# # parameters_for_testing = {

# #     'colsample_bytree':[0.3,0.4,0.5,0.6,0.7, 0.8],

# #     "gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4, 0.03 ],

# #     'min_child_weight':[1.5,6,3,7,10],

# #     'learning_rate':  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

# #     'max_depth':[ 3, 4, 5, 6, 8, 10, 12, 15],

# #     'n_estimators':[100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300],

# #     'reg_alpha':[1e-5, 1e-2,  0.75],

# #     'reg_lambda':[1e-5, 1e-2, 0.45],

# #     'subsample':[0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],

# #     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   

# # }



# parameters_for_testing = {

#     "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

#  "max_depth"        : [ 3, 4, 5, 6],

#  "min_child_weight" : [ 1, 3, 5, 7 ],

#  "gamma"            : [ 0.0, 0.1],

#  "colsample_bytree" : [ 0.3, 0.4, 0.5] 

# }





                    

# xgb_model = xgb.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,

#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)



# gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')

# gsearch1.fit(train_data_dm[features],train_data_dm['τmax'])

# print (gsearch1.grid_scores_)

# print('best params')

# print (gsearch1.best_params_)

# print('best score')

# print (gsearch1.best_score_)#for tuning parameters

# parameters_for_testing = {

#    'colsample_bytree':[0.4,0.6,0.8],

#    'gamma':[0,0.03,0.1,0.3],

#    'min_child_weight':[1.5,6,10],

#    'learning_rate':[0.1,0.07],

#    'max_depth':[3,5],

#    'n_estimators':[10000],

#    'reg_alpha':[1e-5, 1e-2,  0.75],

#    'reg_lambda':[1e-5, 1e-2, 0.45],

#    'subsample':[0.6,0.95]  

# }





# {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

#  "max_depth"        : [ 3, 4, 5, 6],

#  "min_child_weight" : [ 1, 3, 5, 7 ],

#  "gamma"            : [ 0.0, 0.1],

#  "colsample_bytree" : [ 0.3, 0.4, 0.5] }

                    

# # xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,

# #     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)



# gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')

# gsearch1.fit(X_train, y_train)

# print (gsearch1.grid_scores_)

# print('best params')

# print (gsearch1.best_params_)

# print('best params printed')

# print('best score')

# print (gsearch1.best_score_)
# print('best params')

# print (gsearch1.best_params_)

# print('best params printed')

# print('best score')

# print (gsearch1.best_score_)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import xgboost as xgb

from xgboost import plot_importance

XGB_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.20, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)



XGB_model.fit(X_train, y_train)



pred = XGB_model.predict(X_test)

rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

rtrcm = float(format(XGB_model.score(X_train, y_train),'.3f'))

artrcm = float(format(adjustedR2(XGB_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

rtecm = float(format(XGB_model.score(X_test, y_test),'.3f'))

artecm = float(format(adjustedR2(XGB_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

cv = float(format(cross_val_score(XGB_model,X_train, y_train,cv=10).mean(),'.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['XG Boost','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)


# from sklearn.ensemble import AdaBoostRegressor





# ADB_model = AdaBoostRegressor(random_state=30, n_estimators=100)

# ADB_model.fit(X_train, y_train)





# pred = ADB_model.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(ADB_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(ADB_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(ADB_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(ADB_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(ADB_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['AdaBoost','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)

from sklearn.ensemble import AdaBoostRegressor

print("For Adaboost regression")

complex_model_1 = AdaBoostRegressor(random_state=30, n_estimators=100)

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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['ADABOOST Regression','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['ADABOOST Regression','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['ADABOOST Regression','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['ADABOOST Regression','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['ADABOOST Regression','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['ADABOOST Regression','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['ADABOOST Regression','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['ADABOOST Regression','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['ADABOOST Regression','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# # Print the predicted and actual value for the test set

# AD_y_test_prediction= ADB_model.predict(X_test)

# np.savetxt('AD_test_prediction.csv', AD_y_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('AD_test_actual.csv', y_test, delimiter=',', fmt='%s')



# # Print the predicted and actual value for the traing set

# AD_y_train_prediction= ADB_model.predict(X_train)

# np.savetxt('AD_train_prediction.csv', AD_y_train_prediction, delimiter=',', fmt='%s')

# np.savetxt('AD_train_actual.csv', y_train, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# AD_y_pred_entire_data = ADB_model.predict(X_standardized)



# np.savetxt('AD_entire_prediction.csv', AD_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('AD_entire_actual.csv', y, delimiter=',', fmt='%s')
# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('AdaBOOST_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('AdaBOOST_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('AdaBOOST_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('AdaBOOST_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('AdaBOOST_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('AdaBOOST_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
# !pip3 install catboost

# from catboost import CatBoostRegressor

# CB_model = CatBoostRegressor(iterations=700,learning_rate=0.02,depth=12,eval_metric='RMSE',random_seed = 23,bagging_temperature = 0.2,od_type='Iter',

#                              metric_period = 75,

#                              od_wait=100)



# CB_model.fit(X_train, y_train)



# pred = CB_model.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(CB_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(CB_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(CB_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(CB_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(CB_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['CatBoost','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)

print("For CatBoost regression")

!pip3 install catboost

from catboost import CatBoostRegressor

complex_model_1 = CatBoostRegressor(iterations=700,learning_rate=0.02,depth=12,eval_metric='RMSE',random_seed = 23,bagging_temperature = 0.2,od_type='Iter',

                             metric_period = 75,

                             od_wait=100)



complex_model_1.fit(X_train, y_train)





# pred = complex_model_1.predict(X_test)

# rmse_train = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_train), y_train)),'.3f'))

# r2_train = float(format(complex_model_1.score(X_train, y_train),'.3f'))

# ar2_train = float(format(adjustedR2(complex_model_1.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# mae_train=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_train), y_train)),'.3f'))



# rmse_test = float(format(np.sqrt(metrics.mean_squared_error(complex_model_1.predict(X_test), y_test)),'.3f'))

# r2_test = float(format(complex_model_1.score(X_test, y_test),'.3f'))

# ar2_test = float(format(adjustedR2(complex_model_1.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# mae_test=float(format((metrics.mean_absolute_error(complex_model_1.predict(X_test), y_test)),'.3f'))



# cv = float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10).mean(),'.3f'))



# cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

# cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



# cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

# cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



# cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

# cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



# cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

# cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



# cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

# cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



# cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

# cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



# cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

# cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



# cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

# cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



# r = evaluation.shape[0]

# evaluation.loc[r] = ['Multiple Regression-1','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



# r = evaluation2.shape[0]

# evaluation2.loc[r] = ['CATBOOST','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

# evaluation2.loc[r+1] = ['CATBOOST','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

# evaluation2.loc[r+2] = ['CATBOOST','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

# evaluation2.loc[r+3] = ['CATBOOST','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

# evaluation2.loc[r+4] = ['CATBOOST','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

# evaluation2.loc[r+5] = ['CATBOOST','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

# evaluation2.loc[r+6] = ['CATBOOST','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

# evaluation2.loc[r+7] = ['CATBOOST','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# # Print the predicted and actual value for the test set

# CB_y_test_prediction= CB_model.predict(X_test)

# np.savetxt('CB_test_prediction.csv', CB_y_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('CB_test_actual.csv', y_test, delimiter=',', fmt='%s')



# # Print the predicted and actual value for the traing set

# CB_y_train_prediction= CB_model.predict(X_train)

# np.savetxt('CB_train_prediction.csv', CB_y_train_prediction, delimiter=',', fmt='%s')

# np.savetxt('CB_train_actual.csv', y_train, delimiter=',', fmt='%s')



# X_standardized = scaler.transform(X)

# CB_y_pred_entire_data = CB_model.predict(X_standardized)

# np.savetxt('CB_entire_prediction.csv', CB_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('CB_entire_actual.csv', y, delimiter=',', fmt='%s')
# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('CatBOOST_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('CatBOOST_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('CatBOOST_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('CatBOOST_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('CatBOOST_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('CatBOOST_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()





features = list(X.columns.values)

importances = complex_model_1.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)

# from sklearn.neural_network import MLPRegressor



# # ADB_model = AdaBoostRegressor(random_state=30, n_estimators=100)

# # ADB_model.fit(X_train, y_train)

# regr = MLPRegressor(random_state=1, learning_rate='adaptive', max_iter=500).fit(X_train, y_train)



# pred = regr.predict(X_test)

# rmsecm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))

# rtrcm = float(format(ADB_model.score(X_train, y_train),'.3f'))

# artrcm = float(format(adjustedR2(ADB_model.score(X_train, y_train),X_train.shape[0],len(features)),'.3f'))

# rtecm = float(format(ADB_model.score(X_test, y_test),'.3f'))

# artecm = float(format(adjustedR2(ADB_model.score(X_test, y_test),X_test.shape[0],len(features)),'.3f'))

# cv = float(format(cross_val_score(ADB_model,X_train, y_train,cv=10).mean(),'.3f'))



# r = evaluation.shape[0]

# evaluation.loc[r] = ['ANN','All features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]

# evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)
print("For ANN regression")

from sklearn.neural_network import MLPRegressor



complex_model_1 = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)





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



cv_train_rmse=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error')

cv_train_rmse_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_root_mean_squared_error').mean(),'.3f'))



cv_train_r2=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2')

cv_train_r2_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),'.3f'))



cv_train_ar2=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2'),X_train.shape[0],len(features))

cv_train_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='r2').mean(),X_train.shape[0],len(features))



cv_train_mae=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error')

cv_train_mae_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='neg_mean_absolute_error').mean()



cv_test_rmse=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error')

cv_test_rmse_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_root_mean_squared_error').mean()



cv_test_r2=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2')

cv_test_r2_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean()



cv_test_ar2=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2'),X_test.shape[0],len(features))

cv_test_ar2_m=adjustedR2(cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='r2').mean(),X_test.shape[0],len(features))



cv_test_mae=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error')

cv_test_mae_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='neg_mean_absolute_error').mean()



r = evaluation.shape[0]

evaluation.loc[r] = ['ANN Regression-1','All features',rmse_train,r2_train,ar2_train,mae_train,rmse_test,r2_test,ar2_test,mae_test,cv]

evaluation.sort_values(by = '10-Fold Cross Validation', ascending=False)



r = evaluation2.shape[0]

evaluation2.loc[r] = ['ANN','Train RMSE',float(format(cv_train_rmse[0],'.3f')),float(format(cv_train_rmse[1],'.3f')),float(format(cv_train_rmse[2],'.3f')),float(format(cv_train_rmse[3],'.3f')),float(format(cv_train_rmse[4],'.3f')),float(format(cv_train_rmse[5],'.3f')),float(format(cv_train_rmse[6],'.3f')),float(format(cv_train_rmse[7],'.3f')),float(format(cv_train_rmse[8],'.3f')),float(format(cv_train_rmse[9],'.3f')),float(format(cv_train_rmse_m,'.3f'))]

evaluation2.loc[r+1] = ['ANN','Train R2',float(format(cv_train_r2[0],'.3f')),float(format(cv_train_r2[1],'.3f')),float(format(cv_train_r2[2],'.3f')),float(format(cv_train_r2[3],'.3f')),float(format(cv_train_r2[4],'.3f')),float(format(cv_train_r2[5],'.3f')),float(format(cv_train_r2[6],'.3f')),float(format(cv_train_r2[7],'.3f')),float(format(cv_train_r2[8],'.3f')),float(format(cv_train_r2[9],'.3f')),float(format(cv_train_r2_m,'.3f'))]

evaluation2.loc[r+2] = ['ANN','Train ar2',float(format(cv_train_ar2[0],'.3f')),float(format(cv_train_ar2[1],'.3f')),float(format(cv_train_ar2[2],'.3f')),float(format(cv_train_ar2[3],'.3f')),float(format(cv_train_ar2[4],'.3f')),float(format(cv_train_ar2[5],'.3f')),float(format(cv_train_ar2[6],'.3f')),float(format(cv_train_ar2[7],'.3f')),float(format(cv_train_ar2[8],'.3f')),float(format(cv_train_ar2[9],'.3f')),float(format(cv_train_ar2_m,'.3f'))]

evaluation2.loc[r+3] = ['ANN','Train mae',float(format(cv_train_mae[0],'.3f')),float(format(cv_train_mae[1],'.3f')),float(format(cv_train_mae[2],'.3f')),float(format(cv_train_mae[3],'.3f')),float(format(cv_train_mae[4],'.3f')),float(format(cv_train_mae[5],'.3f')),float(format(cv_train_mae[6],'.3f')),float(format(cv_train_mae[7],'.3f')),float(format(cv_train_mae[8],'.3f')),float(format(cv_train_mae[9],'.3f')),float(format(cv_train_mae_m,'.3f'))]

evaluation2.loc[r+4] = ['ANN','Test RMSE',float(format(cv_test_rmse[0],'.3f')),float(format(cv_test_rmse[1],'.3f')),float(format(cv_test_rmse[2],'.3f')),float(format(cv_test_rmse[3],'.3f')),float(format(cv_test_rmse[4],'.3f')),float(format(cv_test_rmse[5],'.3f')),float(format(cv_test_rmse[6],'.3f')),float(format(cv_test_rmse[7],'.3f')),float(format(cv_test_rmse[8],'.3f')),float(format(cv_test_rmse[9],'.3f')),float(format(cv_test_rmse_m,'.3f'))]

evaluation2.loc[r+5] = ['ANN','Test R2',float(format(cv_test_r2[0],'.3f')),float(format(cv_test_r2[1],'.3f')),float(format(cv_test_r2[2],'.3f')),float(format(cv_test_r2[3],'.3f')),float(format(cv_test_r2[4],'.3f')),float(format(cv_test_r2[5],'.3f')),float(format(cv_test_r2[6],'.3f')),float(format(cv_test_r2[7],'.3f')),float(format(cv_test_r2[8],'.3f')),float(format(cv_test_r2[9],'.3f')),float(format(cv_test_r2_m,'.3f'))]

evaluation2.loc[r+6] = ['ANN','Test ar2',float(format(cv_test_ar2[0],'.3f')),float(format(cv_test_ar2[1],'.3f')),float(format(cv_test_ar2[2],'.3f')),float(format(cv_test_ar2[3],'.3f')),float(format(cv_test_ar2[4],'.3f')),float(format(cv_test_ar2[5],'.3f')),float(format(cv_test_ar2[6],'.3f')),float(format(cv_test_ar2[7],'.3f')),float(format(cv_test_ar2[8],'.3f')),float(format(cv_test_ar2[9],'.3f')),float(format(cv_test_ar2_m,'.3f'))]

evaluation2.loc[r+7] = ['ANN','Train mae',float(format(cv_test_mae[0],'.3f')),float(format(cv_test_mae[1],'.3f')),float(format(cv_test_mae[2],'.3f')),float(format(cv_test_mae[3],'.3f')),float(format(cv_test_mae[4],'.3f')),float(format(cv_test_mae[5],'.3f')),float(format(cv_test_mae[6],'.3f')),float(format(cv_test_mae[7],'.3f')),float(format(cv_test_mae[8],'.3f')),float(format(cv_test_mae[9],'.3f')),float(format(cv_test_mae_m,'.3f')) ]

# # Print the predicted and actual value for the test set

# CB_y_test_prediction= regr.predict(X_test)

# np.savetxt('ANN_test_prediction.csv', CB_y_test_prediction, delimiter=',', fmt='%s')

# np.savetxt('ANN_test_actual.csv', y_test, delimiter=',', fmt='%s')









# # Print the predicted and actual value for the traing set

# CB_y_train_prediction= regr.predict(X_train)

# np.savetxt('ANN_train_prediction.csv', CB_y_train_prediction, delimiter=',', fmt='%s')

# np.savetxt('ANN_train_actual.csv', y_train, delimiter=',', fmt='%s')







# X_standardized = scaler.transform(X)

# CB_y_pred_entire_data = regr.predict(X_standardized)

# np.savetxt('ANN_entire_prediction.csv', CB_y_pred_entire_data, delimiter=',', fmt='%s')

# np.savetxt('ANN_entire_actual.csv', y, delimiter=',', fmt='%s')


# Print the predicted and actual value for the test set

MLR_y_test_prediction= complex_model_1.predict(X_test)

np.savetxt('ANN_test_prediction.csv', MLR_y_test_prediction, delimiter=',', fmt='%s')

np.savetxt('ANN_test_actual.csv', y_test, delimiter=',', fmt='%s')



# Print the predicted and actual value for the traing set

MLR_y_train_prediction= complex_model_1.predict(X_train)

np.savetxt('ANN_train_prediction.csv', MLR_y_train_prediction, delimiter=',', fmt='%s')

np.savetxt('ANN_train_actual.csv', y_train, delimiter=',', fmt='%s')



X_standardized = scaler.transform(X)

MLR_y_pred_entire_data = complex_model_1.predict(X_standardized)

np.savetxt('ANN_entire_prediction.csv', MLR_y_pred_entire_data, delimiter=',', fmt='%s')

np.savetxt('ANN_entire_actual.csv', y, delimiter=',', fmt='%s')



import matplotlib.pyplot as plt

plt.plot(y, MLR_y_pred_entire_data,  'ro')

plt.ylabel('Predicted data')

plt.xlabel('Actual data')

plt.show()



# features = list(X.columns.values)

# importances = complex_model_1.coefs_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)


#equation for Linear Regression

print('Intercept: {}'.format(complex_model_1.intercept_))

print('Coefficients: {}'.format(complex_model_1.coef_))
evaluation.to_csv("model_results_with_cat.csv")

evaluation2.to_csv("cross_val_results_with_cat.csv")