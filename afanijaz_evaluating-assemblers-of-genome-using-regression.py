import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sklearn.metrics as metrics

import pandas_profiling as pdp

import warnings

import adj_helper as helper

import pickle as pk





from math import sqrt

from sklearn import datasets

from sklearn.metrics import mean_squared_error 

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.linear_model import LinearRegression

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import ExtraTreesClassifier



%matplotlib inline



warnings.filterwarnings('ignore')
with pd.ExcelFile('../input/regression-genome-assembly/genome.xlsx') as xlsx:

    plasmid = pd.read_excel(xlsx, 'NBB4 Plasmid')

    hamburgensis = pd.read_excel(xlsx, 'Hamburgensis X14')

    vibrio = pd.read_excel(xlsx, 'Vibrio Cholerae')

    pab1 = pd.read_excel(xlsx, 'PAb1')
plasmid.describe()
plasmid.info()
plasmid.drop("Sr.No",axis=1, inplace = True)

plasmid = plasmid.replace(['-'],'NaN')

plasmid['MARAGAP'] = pd.to_numeric(plasmid['MARAGAP'], downcast='float')

plasmid['Mira'] = plasmid['Mira'].astype(float)

plasmid['Mira2'] = plasmid['Mira2'].astype(float)

plasmid['Maq'] = plasmid['Maq'].astype(float)

plasmid = plasmid.fillna('0')
plasmid = plasmid.set_index('Assembly_Metrics').T.astype(float)

plasmid
plasmid.info()
X = plasmid.iloc[:,0:13]

Y = plasmid[['Order']]
X = MinMaxScaler().fit_transform(X)

X = pd.DataFrame({'Number_of_Contigs': X[:, 0], 'Length_of_Largest_Contig': X[:, 1],'N50': X[:, 2],'N75': X[:, 3], 'N90': X[:, 4], 'NG50': X[:,5], 'NG75': X[:, 6], 'Contigs_greater_and_equal_N50': X[:, 7],'Contigs_greater_and_equal_200': X[:, 8],'Mean': X[:, 9], 'Median': X[:, 10],'Sum_of_the_Contig_Lengths': X[:, 11], 'Coverage':X[:, 12]})

X
data = {'Assembly_Metrics':['Velvet', 'VCAKE', 'SSAKE', 'QSRA','SHARCGS','IDBA','Mira','Mira2','Maq','MARAGAP']} 

data = pd.DataFrame(data)

X['Assembly_Metrics'] = data

X = X.set_index('Assembly_Metrics')

X
plasmid = pd.concat([X, Y], axis=1)

plasmid
a = sns.pairplot(plasmid, x_vars=['Number_of_Contigs','Length_of_Largest_Contig','N50'], y_vars='Order', height=5, aspect=0.9)

b = sns.pairplot(plasmid, x_vars=['N75','Contigs_greater_and_equal_N50','Contigs_greater_and_equal_200'], y_vars='Order', height=5, aspect=0.9)

c = sns.pairplot(plasmid, x_vars=['Mean','Median','Sum_of_the_Contig_Lengths','Coverage'], y_vars='Order', height=5, aspect=0.9)
correlations = plasmid.corr()

correlations
mask = np.triu(np.ones_like(correlations, dtype=np.bool))



plt.figure(figsize=(11,8))

sns.heatmap(correlations*100,mask=mask,annot=True, fmt='.0f' )
X_ = plasmid.drop(['Contigs_greater_and_equal_N50','Contigs_greater_and_equal_200','Median','Sum_of_the_Contig_Lengths','Coverage'],axis=1)

Y_ = plasmid[['Order']]

print(X_.shape)

print(Y_.shape)
X1 = X_.corr()
mask = np.triu(np.ones_like(X1, dtype=np.bool))



plt.figure(figsize=(11,8))

sns.heatmap(X1*100,mask=mask,annot=True, fmt='.0f' )
bestfeatures = SelectKBest(score_func=chi2, k=9)

fit = bestfeatures.fit(X_,Y_)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
feature_imp = ExtraTreesClassifier()

feature_imp.fit(X_,Y_)

print(feature_imp.feature_importances_)
feature_importance = pd.Series(feature_imp.feature_importances_, index=X_.columns)

plt.figure(figsize=(12,8))

feature_importance.nlargest(13).plot(kind='barh')

plt.show()

X__ = X_.drop(['Order','Number_of_Contigs'],axis=1)

predict_plasmid = X_.drop(['Order','Number_of_Contigs'],axis=1)

X__
#profile = pdp.ProfileReport(X__)

#profile
model = LinearRegression()
model = model.fit(X__, Y_)
coefficients = model.coef_

co = pd.DataFrame(coefficients).astype(float)

co
print("Intercept: ", model.intercept_)
pred = model.predict(X__)

pred.astype(int)
r2_regression = model.score(X__, Y_)

print('R^2: {0}'.format(r2_regression))
train_r2_1=r2_score(Y_,model.predict(X__))

test_r2_1=r2_score(Y_,pred)
Xb = X__['N75']

plt.plot(Xb, Y_,'o')

m,b = np.polyfit(Xb,Y_,1)

plt.plot(Xb,m*Xb+b)
Xd = X__['N90']

plt.plot(Xd, Y_,'o')

m,b = np.polyfit(Xd,Y_,1)

plt.plot(Xd,m*Xd+b)

Xf = X__['NG75']

plt.plot(Xf, Y_,'o')

m,b = np.polyfit(Xf,Y_,1)

plt.plot(Xf,m*Xf+b)

r_2_1 = []

for i in range(1, (X__.shape[-1])+1):

    m1=LinearRegression()

    m1.fit(X__.values[:,:i],Y_)

    prd1=m1.predict(X__.values[:,:i])

    r_2_1.append(r2_score(Y_,prd1))

    

plt.figure(figsize=(15,5))

plt.plot(r_2_1);

plt.xlabel('Features')

plt.ylabel('R_2 Score')
plt.figure(figsize=(15,6)) 

visualizer = ResidualsPlot(model,hist=True)

visualizer.fit(X__.values, Y_.values)  

visualizer.score(X__.values, Y_.values)  

visualizer.poof()    
mse_regression = mean_squared_error(Y_,pred)

rmse_regression = sqrt(mse_regression)

mae_regression = metrics.mean_absolute_error(Y_,pred)

vrs_regression = metrics.explained_variance_score(Y_,pred)

r_square_model = model.score(X__, Y_)



print('MEAN SQUARE ERROR : {0:.4f}'.format(mse_regression))

print('ROOT MEAN SQUARE ERROR : {0:.4f}' .format(rmse_regression))

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae_regression))

print('VARIANCE REGRESSION SCORE :{0:.4f}'.format(vrs_regression))

print('Model R Square :{0:.4f}'.format(r_square_model))
pk.dump(model, open("model_linear.pkl","wb"))
model_linear=pk.load(open("model_linear.pkl",'rb'))

predict_linear=model_linear.predict(predict_plasmid)

predict_linear +=1

pred_linear=predict_linear.astype(int).flatten()

Order_reg = pd.DataFrame(pred_linear,columns=['Predicted_Order'])

Order_reg['Assemblers'] = data

Order_reg.sort_values(by='Predicted_Order')
Y_=np.ravel(Y_)
randomregressor = RandomForestRegressor(max_depth=3, random_state=0, max_features=7)

random_regressor = randomregressor.fit(X__,Y_)

random_model=random_regressor.predict(X__)
random_model.astype(int)
r2_forest = randomregressor.score(X__, Y_)

print('R^2: {0:.4f}'.format(r2_forest))
mse_random = mean_squared_error(Y_,random_model)

rmse_random = sqrt(mse_random)

mae_random = metrics.mean_absolute_error(Y_,random_model)

vrs_random = metrics.explained_variance_score(Y_,random_model)

print('MEAN SQUARE ERROR : {0:.4f}'.format(mse_random))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse_random)) 

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae_random))

print('VARIANCE REGRESSION SCORE :{0:.4f}'.format(vrs_random))

print('R Square : {0:.4f}'.format(r2_forest))
pk.dump(randomregressor, open("model_random.pkl","wb"))
model_random=pk.load(open("model_random.pkl",'rb'))

pred_random=model_random.predict(predict_plasmid)

pred_random=pred_random.astype(int)

Order_random = pd.DataFrame(pred_random,columns=['Predicted_Order'])

Order_random['Assemblers'] = data

Order_random.sort_values(by='Predicted_Order')
ridge = Ridge()

parameters = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

ridge_model_params = GridSearchCV(ridge, parameters,scoring='r2',cv=7)

ridge_model_params.fit(X__,Y_)
best_ridge_params = ridge_model_params.best_params_

print("Best Ridge Params : ", best_ridge_params)
ridge_model=Ridge(alpha=0.000001).fit(X__, Y_)

ridge_coefficients= ridge_model.coef_

pd.DataFrame(ridge_coefficients).astype(float)
print('Model Intercept : {0:.4f}'.format(ridge_model.intercept_))
ridge_model_predict=ridge_model.predict(X__)
r_square_ridge = ridge_model.score(X__, Y_)

mse_ridge = mean_squared_error(Y_,ridge_model_predict)

rmse_ridge = sqrt(mse_ridge)

mae_ridge = metrics.mean_absolute_error(Y_,ridge_model_predict)

vrs_ridge = metrics.explained_variance_score(Y_,ridge_model_predict)



print('MEAN SQUARE ERROR : {0:.4f}'.format(mse_ridge))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse_ridge)) 

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae_ridge))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs_ridge))

print('Model R Square: {0:.4f}'.format(r_square_ridge))
pk.dump(ridge_model, open("model_ridge.pkl","wb"))
model_ridge=pk.load(open("model_ridge.pkl",'rb'))

pred_ridge=model_ridge.predict(predict_plasmid)

pred_ridge+=1

pred_ridge = pred_ridge.astype(int)

Order_ridge = pd.DataFrame(pred_ridge,columns=['Predicted_Order'])

Order_ridge['Assemblers'] = data

Order_ridge.sort_values(by='Predicted_Order')
lasso = Lasso()

parameters = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

lasso_model = GridSearchCV(lasso, parameters,scoring='r2',cv=7)

lasso_model = lasso_model.fit(X__,Y_)
best_lasso_params = lasso_model.best_params_

print("Best lasso Params : ", best_lasso_params)
lasso = Lasso()

lasso_model=Lasso(alpha=0.000001).fit(X__, Y_)
lasso_model.coef_

lasso_coefficients= lasso_model.coef_

pd.DataFrame(lasso_coefficients).astype(float)
print('Model Intercept : {0:.4f}'.format(lasso_model.intercept_))
lasso_model_predict=lasso_model.predict(X__)
r_square_lasso = lasso_model.score(X__, Y_)

mse_lasso = mean_squared_error(Y_,lasso_model_predict)

rmse_lasso = sqrt(mse_lasso)

mae_lasso = metrics.mean_absolute_error(Y_,lasso_model_predict)

vrs_lasso = metrics.explained_variance_score(Y_,lasso_model_predict)



print('MEAN SQUARE ERROR : {0:.4f}'.format(mse_lasso))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse_lasso)) 

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae_lasso))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs_lasso))

print('Model R Square: {0:.4f}'.format(r_square_lasso))
pk.dump(lasso_model, open("model_lasso.pkl","wb"))
model_lasso=pk.load(open("model_lasso.pkl",'rb'))

pred_lasso=model_lasso.predict(predict_plasmid)

pred_lasso+=1

pred_lasso = pred_lasso.astype(int)

Order_lasso = pd.DataFrame(pred_lasso,columns=['Predicted_Order'])

Order_lasso['Assemblers'] = data

Order_lasso.sort_values(by='Predicted_Order')
alphas = [0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]

for a in alphas:

    model = ElasticNet(alpha=a).fit(X__,Y_)   

    score = model.score(X__, Y_)

    pred_y = model.predict(X__)

    mse_elastic = mean_squared_error(Y_, pred_y)   

    print("Alpha:{0:.6f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(a, score, mse_elastic, np.sqrt(mse_elastic)))
elastic_model=ElasticNet(alpha=0.000001).fit(X__, Y_)

elastic_model_predict = elastic_model.predict(X__)

score = elastic_model.score(X__,Y_)

mse = mean_squared_error(Y_, elastic_model_predict)

print("R2:{0:.2f}, MSE:{1:.2f}, RMSE:{2:.2f}"

      .format(score, mse, np.sqrt(mse)))
elastic_model.coef_
r_square_elastic = elastic_model.score(X__, Y_)

mse_elastic = mean_squared_error(Y_,elastic_model_predict)

rmse_elastic = sqrt(mse_elastic)

mae_elastic = metrics.mean_absolute_error(Y_,elastic_model_predict)

vrs_elastic = metrics.explained_variance_score(Y_,elastic_model_predict)





print('MEAN SQUARE ERROR : {0:.4f}'.format(mse_elastic))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse_elastic)) 

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae_elastic))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs_elastic))

print('Model R Square: {0:.4f}'.format(r_square_elastic))

pk.dump(elastic_model, open("model_elastic.pkl","wb"))
model_elastic=pk.load(open("model_elastic.pkl",'rb'))

pred_elastic=model_elastic.predict(predict_plasmid)

pred_elastic+=1

pred_elastic = pred_elastic.astype(int)

Order_elastic = pd.DataFrame(pred_elastic,columns=['Predicted_Order'])

Order_elastic['Assemblers'] = data

Order_elastic.sort_values(by='Predicted_Order')
comparison={"Models":["Simple Regression", "Random Forest","Ridge Regression","Lasso Regression","ElasticNET Regression" ],

            "R_Square/Accuracy": [r2_regression, r2_forest,r_square_ridge,r_square_lasso,r_square_elastic],

            "Mean Square Error/Accuracy": [mse_regression, mse_random, mse_ridge,mse_lasso,mse_elastic],

            "Root Mean Sqaure Error":[rmse_regression,rmse_random,rmse_ridge,rmse_lasso,rmse_elastic],

            "Mean Absoulate Error" :[mae_regression,mae_random,mae_ridge,mae_lasso,mae_elastic],

            "Variance Regression Score" :[vrs_regression,vrs_random,vrs_ridge,vrs_lasso,vrs_elastic]

                   }

comparison = pd.DataFrame(comparison)

comparison
sort=comparison.sort_values('R_Square/Accuracy',ascending=False)

sort
hamburgensis.info()
hamburgensis.drop("Sr.No",axis=1, inplace = True)

hamburgensis = hamburgensis.replace(['-'],'NaN')

hamburgensis['Maq'] = hamburgensis['Maq'].astype(float)

hamburgensis = hamburgensis.fillna(0)

hamburgensis = hamburgensis.set_index('Assembly_Metrics').T

hamburgensis
X2 = hamburgensis.drop(['Order'],axis=1)

Y2 = hamburgensis[['Order']]

print(X2.shape)

print(Y2.shape)
X2 = MinMaxScaler().fit_transform(X2)

X2 = pd.DataFrame({'Number_of_Contigs': X2[:, 0], 'Length_of_Largest_Contig': X2[:, 1],'N50': X2[:, 2],'N75': X2[:, 3], 'N90': X2[:, 4], 'NG50': X2[:,5], 'NG75': X2[:, 6], 'Contigs_greater_and_equal_N50': X2[:, 7],'Contigs_greater_and_equal_200': X2[:, 8],'Mean': X2[:, 9], 'Median': X2[:, 10],'Sum_of_the_Contig_Lengths': X2[:, 11], 'Coverage':X2[:, 12]})

X2
data2 = {'Assembly_Metrics':['Velvet', 'VCAKE', 'SSAKE', 'QSRA','SHARCGS','IDBA','Mira','Mira2','Maq','MARAGAP']} 

data2 = pd.DataFrame(data2)

X2['Assembly_Metrics'] = data2

X2 = X2.set_index('Assembly_Metrics')

X2
hamburgensis = pd.concat([X2, Y2], axis=1)

hamburgensis
a2 = sns.pairplot(hamburgensis, x_vars=['Number_of_Contigs','Length_of_Largest_Contig','N50'], y_vars='Order', height=5, aspect=0.9)

b2 = sns.pairplot(hamburgensis, x_vars=['N75','Contigs_greater_and_equal_N50','Contigs_greater_and_equal_200'], y_vars='Order', height=5, aspect=0.9)

c2 = sns.pairplot(hamburgensis, x_vars=['Mean','Median','Sum_of_the_Contig_Lengths','Coverage'], y_vars='Order', height=5, aspect=0.9)
correlations2 = hamburgensis.corr()

correlations2
mask = np.triu(np.ones_like(correlations2, dtype=np.bool))

plt.figure(figsize=(11,8))

sns.heatmap(correlations2*100,mask=mask,annot=True, fmt='.0f' )
X_2 = hamburgensis.drop(['Contigs_greater_and_equal_200','Sum_of_the_Contig_Lengths','Coverage'],axis=1)



Y_2 = hamburgensis[['Order']]

print(X_2.shape)

print(Y_2.shape)
bestfeatures2 = SelectKBest(score_func=chi2, k=11)

fit2 = bestfeatures2.fit(X_2,Y_2)

dfscores2 = pd.DataFrame(fit2.scores_)

dfcolumns2 = pd.DataFrame(X_2.columns)

#concat two dataframes for better visualization 

featureScores2 = pd.concat([dfcolumns2,dfscores2],axis=1)

featureScores2.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores2.nlargest(8,'Score'))  #print 9 best features
feature_imp_ham = ExtraTreesClassifier()

feature_imp_ham.fit(X_2,Y_2)

print(feature_imp_ham.feature_importances_)
feature_importance_ham = pd.Series(feature_imp_ham.feature_importances_, index=X_2.columns)

plt.figure(figsize=(12,8))

feature_importance_ham.nlargest(13).plot(kind='barh')

plt.show()

X_Final = X_2.drop(['Order','Length_of_Largest_Contig'],axis=1)

predict_hamburgensis = X_2.drop(['Order','Length_of_Largest_Contig'],axis=1)

X_Final
X_Final.describe()
profile_hamurgensis = pdp.ProfileReport(X_Final)

profile_hamurgensis
model2 = LinearRegression()
model2 = model2.fit(X_Final, Y_2)
pred2 = model2.predict(X_Final)
r2_regression2 = model2.score(X_Final, Y_2)

print('R^2: {0}'.format(r2_regression2))
model2.coef_
model2.intercept_
test_r2_2=r2_score(Y_2,pred2)
train_r2_2=r2_score(Y_2,model2.predict(X_Final))
print('R2 score for testing:',test_r2_2 )

print('R2 score for training:',train_r2_2 )
r_2_2 = []

for i in range(1, (X_Final.shape[-1])+1):

    m2=LinearRegression()

    m2.fit(X_Final.values[:,:i],Y_2)

    prd2=m2.predict(X_Final.values[:,:i])

    r_2_2.append(r2_score(Y_2,prd2))
plt.figure(figsize=(15,5))

plt.plot(r_2_2);

plt.xlabel('Features')

plt.ylabel('R_2 Score')


helper.adjR2(X_Final,Y_2,test_r2_2)
from yellowbrick.regressor import ResidualsPlot



plt.figure(figsize=(15,6)) 

visualizer2 = ResidualsPlot(model2,hist=True)

visualizer2.fit(X_Final.values, Y_2.values)  

visualizer2.score(X_Final.values, Y_2.values)  

visualizer2.poof()    
mse2_regression = mean_squared_error(Y_2,pred2)

rmse2_regression = sqrt(mse2_regression)

mae2_regression = metrics.mean_absolute_error(Y_2,pred2)

vrs2_regression = metrics.explained_variance_score(Y_2,pred2)



print("MEAN SQUARE ERROR : ", mse2_regression)

print("ROOT MEAN SQUARE ERROR : ", rmse2_regression)

print("MEAN ABSOLUTE ERROR : ", mae2_regression)

print("VARIANCE REGRESSION SCORE : ",vrs2_regression)
pk.dump(model2, open("model2_linear.pkl","wb"))
model2_linear=pk.load(open("model2_linear.pkl",'rb'))

pred_linear2=model2_linear.predict(predict_hamburgensis)

pred_linear2 = pred_linear2.astype(int)

Order_linear2 = pd.DataFrame(pred_linear2,columns=['Predicted_Order'])

Order_linear2['Assemblers'] = data2

Order_linear2.sort_values(by='Predicted_Order')
Y_2=np.ravel(Y_2)
randomregressor2 = RandomForestRegressor(max_depth=3, random_state=0, max_features=7)

random_regressor2 = randomregressor2.fit(X_Final,Y_2)

random_model2=random_regressor2.predict(X_Final)
random_model2.astype(int)
mse2_random = mean_squared_error(Y_2,random_model2)

rmse2_random = sqrt(mse2_random)

mae2_random = metrics.mean_absolute_error(Y_2,random_model2)

vrs2_random = metrics.explained_variance_score(Y_2,random_model2)

print("MEAN SQUARE ERROR : ",mse2_random)

print("ROOT MEAN SQUARE ERROR : ",rmse2_random) 

print("MEAN ABSOLUTE ERROR : ", mae2_random)

print("VARIANCE REGRESSION SCORE : ",vrs2_random)
r2_forest2 = randomregressor2.score(X_Final, Y_2)

print('R^2: {0}'.format(r2_forest2))
pk.dump(randomregressor2, open("model2_random.pkl","wb"))
model2_random=pk.load(open("model2_random.pkl",'rb'))

pred2_random=model2_random.predict(predict_hamburgensis)

pred2_random=pred2_random.astype(int)

Order2_random = pd.DataFrame(pred2_random,columns=['Predicted_Order'])

Order2_random['Assemblers'] = data2

Order2_random.sort_values(by='Predicted_Order')
ridge2 = Ridge()

parameters2 = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

ridge2_model_params = GridSearchCV(ridge2, parameters2,scoring='r2',cv=7)

ridge2_model_params.fit(X_Final,Y_2)
best_ridge2_params = ridge2_model_params.best_params_



print("Best Ridge Params : ", best_ridge2_params)
ridge2_model=Ridge(alpha=0.000001).fit(X_Final, Y_2)
ridge2_coefficients= ridge2_model.coef_

pd.DataFrame(ridge2_coefficients).astype(float)
r2_ridge2 = ridge2_model.score(X_Final, Y_2)

print('R^2: {0}'.format(r2_ridge2))
ridge2_model.intercept_
ridge2_model_predict=ridge2_model.predict(X_Final)
r_square_ridge2 = ridge2_model.score(X_Final, Y_2)

mse2_ridge = mean_squared_error(Y_2,ridge2_model_predict)

rmse2_ridge = sqrt(mse2_ridge)

mae2_ridge = metrics.mean_absolute_error(Y_2,ridge2_model_predict)

vrs2_ridge = metrics.explained_variance_score(Y_2,ridge2_model_predict)



print("MEAN SQUARE ERROR : ",mse2_ridge)

print("ROOT MEAN SQUARE ERROR : ",rmse2_ridge) 

print("MEAN ABSOLUTE ERROR : ", mae2_ridge)

print("VARIANCE REGRESSION SCORE : ",vrs2_ridge)



print("Model R Square: ",r_square_ridge2)
pk.dump(ridge2_model, open("model2_ridge.pkl","wb"))
model2_ridge=pk.load(open("model2_ridge.pkl",'rb'))

pred2_ridge=model2_ridge.predict(predict_hamburgensis)

pred2_ridge = pred2_ridge.astype(int)

Order2_ridge = pd.DataFrame(pred2_ridge,columns=['Predicted_Order'])

Order2_ridge['Assemblers'] = data2

Order2_ridge.sort_values(by='Predicted_Order')
lasso2 = Lasso()

parameters_2 = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

lasso2_model = GridSearchCV(lasso2, parameters_2,scoring='r2',cv=7)

lasso2_model = lasso2_model.fit(X_Final,Y_2)
best_lasso2_params = lasso2_model.best_params_

print("Best lasso Params : ", best_lasso2_params)
lasso2 = Lasso()

lasso2_model=Lasso(alpha=0.000001).fit(X_Final, Y_2)
lasso2_model.coef_

lasso2_coefficients= lasso2_model.coef_

pd.DataFrame(lasso2_coefficients).astype(float)
lasso2_model.intercept_
lasso2_model_predict=lasso2_model.predict(X_Final)
r_square_lasso2 = lasso2_model.score(X_Final, Y_2)



mse2_lasso = mean_squared_error(Y_2,lasso2_model_predict)



rmse2_lasso = sqrt(mse2_lasso)



mae2_lasso = metrics.mean_absolute_error(Y_2,lasso2_model_predict)



vrs2_lasso = metrics.explained_variance_score(Y_2,lasso2_model_predict)



print('MEAN SQUARE ERROR :{0:.4f} '.format(mse2_lasso))



print('ROOT MEAN SQUARE ERROR :{0:.4f} '.format(rmse2_lasso)) 



print('MEAN ABSOLUTE ERROR :{0:.4f}'.format(mae2_lasso))



print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs2_lasso))



print('Model R Square : {0:.4f} '.format(r_square_lasso2))

pk.dump(lasso2_model, open("model2_lasso.pkl","wb"))
model2_lasso=pk.load(open("model2_lasso.pkl",'rb'))

pred2_lasso=model2_lasso.predict(predict_hamburgensis)

pred2_lasso = pred2_lasso.astype(int)

Order2_lasso = pd.DataFrame(pred2_lasso,columns=['Predicted_Order'])

Order2_lasso['Assemblers'] = data2

Order2_lasso.sort_values(by='Predicted_Order')
alphas = [0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]

for a in alphas:

    model2 = ElasticNet(alpha=a).fit(X_Final,Y_2)   

    score2 = model2.score(X_Final, Y_2)

    pred_y2 = model2.predict(X_Final)

    mse2_elastic = mean_squared_error(Y_2, pred_y2)   

    print("Alpha:{0:.6f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(a, score2, mse2_elastic, np.sqrt(mse2_elastic)))
elastic2_model=ElasticNet(alpha=0.00001).fit(X_Final, Y_2)

elastic2_model_predict = elastic2_model.predict(X_Final)

score2 = elastic2_model.score(X_Final,Y_2)

mse2 = mean_squared_error(Y_2, elastic2_model_predict)

print("R2:{0:.2f}, MSE:{1:.2f}, RMSE:{2:.2f}"

      .format(score2, mse2, np.sqrt(mse2)))
elastic2_model.coef_
r_square_elastic2 = elastic2_model.score(X_Final, Y_2)



mse2_elastic = mean_squared_error(Y_2,elastic2_model_predict)



rmse2_elastic = sqrt(mse2_elastic)



mae2_elastic = metrics.mean_absolute_error(Y_2,elastic2_model_predict)



vrs2_elastic = metrics.explained_variance_score(Y_2,elastic2_model_predict)



print("MEAN SQUARE ERROR : ",mse2_elastic)



print("ROOT MEAN SQUARE ERROR : ",rmse2_elastic) 

print("MEAN ABSOLUTE ERROR : ", mae2_elastic)

print("VARIANCE REGRESSION SCORE : ",vrs2_elastic)

print("Model R Square: ",r_square_elastic2)

pk.dump(elastic2_model, open("model2_elastic.pkl","wb"))
model2_elastic=pk.load(open("model2_elastic.pkl",'rb'))

pred2_elastic=model2_elastic.predict(predict_hamburgensis)

pred2_elastic = pred2_elastic.astype(int)

Order2_elastic = pd.DataFrame(pred2_elastic,columns=['Predicted_Order'])

Order2_elastic['Assemblers'] = data2

Order2_elastic.sort_values(by='Predicted_Order')
comparison2={"Models":["Simple Regression", "Random Forest","Ridge Regression","Lasso Regression","ElasticNET Regression" ],

            "R_Square/Accuracy": [r2_regression2, r2_forest2, r_square_ridge2,r_square_lasso2,r_square_elastic2],

            "Mean Square Error/Accuracy": [mse2_regression, mse2_random, mse2_ridge,mse2_lasso,mse2_elastic],

            "Root Mean Sqaure Error":[rmse2_regression,rmse2_random,rmse2_ridge,rmse2_lasso,rmse2_elastic],

            "Mean Absoulate Error" :[mae2_regression,mae2_random,mae2_ridge,mae2_lasso,mae2_elastic],

            "Variance Regression Score" :[vrs2_regression,vrs2_random,vrs2_ridge,vrs2_lasso,vrs2_elastic]

                   }

comparison2 = pd.DataFrame(comparison2)

comparison2
sort2=comparison2.sort_values('R_Square/Accuracy',ascending=False)

sort2
vibrio.info()
vibrio.drop("Sr.No",axis=1, inplace = True)
vibrio.isnull().any().sum()
vibrio = vibrio.set_index('Assembly_Metrics').T
X3 = vibrio.iloc[:,0:13]

Y3 = vibrio[['Order']]

print(X3.shape)

print(Y3.shape)
X3 = MinMaxScaler().fit_transform(X3)

X3= pd.DataFrame({'Number_of_Contigs': X3[:, 0], 'Length_of_Largest_Contig': X3[:, 1],'N50': X3[:, 2],'N75': X3[:, 3], 'N90': X3[:, 4], 'NG50': X3[:,5], 'NG75': X3[:, 6], 'Contigs_greater_and_equal_N50': X3[:, 7],'Contigs_greater_and_equal_200': X3[:, 8],'Mean': X3[:, 9], 'Median': X3[:, 10],'Sum_of_the_Contig_Lengths': X3[:, 11], 'Coverage':X3[:, 12]})

X3
data3 = {'Assembly_Metrics':['Velvet', 'VCAKE', 'SSAKE', 'QSRA','SHARCGS','IDBA','Mira','Mira2','Maq','MARAGAP']} 

data3 = pd.DataFrame(data3)

X3['Assembly_Metrics'] = data3

X3 = X3.set_index('Assembly_Metrics')

X3
vibrio = pd.concat([X3, Y3], axis=1)
a3 = sns.pairplot(vibrio, x_vars=['Number_of_Contigs','Length_of_Largest_Contig','N50'], y_vars='Order', height=5, aspect=0.9)

b3 = sns.pairplot(vibrio, x_vars=['N75','Contigs_greater_and_equal_N50','Contigs_greater_and_equal_200'], y_vars='Order', height=5, aspect=0.9)

c3 = sns.pairplot(vibrio, x_vars=['Mean','Median','Sum_of_the_Contig_Lengths','Coverage'], y_vars='Order', height=5, aspect=0.9)
correlations3 = vibrio.corr()

correlations3
mask = np.triu(np.ones_like(correlations3, dtype=np.bool))

plt.figure(figsize=(11,8))

sns.heatmap(correlations3*100,mask=mask,annot=True, fmt='.0f' )
X_3 = vibrio.drop(['Contigs_greater_and_equal_200','Sum_of_the_Contig_Lengths'],axis=1)

Y_3 = vibrio[['Order']]

print(X_3.shape)

print(Y_3.shape)
bestfeatures3 = SelectKBest(score_func=chi2, k=12)

fit3 = bestfeatures3.fit(X_3,Y_3)

dfscores3 = pd.DataFrame(fit3.scores_)

dfcolumns3 = pd.DataFrame(X_3.columns)

#concat two dataframes for better visualization 

featureScores3 = pd.concat([dfcolumns3,dfscores3],axis=1)

featureScores3.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores3.nlargest(12,'Score'))
feature_imp_vib = ExtraTreesClassifier()

feature_imp_vib.fit(X_3,Y_3)

print(feature_imp_vib.feature_importances_)
feature_importance_vibrio = pd.Series(feature_imp_vib.feature_importances_, index=X_3.columns)

plt.figure(figsize=(12,8))

feature_importance_vibrio.nlargest(13).plot(kind='barh')

plt.show()

X_Final3 = X_3.drop(['Order','Number_of_Contigs','Contigs_greater_and_equal_N50'],axis=1)

predict_vibrio = X_3.drop(['Order','Number_of_Contigs','Contigs_greater_and_equal_N50'],axis=1)

predict_vibrio.info()
X_Final3.describe()
profile_vibrio = pdp.ProfileReport(X_Final3)

profile_vibrio
model3 = LinearRegression()
model3=model3.fit(X_Final3, Y3)
pred3 = model3.predict(X_Final3)
model3.coef_
model3.intercept_
r2_regression3 = model3.score(X_Final3, Y_3)

print('R^2: {0}'.format(r2_regression3))
test_r2_3=r2_score(Y3,pred3)

train_r2_3=r2_score(Y3,model3.predict(X_Final3))
print('R2 score for testing:',test_r2_3)

print('R2 score for training:',train_r2_3)
r_2_3 = []

for i in range(1, (X3.shape[-1])+1):

    m3=LinearRegression()

    m3.fit(X3.values[:,:i],Y3)

    prd3=m3.predict(X3.values[:,:i])

    r_2_3.append(r2_score(Y3,prd3))
plt.figure(figsize=(15,5))

plt.plot(r_2_3);

plt.xlabel('Features')

plt.ylabel('R_2 Score')
r_square_regression3 = model3.score(X_Final3, Y_3)

mse3_regression = mean_squared_error(Y_3,pred3)

rmse3_regression = sqrt(mse3_regression)

mae3_regression = metrics.mean_absolute_error(Y_3,pred3)

vrs3_regression = metrics.explained_variance_score(Y_3,pred3)



print("MEAN SQUARE ERROR : ", mse3_regression)

print("ROOT MEAN SQUARE ERROR : ", rmse3_regression)

print("MEAN ABSOLUTE ERROR : ", mae3_regression)

print("VARIANCE REGRESSION SCORE : ",vrs3_regression)

print("R^2 :",r_square_regression3)
pk.dump(model3, open("model3_linear.pkl","wb"))
model3_linear=pk.load(open("model3_linear.pkl",'rb'))

pred_linear3=model3_linear.predict(predict_vibrio)

pred_linear3 = pred_linear3.astype(int)

Order_linear3 = pd.DataFrame(pred_linear3,columns=['Predicted_Order'])

Order_linear3['Assemblers'] = data3

Order_linear3.sort_values(by='Predicted_Order')
Y_3=np.ravel(Y_3)
randomregressor3 = RandomForestRegressor(max_depth=5, random_state=0, max_features=9)

random_regressor3 = randomregressor3.fit(X_Final3,Y_3)

random_model3=random_regressor3.predict(X_Final3)
random_model3.astype(int)
mse3_random = mean_squared_error(Y_3,random_model3)

rmse3_random = sqrt(mse3_random)

mae3_random = metrics.mean_absolute_error(Y_3,random_model3)

vrs3_random = metrics.explained_variance_score(Y_3,random_model3)

print('MEAN SQUARE ERROR : {0:.4f} '.format(mse3_random))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse3_random)) 

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae3_random))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs3_random))
r2_forest3 = randomregressor3.score(X_Final3, Y_3)

print('R^2: {0:4f}'.format(r2_forest3))
pk.dump(randomregressor3, open("model3_random.pkl","wb"))
model3_random=pk.load(open("model3_random.pkl",'rb'))

pred3_random=model3_random.predict(predict_vibrio)

pred3_random=pred3_random.astype(int)

Order3_random = pd.DataFrame(pred3_random,columns=['Predicted_Order'])

Order3_random['Assemblers'] = data3

Order3_random.sort_values(by='Predicted_Order')
ridge3 = Ridge()

parameters3 = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

ridge3_model_params = GridSearchCV(ridge3, parameters3,scoring='r2',cv=7)

ridge3_model_params.fit(X_Final3,Y_3)
best_ridge3_params = ridge3_model_params.best_params_

print("Best Ridge Params : ", best_ridge3_params)
ridge3_model=Ridge(alpha=0.000001).fit(X_Final3, Y_3)
ridge3_coefficients= ridge3_model.coef_

pd.DataFrame(ridge3_coefficients).astype(float)
ridge3_model.intercept_
ridge3_model_predict=ridge3_model.predict(X_Final3)
r_square_ridge3 = ridge3_model.score(X_Final3, Y_3)

mse3_ridge = mean_squared_error(Y_3,ridge3_model_predict)

rmse3_ridge = sqrt(mse3_ridge)

mae3_ridge = metrics.mean_absolute_error(Y_3,ridge3_model_predict)

vrs3_ridge = metrics.explained_variance_score(Y_3,ridge3_model_predict)



print('MEAN SQUARE ERROR : {0:.4f}'.format(mse3_ridge))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse3_ridge))

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae3_ridge))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs3_ridge))

print('Model R Square: {0:.4f}'.format(r_square_ridge3))
pk.dump(ridge3_model, open("model3_ridge.pkl","wb"))
model3_ridge=pk.load(open("model3_ridge.pkl",'rb'))

pred3_ridge=model3_ridge.predict(predict_vibrio)

pred3_ridge = pred3_ridge.astype(int)

Order3_ridge = pd.DataFrame(pred3_ridge,columns=['Predicted_Order'])

Order3_ridge['Assemblers'] = data3

Order3_ridge.sort_values(by='Predicted_Order')
lasso3 = Lasso()

parameters_3 = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

lasso3_model = GridSearchCV(lasso3, parameters_3,scoring='r2',cv=7)

lasso3_model = lasso3_model.fit(X_Final3,Y_3)
best_lasso3_params = lasso3_model.best_params_

print("Best lasso Params : ", best_lasso3_params)
lasso3 = Lasso()

lasso3_model=Lasso(alpha=0.000001).fit(X_Final3, Y_3)
lasso3_model.coef_

lasso3_coefficients= lasso3_model.coef_

pd.DataFrame(lasso3_coefficients).astype(float)
lasso3_model.intercept_
lasso3_model_predict=lasso3_model.predict(X_Final3)
r_square_lasso3 = lasso3_model.score(X_Final3, Y_3)

mse3_lasso = mean_squared_error(Y_3,lasso3_model_predict)

rmse3_lasso = sqrt(mse3_lasso)

mae3_lasso = metrics.mean_absolute_error(Y_3,lasso3_model_predict)

vrs3_lasso = metrics.explained_variance_score(Y_3,lasso3_model_predict)



print('MEAN SQUARE ERROR : {0:.4f}'.format(mse3_lasso))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse3_lasso))

print('MEAN ABSOLUTE ERROR : {0:.4f}'.format(mae3_lasso))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs3_lasso))

print('Model R Square: {0:.4f}'.format(r_square_lasso3))

pk.dump(lasso3_model, open("model3_lasso.pkl","wb"))
model3_lasso=pk.load(open("model3_lasso.pkl",'rb'))

pred3_lasso=model3_lasso.predict(predict_vibrio)

pred3_lasso = pred3_lasso.astype(int)

Order3_lasso = pd.DataFrame(pred3_lasso,columns=['Predicted_Order'])

Order3_lasso['Assemblers'] = data3

Order3_lasso.sort_values(by='Predicted_Order')
alphas = [0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]

for a in alphas:

    model3 = ElasticNet(alpha=a).fit(X_Final3,Y_3)   

    score3 = model3.score(X_Final3, Y_3)

    pred_y3 = model3.predict(X_Final3)

    mse3_elastic = mean_squared_error(Y_3, pred_y3)   

    print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.3f}, RMSE:{3:.3f}".format(a, score3, mse3_elastic, np.sqrt(mse3_elastic)))
elastic3_model=ElasticNet(alpha=0.00001).fit(X_Final3, Y_3)

elastic3_model_predict = elastic3_model.predict(X_Final3)

score3 = elastic3_model.score(X_Final3,Y_3)

mse3 = mean_squared_error(Y_3, elastic3_model_predict)

print("R2:{0:.2f}, MSE:{1:.2f}, RMSE:{2:.2f}"

      .format(score3, mse3, np.sqrt(mse3)))
elastic3_model.coef_
r_square_elastic3 = elastic3_model.score(X_Final3, Y_3)



mse3_elastic = mean_squared_error(Y_3,elastic3_model_predict)



rmse3_elastic = sqrt(mse3_elastic)



mae3_elastic = metrics.mean_absolute_error(Y_3,elastic3_model_predict)



vrs3_elastic = metrics.explained_variance_score(Y_3,elastic3_model_predict)



print('MEAN SQUARE ERROR : {0:.4f}'.format(mse3_elastic))

print('ROOT MEAN SQUARE ERROR : {0:.4f}'.format(rmse3_elastic))

print('MEAN ABSOLUTE ERROR : {0:.4f}' .format(mae3_elastic))

print('VARIANCE REGRESSION SCORE : {0:.4f}'.format(vrs3_elastic))

print('Model R Square: {0:.4f}'.format(r_square_elastic3))

pk.dump(elastic3_model, open("model3_elastic.pkl","wb"))
model3_elastic=pk.load(open("model3_elastic.pkl",'rb'))

pred3_elastic=model3_elastic.predict(predict_vibrio)

pred3_elastic = pred3_elastic.astype(int)

Order3_elastic = pd.DataFrame(pred3_elastic,columns=['Predicted_Order'])

Order3_elastic['Assemblers'] = data3

Order3_elastic.sort_values(by='Predicted_Order')
comparison3={"Models":["Simple Regression", "Random Forest","Ridge Regression","Lasso Regression","ElasticNET Regression" ],

            "R_Square/Accuracy": [r2_regression3, r2_forest3, r_square_ridge3,r_square_lasso3,r_square_elastic3],

            "Mean Square Error/Accuracy": [mse3_regression, mse3_random, mse3_ridge,mse3_lasso,mse3_elastic],

            "Root Mean Sqaure Error":[rmse3_regression,rmse3_random,rmse3_ridge,rmse3_lasso,rmse3_elastic],

            "Mean Absoulate Error" :[mae3_regression,mae3_random,mae3_ridge,mae3_lasso,mae3_elastic],

            "Variance Regression Score" :[vrs3_regression,vrs3_random,vrs3_ridge,vrs3_lasso,vrs3_elastic]

                   }

comparison3 = pd.DataFrame(comparison3)

comparison3
sort=comparison3.sort_values('R_Square/Accuracy',ascending=False)

sort
pab1.info()
pab1.drop("Sr.No",axis=1, inplace = True)
pab1 = pab1.set_index('Assembly_Metrics').T.astype(float)
pab1['Order'] = pab1['Order'].astype(int)

X4 = pab1.iloc[:,0:13]

Y4 = pab1[['Order']]

print(X4.shape)

print(Y4.shape)
X4 = MinMaxScaler().fit_transform(X4)

X4 = pd.DataFrame({'Number_of_Contigs': X4[:, 0], 'Length_of_Largest_Contig': X4[:, 1],'N50': X4[:, 2],'N75': X4[:, 3], 'N90': X4[:, 4], 'NG50': X4[:,5], 'NG75': X4[:, 6], 'Contigs_greater_and_equal_N50': X4[:, 7],'Contigs_greater_and_equal_200': X4[:, 8],'Mean': X4[:, 9], 'Median': X4[:, 10],'Sum_of_the_Contig_Lengths': X4[:, 11], 'Coverage':X4[:, 12]})

X4
data4 = {'Assembly_Metrics':['VCAKE', 'QSRA','IDBA','MARAGAP']} 

data4 = pd.DataFrame(data4)

X4['Assembly_Metrics'] = data4

X4 = X4.set_index('Assembly_Metrics')

X4
pab1 = pd.concat([X4, Y4], axis=1)

pab1
a4 = sns.pairplot(pab1, x_vars=['Number_of_Contigs','Length_of_Largest_Contig','N50'], y_vars='Order', height=5, aspect=0.9)

b4 = sns.pairplot(pab1, x_vars=['N75','Contigs_greater_and_equal_N50','Contigs_greater_and_equal_200'], y_vars='Order', height=5, aspect=0.9)

c4 = sns.pairplot(pab1, x_vars=['Mean','Median','Sum_of_the_Contig_Lengths','Coverage'], y_vars='Order', height=5, aspect=0.9)
correlations4 = pab1.corr()

correlations4
mask = np.triu(np.ones_like(correlations4, dtype=np.bool))



plt.figure(figsize=(11,8))

sns.heatmap(correlations4*100,mask=mask,annot=True, fmt='.0f' )
X_4 = pab1.iloc[:,0:13]

Y_4 = pab1[['Order']]

print(X_4.shape)

print(Y_4.shape)
bestfeatures4 = SelectKBest(score_func=chi2, k=12)

fit4 = bestfeatures4.fit(X_4,Y_4)

dfscores4 = pd.DataFrame(fit4.scores_)

dfcolumns4 = pd.DataFrame(X_4.columns)

#concat two dataframes for better visualization 

featureScores4 = pd.concat([dfcolumns4,dfscores4],axis=1)

featureScores4.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores4.nlargest(12,'Score'))  #print 10 best features
feature_imp_pab = ExtraTreesClassifier()

feature_imp_pab.fit(X_4,Y_4)

print(feature_imp_pab.feature_importances_)
feature_importance_pab1 = pd.Series(feature_imp_pab.feature_importances_, index=X_4.columns)

plt.figure(figsize=(12,8))

feature_importance_pab1.nlargest(13).plot(kind='barh')

plt.show()

X_Final4 = X_4.drop(['Number_of_Contigs','Contigs_greater_and_equal_N50','Coverage'],axis=1)

predict_pab1 = X_4.drop(['Number_of_Contigs','Contigs_greater_and_equal_N50','Coverage'],axis=1)

X_Final4

X_Final4.describe()
profile_pab1 = pdp.ProfileReport(X_Final4)

profile_pab1
model4 = LinearRegression()
model4.fit(X_Final4, Y_4)
pred4 = model4.predict(X_Final4)
coefficients4 = model4.coef_

co4 = pd.DataFrame(coefficients4).astype(float)

co4
model4.intercept_
pred4 = model4.predict(X_Final4)

pred4.astype(int)
test_r2_4=r2_score(Y_4,pred4)
train_r2_4=r2_score(Y_4,model4.predict(X_Final4))
print('R2 score for testing:',test_r2_4 )

print('R2 score for training:',train_r2_4 )
r_2_4 = []

for i in range(1, (X_Final4.shape[-1])+1):

    m4=LinearRegression()

    m4.fit(X_Final4.values[:,:i],Y_4)

    prd4=m4.predict(X_Final4.values[:,:i])

    r_2_4.append(r2_score(Y_4,prd4))
plt.figure(figsize=(15,5))

plt.plot(r_2_4);

plt.xlabel('Features')

plt.ylabel('R_2 Score')
helper.adjR2(X_Final4,Y_4,test_r2_4)
r2_regression4 = model4.score(X_Final4, Y_4)

print('R^2: {0}'.format(r2_regression4))
plt.figure(figsize=(15,6)) 

visualizer4 = ResidualsPlot(model4,hist=True)

visualizer4.fit(X_Final4.values, Y_4.values)  

visualizer4.score(X_Final4.values, Y_4.values)  

visualizer4.poof()    
mse4_regression = mean_squared_error(Y_4,pred4)

rmse4_regression = sqrt(mse4_regression)

mae4_regression = metrics.mean_absolute_error(Y_4,pred4)

vrs4_regression = metrics.explained_variance_score(Y_4,pred4)



print("MEAN SQUARE ERROR : ", mse4_regression)



print("ROOT MEAN SQUARE ERROR : ", rmse4_regression)



print("MEAN ABSOLUTE ERROR : ", mae4_regression)



print("VARIANCE REGRESSION SCORE : ",vrs4_regression)
pk.dump(model4, open("model4_linear.pkl","wb"))
model4_linear=pk.load(open("model4_linear.pkl",'rb'))

predict4_linear=model4_linear.predict(predict_pab1)

pred4_linear=predict4_linear.astype(int).flatten()

Order4_reg = pd.DataFrame(pred4_linear,columns=['Predicted_Order'])

Order4_reg['Assemblers'] = data4

Order4_reg.sort_values(by='Predicted_Order')
Y_4=np.ravel(Y_4)
randomregressor4 = RandomForestRegressor(max_depth=5, random_state=0, max_features=7)

random_regressor4 = randomregressor4.fit(X_Final4,Y_4)

random_model4 = random_regressor4.predict(X_Final4)
random_model4.astype(int)
r2_forest4 = randomregressor4.score(X_Final4, Y_4)

mse4_random = mean_squared_error(Y_4,random_model4)

rmse4_random = sqrt(mse4_random)

mae4_random = metrics.mean_absolute_error(Y_4,random_model4)

vrs4_random = metrics.explained_variance_score(Y_4,random_model4)

print("MEAN SQUARE ERROR : ",mse4_random)

print("ROOT MEAN SQUARE ERROR : ",rmse4_random) 

print("MEAN ABSOLUTE ERROR : ", mae4_random)

print("VARIANCE REGRESSION SCORE: ",vrs4_random)

print('R^2: {0:.2f}'.format(r2_forest4))
pk.dump(randomregressor4, open("model4_random.pkl","wb"))
model4_random=pk.load(open("model4_random.pkl",'rb'))

pred4_random=model4_random.predict(predict_pab1)

pred4_random=pred4_random.astype(int)

Order4_random = pd.DataFrame(pred4_random,columns=['Predicted_Order'])

Order4_random['Assemblers'] = data4

Order4_random.sort_values(by='Predicted_Order')
ridge4 = Ridge()

parameters4 = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

ridge4_model_params = GridSearchCV(ridge4, parameters4,scoring='r2',cv=4)

ridge4_model_params.fit(X_Final4,Y_4)
best_ridge4_params = ridge4_model_params.best_params_



print("Best Ridge Params : ", best_ridge4_params)
ridge4_model=Ridge(alpha=0.000001).fit(X_Final4, Y_4)
ridge4_coefficients= ridge4_model.coef_

pd.DataFrame(ridge4_coefficients).astype(float)
ridge4_model.intercept_
ridge4_model_predict=ridge4_model.predict(X_Final4)
r_square_ridge4 = ridge4_model.score(X_Final4, Y_4)

mse4_ridge = mean_squared_error(Y_4,ridge4_model_predict)

rmse4_ridge = sqrt(mse4_ridge)

mae4_ridge = metrics.mean_absolute_error(Y_4,ridge4_model_predict)

vrs4_ridge = metrics.explained_variance_score(Y_4,ridge4_model_predict)



print('MEAN SQUARE ERROR : {0:.3f}'.format(mse4_ridge))

print("ROOT MEAN SQUARE ERROR : ",rmse4_ridge) 

print("MEAN ABSOLUTE ERROR : ", mae4_ridge)

print('VARIANCE REGRESSION SCORE :{0:.3f}' .format(vrs4_ridge))

print('Model R Square: {0:.3f}'.format(r_square_ridge4))
pk.dump(ridge4_model, open("model4_ridge.pkl","wb"))
model4_ridge=pk.load(open("model4_ridge.pkl",'rb'))

pred4_ridge=model4_ridge.predict(predict_pab1)

pred4_ridge = pred4_ridge.astype(int)

Order4_ridge = pd.DataFrame(pred4_ridge,columns=['Predicted_Order'])

Order4_ridge['Assemblers'] = data4

Order4_ridge.sort_values(by='Predicted_Order')
lasso4 = Lasso()

parameters_4 = {'alpha':[0.000001,0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]}

lasso4_model = GridSearchCV(lasso4, parameters_4,scoring='r2',cv=4)

lasso4_model = lasso4_model.fit(X_Final4,Y_4)
best_lasso4_params = lasso4_model.best_params_

print("Best lasso Params : ", best_lasso4_params)
lasso4 = Lasso()

lasso4_model=Lasso(alpha=0.000001).fit(X_Final4, Y_4)
lasso4_model.coef_

lasso4_coefficients= lasso4_model.coef_

pd.DataFrame(lasso4_coefficients).astype(float)
lasso4_model.intercept_
lasso4_model_predict=lasso4_model.predict(X_Final4)
r_square_lasso4 = lasso4_model.score(X_Final4, Y_4)

mse4_lasso = mean_squared_error(Y_4,lasso4_model_predict)

rmse4_lasso = sqrt(mse4_lasso)

mae4_lasso = metrics.mean_absolute_error(Y_4,lasso4_model_predict)

vrs4_lasso = metrics.explained_variance_score(Y_4,lasso4_model_predict)

print("MEAN SQUARE ERROR : ",mse4_lasso)

print("ROOT MEAN SQUARE ERROR : ",rmse4_lasso) 

print("MEAN ABSOLUTE ERROR : ", mae4_lasso)

print("VARIANCE REGRESSION SCORE : ",vrs4_lasso)

print("Model R Square: ",r_square_lasso4)
pk.dump(lasso4_model, open("model4_lasso.pkl","wb"))
model4_lasso=pk.load(open("model4_lasso.pkl",'rb'))

pred4_lasso=model4_lasso.predict(predict_pab1)

pred4_lasso = pred4_lasso.astype(int)

Order4_lasso = pd.DataFrame(pred4_lasso,columns=['Predicted_Order'])

Order4_lasso['Assemblers'] = data4

Order4_lasso.sort_values(by='Predicted_Order')
alphas = [0.00001, 0.001, 0.01, 0.1, 0.3, 0.5]

for a in alphas:

    model4 = ElasticNet(alpha=a).fit(X_Final4,Y_4)   

    score4 = model4.score(X_Final4, Y_4)

    pred_y4 = model4.predict(X_Final4)

    mse4_elastic = mean_squared_error(Y_4, pred_y4)   

    print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.3f}, RMSE:{3:.3f}".format(a, score4, mse4_elastic, np.sqrt(mse4_elastic)))
elastic4_model=ElasticNet(alpha=0.001).fit(X_Final4, Y_4)

elastic4_model_predict = elastic4_model.predict(X_Final4)

score4 = elastic4_model.score(X_Final4,Y_4)

mse4 = mean_squared_error(Y_4, elastic4_model_predict)

print("R2:{0:.2f}, MSE:{1:.2f}, RMSE:{2:.2f}"

      .format(score4, mse4, np.sqrt(mse4)))
elastic4_model.coef_
r_square_elastic4 = elastic4_model.score(X_Final4, Y_4)



mse4_elastic = mean_squared_error(Y_4,elastic4_model_predict)



rmse4_elastic = sqrt(mse4_elastic)



mae4_elastic = metrics.mean_absolute_error(Y_4,elastic4_model_predict)



vrs4_elastic = metrics.explained_variance_score(Y_4,elastic4_model_predict)



print("MEAN SQUARE ERROR : ",mse4_elastic)



print("ROOT MEAN SQUARE ERROR : ",rmse4_elastic) 

print("MEAN ABSOLUTE ERROR : ", mae4_elastic)

print("VARIANCE REGRESSION SCORE : ",vrs4_elastic)

print("Model R Square: ",r_square_elastic4)

pk.dump(elastic4_model, open("model4_elastic.pkl","wb"))
model4_elastic=pk.load(open("model4_elastic.pkl",'rb'))

pred4_elastic=model4_elastic.predict(predict_pab1)

pred4_elastic = pred4_elastic.astype(int)

Order4_elastic = pd.DataFrame(pred4_elastic,columns=['Predicted_Order'])

Order4_elastic['Assemblers'] = data4

Order4_elastic.sort_values(by='Predicted_Order')
comparison4={"Models":["Simple Regression", "Random Forest","Ridge Regression","Lasso Regression","ElasticNET Regression" ],

            "R_Square/Accuracy": [r2_regression4, r2_forest4, r_square_ridge4,r_square_lasso4,r_square_elastic4],

            "Mean Square Error/Accuracy": [mse4_regression, mse4_random, mse4_ridge,mse4_lasso,mse4_elastic],

            "Root Mean Sqaure Error":[rmse4_regression,rmse4_random,rmse4_ridge,rmse4_lasso,rmse4_elastic],

            "Mean Absoulate Error" :[mae4_regression,mae4_random,mae4_ridge,mae4_lasso,mae4_elastic],

            "Variance Regression Score" :[vrs4_regression,vrs4_random,vrs4_ridge,vrs4_lasso,vrs4_elastic]

                   }

comparison4 = pd.DataFrame(comparison4)

comparison4
sort4=comparison4.sort_values('R_Square/Accuracy',ascending=False)

sort4