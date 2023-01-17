# Importing the libraries
import pandas as pd        # for data manipulation
import seaborn as sns      # for statistical data visualisation
import numpy as np         # for linear algebra
import matplotlib.pyplot as plt      # for data visualization
from scipy import stats        # for calculating statistics

# Importing various machine learning algorithm from sklearn

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error,roc_curve,auc,accuracy_score,mean_squared_error,r2_score
from scipy.stats import zscore
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import resample
from math import sqrt
dataframe= pd.read_csv("concrete (1).csv")  # Reading the data
dataframe.head()   # showing first 5 datas
dataframe.shape
dataframe.info()
dataframe.isnull().sum()
dataframe.apply(lambda x: len(x.unique()))
dataframe.describe()
dataframe.skew()
print('Range of values',dataframe.cement.max()-dataframe.cement.min())
print('Interquartile range: ',dataframe.cement.describe()['75%']-dataframe.cement.describe()['25%'])
q3=dataframe.cement.describe()['75%']
q1=dataframe.cement.describe()['25%']
iqr=dataframe.cement.describe()['75%']-dataframe.cement.describe()['25%']
print('Upper outliers starts from',q3+1.5*iqr)
print('Lower outliers starts from',q1-1.5*iqr)

print('Number of Upper outliers are : ',dataframe[dataframe['cement']>586.4375].cement.count(),'(',dataframe[dataframe['cement']>586.4375].cement.count()/len(dataframe),'%)')
print('Number of Lower outliers are : ',dataframe[dataframe['cement']<-44.0625].cement.count(),'(',dataframe[dataframe['cement']<-44.0625].cement.count()/len(dataframe),'%)')
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))

sns.distplot(dataframe.cement,ax=a1)
a1.set_title('Cement')
a1.set_ylabel('strengh')
sns.boxplot(x='cement',data=dataframe,ax=a2)

print('Range of values',dataframe.slag.max()-dataframe.slag.min())
print('Interquartile range: ',dataframe.slag.describe()['75%']-dataframe.slag.describe()['25%'])
q3=dataframe.slag.describe()['75%']
q1=dataframe.slag.describe()['25%']
iqr=dataframe.slag.describe()['75%']-dataframe.slag.describe()['25%']
print('Upper outliers starts from',q3+1.5*iqr)
print('Lower outliers starts from',q1-1.5*iqr)

print('Number of Upper outliers are : ',dataframe[dataframe['slag']>357.357].slag.count(),'(',dataframe[dataframe['slag']>357.375].slag.count()/len(dataframe),'%)')
print('Number of Lower outliers are : ',dataframe[dataframe['slag']<-214.42499999999998].slag.count(),'(',dataframe[dataframe['slag']<-214.42499999999998].slag.count()/len(dataframe),'%)')
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))

sns.distplot(dataframe.slag,ax=a1)
a1.set_title('Slag')
a1.set_ylabel('strengh')
sns.boxplot(x='slag',data=dataframe,ax=a2)
print('Range of values',dataframe.ash.max()-dataframe.ash.min())
print('Interquartile range: ',dataframe.ash.describe()['75%']-dataframe.ash.describe()['25%'])
q3=dataframe.ash.describe()['75%']
q1=dataframe.ash.describe()['25%']
iqr=dataframe.ash.describe()['75%']-dataframe.ash.describe()['25%']
print('Upper outliers starts from',q3+1.5*iqr)
print('Lower outliers starts from',q1-1.5*iqr)

print('Number of Upper outliers are : ',dataframe[dataframe['ash']>295.75].ash.count(),'(',dataframe[dataframe['ash']> 295.75].ash.count()/len(dataframe),'%)')
print('Number of Lower outliers are : ',dataframe[dataframe['ash']<-177.45].ash.count(),'(',dataframe[dataframe['ash']<-177.45].ash.count()/len(dataframe),'%)')
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))

sns.distplot(dataframe.ash,ax=a1)
a1.set_title('Ash ')
a1.set_ylabel('strengh')
sns.boxplot(x='ash',data=dataframe,ax=a2)
print('Range of values',dataframe.water.max()-dataframe.water.min())
print('Interquartile range: ',dataframe.water.describe()['75%']-dataframe.water.describe()['25%'])
q3=dataframe.water.describe()['75%']
q1=dataframe.water.describe()['25%']
iqr=dataframe.water.describe()['75%']-dataframe.water.describe()['25%']
print('Upper outliers starts from',q3+1.5*iqr)
print('Lower outliers starts from',q1-1.5*iqr)

print('Number of Upper outliers are : ',dataframe[dataframe['water']>232.64999999999998].water.count(),'(',dataframe[dataframe['water']> 232.64999999999998].water.count()/len(dataframe),'%)')
print('Number of Lower outliers are : ',dataframe[dataframe['water']<124.25000000000001].water.count(),'(',dataframe[dataframe['water']<124.25000000000001].water.count()/len(dataframe),'%)')
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))

sns.distplot(dataframe.water,ax=a1)
a1.set_title('Water')
a1.set_ylabel('strengh')
sns.boxplot(x='water',data=dataframe,ax=a2)
print('Range of values',dataframe.superplastic.max()-dataframe.superplastic.min())
print('Interquartile range: ',dataframe.superplastic.describe()['75%']-dataframe.superplastic.describe()['25%'])
q3=dataframe.superplastic.describe()['75%']
q1=dataframe.superplastic.describe()['25%']
iqr=dataframe.superplastic.describe()['75%']-dataframe.superplastic.describe()['25%']
print('Upper outliers starts from',q3+1.5*iqr)
print('Lower outliers starts from',q1-1.5*iqr)

print('Number of Upper outliers are : ',dataframe[dataframe['superplastic']>25.5].superplastic.count(),'(',dataframe[dataframe['superplastic']> 25.5].superplastic.count()/len(dataframe),'%)')
print('Number of Lower outliers are : ',dataframe[dataframe['superplastic']<-15.299999999999999].superplastic.count(),'(',dataframe[dataframe['superplastic']<-15.299999999999999].superplastic.count()/len(dataframe),'%)')
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))

sns.distplot(dataframe.superplastic,ax=a1)
a1.set_title('Super plastic')
a1.set_ylabel('strengh')
sns.boxplot(x='superplastic',data=dataframe,ax=a2)
fig, ax2 = plt.subplots(3, 3, figsize=(16, 16))
sns.distplot(dataframe['cement'],ax=ax2[0][0])
sns.distplot(dataframe['slag'],ax=ax2[0][1])
sns.distplot(dataframe['ash'],ax=ax2[0][2])
sns.distplot(dataframe['water'],ax=ax2[1][0])
sns.distplot(dataframe['superplastic'],ax=ax2[1][1])
sns.distplot(dataframe['coarseagg'],ax=ax2[1][2])
sns.distplot(dataframe['fineagg'],ax=ax2[2][0])
sns.distplot(dataframe['age'],ax=ax2[2][1])
sns.distplot(dataframe['strength'],ax=ax2[2][2])
sns.regplot(x='cement',y='strength',data=dataframe)
sns.regplot(x='slag',y='strength',data=dataframe)
sns.regplot(x='age',y='strength',data=dataframe)
sns.regplot(x='superplastic',y='strength',data=dataframe)
sns.pairplot(dataframe)
corelation=dataframe.corr()
corelation
plt.figure(figsize=(20,20))
a=sns.heatmap(corelation,annot=True)
dataframe.boxplot(figsize=(35,15))
for columns in dataframe.columns[:-1]:
    q1 = dataframe[columns].quantile(0.25)
    q3 = dataframe[columns].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    dataframe.loc[(dataframe[columns] < low) | (dataframe[columns] > high), columns] = dataframe[columns].median()

dataframe.boxplot(figsize=(35,15))

dataframe.columns
z_dataframe=dataframe.apply(zscore)
z_dataframe
features=['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
       'fineagg', 'age']
    
X=z_dataframe[features]
Y=z_dataframe['strength']         
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 
train_X.head()
test_X.count()
test_X.head()
linear_reg=LinearRegression(n_jobs=1)
linear_reg
l_model=linear_reg.fit(train_X,train_y)
predict=l_model.predict(test_X)
print(predict)

print('Performance on training data using Linear Model:',linear_reg.score(train_X,train_y))
print('Performance on testing data using Linear Model:',linear_reg.score(test_X,test_y))
acc_DT=r2_score(test_y, predict)
print('Accuracy DT: ',acc_DT)
print('MSE: ',mean_squared_error(test_y, predict))
results = pd.DataFrame({'Method':['Linear Regression'], 'accuracy': acc_DT},index={'1'})
results = results[['Method', 'accuracy']]
results
R_model=Ridge(alpha=.3)
R_model=R_model.fit(train_X , train_y)
R_predict=R_model.predict(test_X)
R_predict[0:100]
print('Performance on training data using Ridge Model:',R_model.score(train_X,train_y))
print('Performance on testing data using Ridge Model:',R_model.score(test_X,test_y))
acc_DT=r2_score(test_y, R_predict)
print('Accuracy DT: ',acc_DT)
print('MSE: ',mean_squared_error(test_y, R_predict))
L_results = pd.DataFrame({'Method':['Ridge Regression'], 'accuracy': acc_DT},index={'2'})
results=pd.concat([results,L_results])
results = results[['Method', 'accuracy']]
results
L_model=Lasso(alpha=.2)
L_model=L_model.fit(train_X , train_y)
L_predict=L_model.predict(test_X)
L_predict[0:100]
print('Performance on training data using Lasso Model:',L_model.score(train_X,train_y))
print('Performance on testing data using Lasso Model:',L_model.score(test_X,test_y))
acc_DT=r2_score(test_y, L_predict)
print('Accuracy DT: ',acc_DT)
print('MSE: ',mean_squared_error(test_y, L_predict))
L_results = pd.DataFrame({'Method':['Lasso Regression'], 'accuracy': acc_DT},index={'3'})
results=pd.concat([results,L_results])
results = results[['Method', 'accuracy']]
results
dtree = DecisionTreeRegressor(random_state=1)
dtree=dtree.fit(train_X , train_y)
dtree_predict=dtree.predict(test_X)
dtree_predict[0:100]
print('Performance on training data using Lasso Model:',dtree.score(train_X,train_y))
print('Performance on testing data using DT:',dtree.score(test_X,test_y))
acc_DT=r2_score(test_y, dtree_predict)
print('Accuracy : ',acc_DT)
print('Mean Sqaure Error: ',mean_squared_error(test_y, dtree_predict))
cr_results = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT},index={'4'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
cross_val_score(dtree,train_X,train_y,cv=10).mean()
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(dtree,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Decision Tree'], 'accuracy': kfold_acc},index={'5'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
imp=pd.DataFrame(data=dtree.feature_importances_,index=list(X.columns))
imp
df=z_dataframe.copy()
df
x=df.drop(['ash','coarseagg','fineagg','strength'] , axis=1)
y=df.strength
train_X,test_X,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=1)
train_X.count() 
dtree2 = DecisionTreeRegressor(random_state=1)
dtree2=dtree2.fit(train_X , train_y)
dtree_predict2=dtree2.predict(test_X)
dtree_predict2[0:100]
print('Performance on training data using DT:',dtree2.score(train_X,train_y))
print('Performance on testing data using DT:',dtree2.score(test_X,test_y))
acc_DT=r2_score(test_y, dtree_predict2)
print('Accuracy : ',acc_DT)
print('Mean Sqaure Error: ',mean_squared_error(test_y, dtree_predict2))
dt2_results = pd.DataFrame({'Method':['2nd Decision Tree'], 'accuracy': acc_DT},index={'6'})
results=pd.concat([results,dt2_results])
results = results[['Method', 'accuracy']]
results
features=['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
       'fineagg', 'age']
    
X=z_dataframe[features]
Y=z_dataframe['strength']         
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 
dtree_reg = DecisionTreeRegressor( max_depth = 4,random_state=1,min_samples_leaf=5)
dtree_reg.fit(train_X, train_y)

dtree_reg_pred = dtree_reg.predict(test_X)
print('Performance on training data using DT:',dtree_reg.score(train_X,train_y))
print('Performance on testing data using DT:',dtree_reg.score(test_X,test_y))
acc_RDT=r2_score(test_y, dtree_reg_pred)
print('Accuracy DT: ',acc_RDT)
print('MSE: ',mean_squared_error(test_y,dtree_reg_pred))
RegDtree_result = pd.DataFrame({'Method':['Regularised Decision Tree'], 'accuracy': [acc_RDT]},index={'7'})
results = pd.concat([results, RegDtree_result])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(dtree_reg,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Regularised Decision Tree'], 'accuracy': kfold_acc},index={'8'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
x=df.drop(['ash','coarseagg','fineagg','strength'] , axis=1)
y=df.strength
train_X,test_X,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=1)
train_X.count()
dtree_reg = DecisionTreeRegressor( max_depth = 4,random_state=1,min_samples_leaf=5)
dtree_reg.fit(train_X, train_y)
re_dtree_predict = dtree_reg.predict(test_X)
print('Performance on training data using DT:',dtree_reg.score(train_X,train_y))
print('Performance on testing data using DT:',dtree_reg.score(test_X,test_y))
acc_RDT=r2_score(test_y, re_dtree_predict)
print('Accuracy DT: ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, re_dtree_predict))
cr_results = pd.DataFrame({'Method':['2nd Pruned Decision Tree '], 'accuracy': acc_RDT},index={'9'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
features=['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
       'fineagg', 'age']
    
X=z_dataframe[features]
Y=z_dataframe['strength']         
train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 
rfr_model=RandomForestRegressor(random_state=1)
rfr_model=rfr_model.fit(train_X,train_y)
rfr_model
rfr_predict = rfr_model.predict(test_X)
print('Performance on training data using Lasso Model:',rfr_model.score(train_X,train_y))
print('Performance on testing data using RF:',rfr_model.score(test_X,test_y))
acc_RDT=r2_score(test_y, rfr_predict)
print('Accuracy RF: ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, rfr_predict))
cr_results = pd.DataFrame({'Method':['Random Forest Regressor '], 'accuracy': acc_RDT},index={'10'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(rfr_model,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Random Forest Regressor '], 'accuracy': kfold_acc},index={'11'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
ada_model=AdaBoostRegressor(random_state=1)
ada_model=ada_model.fit(train_X,train_y)
ada_model
ada_predict = ada_model.predict(test_X)
print('Performance on training data using Ada Boosting:',ada_model.score(train_X,train_y))
print('Performance on testing data using Ada Boosting:',ada_model.score(test_X,test_y))
acc_RDT=r2_score(test_y, ada_predict)
print('Accuracy Ada Boostig: ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, ada_predict))
cr_results = pd.DataFrame({'Method':['Ada Boosting'], 'accuracy': acc_RDT},index={'12'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(ada_model,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Ada Boosting '], 'accuracy': kfold_acc},index={'13'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
gb_model=GradientBoostingRegressor(random_state=1)
gb_model=gb_model.fit(train_X,train_y)
gb_model
gb_predict = gb_model.predict(test_X)
print('Performance on training data using Gradient Boosting:',gb_model.score(train_X,train_y))
print('Performance on testing data using Gradient Boosting:',gb_model.score(test_X,test_y))
acc_RDT=r2_score(test_y, gb_predict)
print('Accuracy Gradient Boostig: ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, gb_predict))
cr_results = pd.DataFrame({'Method':['Gradient Boosting'], 'accuracy': acc_RDT},index={'14'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(gb_model,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Gradient Boosting '], 'accuracy': kfold_acc},index={'15'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
svr_model=SVR(kernel="linear")
svr_model=svr_model.fit(train_X,train_y)
svr_model
svr_predict = svr_model.predict(test_X)
print('Performance on training data using SVR:',svr_model.score(train_X,train_y))
print('Performance on testing data using SVR:',svr_model.score(test_X,test_y))
acc_RDT=r2_score(test_y, svr_predict)
print('Accuracy SVR: ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, svr_predict))
cr_results = pd.DataFrame({'Method':['SVR '], 'accuracy': acc_RDT},index={'16'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(svr_model,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val SVR '], 'accuracy': kfold_acc},index={'17'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
bag_model=BaggingRegressor(random_state=1)
bag_model=bag_model.fit(train_X,train_y)
bag_model
bag_predict = bag_model.predict(test_X)
print('Performance on training data using Bagging Regressor:',bag_model.score(train_X,train_y))
print('Performance on testing data using Bagging Regressor :',bag_model.score(test_X,test_y))
acc_RDT=r2_score(test_y, bag_predict)
print('Accuracy Bagging Regressor : ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, bag_predict))
cr_results = pd.DataFrame({'Method':['Bagging Regressor '], 'accuracy': acc_RDT},index={'18'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(bag_model,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Bagging Regressor '], 'accuracy': kfold_acc},index={'19'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
error=[]
for i in range(1,30):
    knn_model = KNeighborsRegressor(n_neighbors=i)
    knn_model.fit(train_X,train_y)
    pred_i = knn_model.predict(test_X)
    error.append(np.mean(pred_i!=test_y))
  

plt.figure(figsize=(12,6))
plt.plot(range(1,30),error,color='red', linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean error')
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(train_X,train_y)
knn_predict = knn_model.predict(test_X)
print('Performance on training data using KNN:',knn_model.score(train_X,train_y))
print('Performance on testing data using KNN :',knn_model.score(test_X,test_y))
acc_RDT=r2_score(test_y, knn_predict)
print('Accuracy KNN : ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, knn_predict))
cr_results = pd.DataFrame({'Method':['KNN Regressor '], 'accuracy': acc_RDT},index={'20'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(knn_model,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val KNN Regressor '], 'accuracy': kfold_acc},index={'21'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results

L=LinearRegression()
K=KNeighborsRegressor(n_neighbors=3)
S=SVR(kernel='linear')
ebl=VotingRegressor(estimators=[('L',L),('K',K),('S',S)])
ebl.fit(train_X,train_y)
ebl_predict = ebl.predict(test_X)
print('Performance on training data using Ensemble technique:',ebl.score(train_X,train_y))
print('Performance on testing data using Ensemble technique :',ebl.score(test_X,test_y))
acc_RDT=r2_score(test_y, ebl_predict)
print('Accuracy Ensemble : ',acc_RDT)
print('MSE: ',mean_squared_error(test_y, ebl_predict))
cr_results = pd.DataFrame({'Method':['Ensemble '], 'accuracy': acc_RDT},index={'22'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
kfold= KFold(n_splits=10,shuffle=True,random_state=1)
kfold_acc=cross_val_score(ebl,train_X,train_y,cv=kfold).mean()
print("Cross Val : ",kfold_acc)
cr_results = pd.DataFrame({'Method':['Cross Val Ensemble '], 'accuracy': kfold_acc},index={'23'})
results=pd.concat([results,cr_results])
results = results[['Method', 'accuracy']]
results
values=z_dataframe.values
values
scores=list()   
for i in range(1000):
    # prepare train and test sets
    train = resample(values, n_samples=len(z_dataframe))  # Sampling with replacement 
    test = np.array([x for x in values if x.tolist() not in train.tolist()])  # picking rest of the data not considered in sample
    
    
     # fit model
    gbm_model = GradientBoostingRegressor(n_estimators=50)
    # fit against independent variables and corresponding target values
    gbm_model.fit(train[:,:-1], train[:,-1]) 
    # Take the target column for all rows in test set

    y_test = test[:,-1]    
    # evaluate model
    # predict based on independent variables in the test data
    score = gbm_model.score(test[:, :-1] , y_test)
    predictions = gbm_model.predict(test[:, :-1])  

    scores.append(score)
scores
plt.hist(scores)
plt.show()
alpha = 0.95                             # for 95% confidence 
p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)
lower = max(0.0, np.percentile(scores, p))  
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(scores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
scores=list()   
for i in range(1000):
    # prepare train and test sets
    train = resample(values, n_samples=len(z_dataframe))  # Sampling with replacement 
    test = np.array([x for x in values if x.tolist() not in train.tolist()])  # picking rest of the data not considered in sample
    
    
     # fit model
    gbm_model = RandomForestRegressor(n_estimators=50)
    # fit against independent variables and corresponding target values
    gbm_model.fit(train[:,:-1], train[:,-1]) 
    # Take the target column for all rows in test set

    y_test = test[:,-1]    
    # evaluate model
    # predict based on independent variables in the test data
    score = gbm_model.score(test[:, :-1] , y_test)
    predictions = gbm_model.predict(test[:, :-1])  

    scores.append(score)
scores
plt.hist(scores)
plt.show()
alpha = 0.95                             # for 95% confidence 
p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)
lower = max(0.0, np.percentile(scores, p))  
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(scores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))