#importing the libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head(10)
#Everything is good for now
data['SalePrice'].describe()
sns.boxplot(data['SalePrice'],linewidth=1.5)
sns.distplot(data['SalePrice'],color='red')
corrmap = data.corr()
ax = plt.subplots(figsize=(19, 16))
sns.heatmap(corrmap, vmax=.8, square=True,cmap='coolwarm')
#here we are going to choose the most correlated features to target data 
corr_cols=corrmap.nlargest(9,'SalePrice')['SalePrice'].index
ax = plt.subplots(figsize=(9, 7))
sns.heatmap(np.corrcoef(data[corr_cols].values.T),cbar=True,cmap='coolwarm',
           annot=True,square=True,fmt='.2f',annot_kws={'size':9},
            xticklabels=corr_cols.values,yticklabels=corr_cols.values)
#here we choose just best8 features the other below 0.5 so it better focus on the important one 
LAvsPrice=pd.concat([data['SalePrice'],data['GrLivArea']],axis=1)
sns.regplot(x='GrLivArea',y='SalePrice',data=LAvsPrice)
data.sort_values(by='GrLivArea',ascending = False)[:2]
data=data.drop(data[data['Id']==1299].index)
data=data.drop(data[data['Id']==524].index)
print("done!!")
LAvsPrice=pd.concat([data['SalePrice'],data['GrLivArea']],axis=1)
sns.regplot(x='GrLivArea',y='SalePrice',data=LAvsPrice)
X=data.filter(['OverallQual', 'GrLivArea', 'GarageCars',
      'TotalBsmtSF','1stFlrSF', 'FullBath','TotRmsAbvGrd'],axis=1)
#notice here that we didnt include 'GarageArea' 'cause obviously is the same as 'GarageCars'
y=data.filter(['SalePrice'],axis=1)

X.head(10)
#y.head(10)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=40)

#X_train
#X_test
#y_train
#y_test
y_train
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

SelectedModel = GradientBoostingRegressor(learning_rate=0.05, max_depth=2, 
                                        min_samples_leaf=14,
                                        min_samples_split=50, n_estimators=3000,
                                        random_state=40)
SelectedParameters = {'loss':('ls','huber','lad'
                            ,'quantile'),'max_features':('auto','sqrt','log2')}


GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters,return_train_score=True)
GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score'
                                                               , 'params' , 'rank_test_score' , 'mean_fit_time']]

print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)#this is what we need 
print('Best Estimator is :', GridSearchModel.best_estimator_)
#accourding to GridSearchCV the best parametre is {'loss': 'huber', 'max_features': 'sqrt'} 
GBR = GradientBoostingRegressor(learning_rate=0.05, loss='huber', max_depth=2, 
                                       max_features='sqrt', min_samples_leaf=14,
                                       min_samples_split=50, n_estimators=3000,
                                       random_state=42)
GBR.fit(X_train, y_train)
print("done!!!")

print("Train Score", GBR.score(X_train, y_train))
print("Test Score", GBR.score(X_test, y_test))
#predict the test data 
y_pred = GBR.predict(X_test)
print("done again !!")
from sklearn.metrics import mean_absolute_error
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)
RealData=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
RealData.head(10)
X1=RealData.filter(['OverallQual', 'GrLivArea', 'GarageCars',
      'TotalBsmtSF','1stFlrSF', 'FullBath','TotRmsAbvGrd'],axis=1)

X1.head(10)
#there is some values are null so we need to get rid of them 
X1=X1.fillna(0)
print("coool!!")
#repeat the same step we did with the train file ...
sc0 = StandardScaler()
X1 = sc0.fit_transform(X1)

#now let create the file and save our prediction 

Doc=pd.DataFrame()
Doc['Id']=RealData['Id']
Doc['SalePrice']=np.round(GBR.predict(X1),2)
print(Doc['SalePrice'].head(10))
Doc.to_csv('SalePrice_submission.csv',index=False)
print('great !!! ')