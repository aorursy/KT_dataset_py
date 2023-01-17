import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
    
df = pd.read_csv("../input/train.csv",na_values=['?',''],delimiter=',',delim_whitespace=False)
data_aux = df

# Correlation Matrix all features
correlation_matrice = data_aux.corr()
f, ax = plt.subplots( figsize=(15, 12))
sns.heatmap(correlation_matrice,vmin=0.2, vmax=0.8, square= True, cmap= 'BuPu')
plt.xlabel('The house features in the x axis',fontsize= 13)
plt.ylabel('The house features in the y axis',fontsize= 13)
plt.title('Figure 1 - The correlation matrix between all the featues ', fontsize= 16);

#Scatter plot of thr most important features
cols = ['SalePrice', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'GrLivArea', 'GarageCars']
sns.pairplot(data_aux[cols], size = 2.8)
plt.suptitle('Figure 2 - The scatter plot of the top features ',x=0.5, y=1.01, verticalalignment='top', fontsize= 18)
plt.tight_layout()
plt.show();

# regplot of GrLivArea/SalePrice

ax = sns.regplot(x=data_aux['GrLivArea'], y=data_aux['SalePrice'])
plt.ylabel('SalePrice', fontsize= 10)
plt.xlabel('GrLivArea', fontsize= 10)
plt.title('Figure 3 - regplot of the GrLivArea with the SalePrice', fontsize= 12)
plt.show();
# Removing the outliers
# We sort the values by GrLivArea and select the two lager values, and we locate the index number 
# to use it in order to drop corresponding rows.
g_out = data_aux.sort_values(by="GrLivArea", ascending = False).head(2)
g_out
data_aux.drop([523,1298], inplace = True)
data_aux.reset_index(inplace=True)

# regplot of TotalBsmtSF/SalePrice

ax = sns.regplot(x=data_aux['TotalBsmtSF'], y=data_aux['SalePrice'])
plt.ylabel('SalePrice', fontsize= 13)
plt.xlabel('TotalBsmtSF', fontsize= 13)
plt.title('Figure 4 regplot of the TotalBsmtSF with the SalePrice', fontsize= 12);
plt.show()
print("Shape of training set: ", df.shape)
print("Missing values before remove NA: ")
print(data_aux.columns[data_aux.isnull().any()])
#Alley
data_aux.Alley.fillna(inplace=True,value='No')
#BsmtQual
data_aux.BsmtQual.fillna(inplace=True,value='No')
#BsmtCond
data_aux.BsmtCond.fillna(inplace=True,value='No')
#BsmtExposure
data_aux.BsmtExposure.fillna(inplace=True,value='No')
#BsmtFinType1
data_aux.BsmtFinType1.fillna(inplace=True,value='No')
#BsmtFinType2
data_aux.BsmtFinType2.fillna(inplace=True,value='No')
#FireplaceQu
data_aux.FireplaceQu.fillna(inplace=True,value='No') 
#GarageType
data_aux.GarageType.fillna(inplace=True,value='No')
#GarageFinish
data_aux.GarageFinish.fillna(inplace=True,value='No')
#GarageQual 
data_aux.GarageQual.fillna(inplace=True,value='No')    
#GarageCond
data_aux.GarageCond.fillna(inplace=True,value='No')
#PoolQC
data_aux.PoolQC.fillna(inplace=True,value='No')    
#Fence
data_aux.Fence.fillna(inplace=True,value='No')
#MiscFeature
data_aux.MiscFeature.fillna(inplace=True,value='No')
    
print("Missing values after insert No, i.e., real missing values: ")
print(data_aux.columns[data_aux.isnull().any()])

#Numeric fields    
data_aux.BsmtFinSF1.fillna(inplace=True,value=0)
data_aux.BsmtFinSF2.fillna(inplace=True,value=0)
data_aux.BsmtUnfSF.fillna(inplace=True,value=0)
data_aux.TotalBsmtSF.fillna(value=0,inplace=True)
data_aux.BsmtFullBath.fillna(inplace=True,value=0)
data_aux.BsmtHalfBath.fillna(inplace=True,value=0)
data_aux.GarageCars.fillna(value=0,inplace=True)
data_aux.GarageArea.fillna(value=0,inplace=True)
data_aux.LotFrontage.fillna(inplace=True,value=0)
data_aux.GarageYrBlt.fillna(inplace=True,value=0)
data_aux.MasVnrArea.fillna(inplace=True,value=0)
    
#####Categorial fields
#KitchenQual
data_aux.KitchenQual = data_aux.KitchenQual.mode()[0]
#Functional
data_aux.Functional = data_aux.Functional.mode()[0]
#Utilities
data_aux.Utilities = data_aux.Utilities.mode()[0]  
#SaleType
data_aux.SaleType  = data_aux.SaleType.mode()[0]
#Exterior1st- nao posso remover linhas do teste
data_aux.Exterior1st = data_aux.Exterior1st.mode()[0]
#Exterior2nd
data_aux.Exterior2nd = data_aux.Exterior2nd.mode()[0]       
#Electrical - remove the records where the value is NA
data_aux.Electrical = df['Electrical'].mode()[0]
#MSZoning   - tem NA apenas na base de teste. Como nao posso remover linhas removo a coluna   
data_aux.MSZoning = data_aux.MSZoning.mode()[0]
#MasVnrType - remove the records where the value is NA 
data_aux.MasVnrType=df['MasVnrType'].mode()[0]
print("After we imputed the missing values, the status of the dataset is: ")
print(data_aux.columns[data_aux.isnull().any()])


#Mapping ordinal features

#LotShape: General shape of property
lotshape_map = {'Reg':'8','IR1':'6','IR2':'4','IR3':'2'}
data_aux.LotShape = data_aux.LotShape.map(lotshape_map)
data_aux.LotShape = data_aux.LotShape.astype('int64')

#Utilities: Type of utilities available       
utilities_map = {'AllPub':'8','NoSewr':'6','NoSeWa':'4','ELO':'2'}
data_aux.Utilities = data_aux.Utilities.map(utilities_map)
data_aux.Utilities = data_aux.Utilities.astype('int64')
    
#LandSlope: Slope of property
landslope_map = {'Gtl':'6','Mod':'4','Sev':'2'}
data_aux.LandSlope = data_aux.LandSlope.map(landslope_map)
data_aux.LandSlope = data_aux.LandSlope.astype('int64')

#ExterQual: Evaluates the quality of the material on the exterior 
quality_map = {'Ex':'10','Gd':'8','TA':'6','Fa':'4','Po':'2','No':'0'}
data_aux.ExterQual = data_aux.ExterQual.map(quality_map)
data_aux.ExterQual = data_aux.ExterQual.astype('int64')

#ExterCond: Evaluates the present condition of the material on the exterior
data_aux.ExterCond = data_aux.ExterCond.map(quality_map)
data_aux.ExterCond = data_aux.ExterCond.astype('int64')

#BsmtQual: Evaluates the height of the basement
data_aux.BsmtQual = data_aux.BsmtQual.map(quality_map)
data_aux.BsmtQual = data_aux.BsmtQual.astype('int64')

#BsmtCond: Evaluates the general condition of the basement
data_aux.BsmtCond = data_aux.BsmtCond.map(quality_map)
data_aux.BsmtCond = data_aux.BsmtCond.astype('int64')

#HeatingQC: Heating quality and condition
data_aux.HeatingQC = data_aux.HeatingQC.map(quality_map)
data_aux.HeatingQC = data_aux.HeatingQC.astype('int64')
        
#KitchenQual: Kitchen quality
data_aux.KitchenQual = data_aux.KitchenQual.map(quality_map)
data_aux.KitchenQual = data_aux.KitchenQual.astype('int64')

#FireplaceQu: Fireplace quality
data_aux.FireplaceQu = data_aux.FireplaceQu.map(quality_map)
data_aux.FireplaceQu = data_aux.FireplaceQu.astype('int64')

#GarageFinish: Interior finish of the garage
garage_map = {'Fin':'6', 'RFn':'4', 'Unf':'2', 'No':'0'}    
data_aux.GarageFinish = data_aux.GarageFinish.map(garage_map)
data_aux.GarageFinish = data_aux.GarageFinish.astype('int64')

#GarageQual: Garage quality
data_aux.GarageQual = data_aux.GarageQual.map(quality_map)
data_aux.GarageQual = data_aux.GarageQual.astype('int64')

#GarageCond: Garage condition
data_aux.GarageCond = data_aux.GarageCond.map(quality_map)
data_aux.GarageCond = data_aux.GarageCond.astype('int64')

#PoolQC: Pool quality
data_aux.PoolQC = data_aux.PoolQC.map(quality_map)
data_aux.PoolQC = data_aux.PoolQC.astype('int64')


#Converting numeric columns to nominal before applying one-hot encoding convertion
#After converting to String they will be treated as categorical

# MSSubClass as str
data_aux['MSSubClass'] = data_aux['MSSubClass'].astype("str")
# Year and Month to categorical
data_aux['YrSold'] = data_aux['YrSold'].astype("str")
data_aux['MoSold'] = data_aux['MoSold'].astype("str")    
#Converting from str to int of ordinal fields
data_aux.OverallCond = data_aux.OverallCond.astype("int64")
data_aux.OverallQual = data_aux.OverallQual.astype("int64")
data_aux['KitchenAbvGr'] = data_aux['KitchenAbvGr'].astype("int64")
#Finally, applying one-hot encoding

data_train = pd.get_dummies(data_aux)
print("New  shape after one-hot encoding:" , np.shape(data_train))



from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.api as sm

x_train = data_train.drop('SalePrice',axis=1)
y_train = data_train['SalePrice']

scaler = preprocessing.StandardScaler()
x_train_s = scaler.fit_transform(x_train)   

linear1 = LinearRegression()
linear1.fit(x_train_s, y_train)
pred = linear1.predict(x_train_s)
ax = sns.regplot(x=pred,y=y_train-pred,lowess=True,line_kws={"color":"black"})
ax.set_title('Figure 5 - Residual plot for original data.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.show();

print("Mean squared error: ",np.log(sum(np.power((y_train-pred),2))/x_train.shape[0]))

#results = sm.OLS(y_train,x_train_s).fit()
#print(results.summary())
linear2 = LinearRegression()
linear2.fit(x_train_s, np.log(y_train))
pred = linear2.predict(x_train_s)
ax = sns.regplot(x=pred,y=np.log(y_train)-pred,lowess=True,line_kws={"color":"black"})
ax.set_title('Figure 6 - Residual plot for log-transformed SalePrice.')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
plt.show()
print("Mean squared error: ",(sum(np.power((np.log(y_train)-pred),2))/x_train.shape[0]))


data_train['SalePrice'] = np.log(data_train.SalePrice)
data_train['TotalSF'] = data_train['TotalBsmtSF'] + data_train['1stFlrSF'] + data_train['2ndFlrSF'] + data_train['GarageArea']
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import SelectFromModel
from tabulate import tabulate

#Tree-based feature selection
y_train = (data_train['SalePrice'])
x_train = (data_train.drop('SalePrice',axis=1))

#clf = ExtraTreesRegressor(random_state=0,n_estimators=1400)
clf = RandomForestRegressor(n_estimators=1400, criterion='mse', 
                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False)

clf = clf.fit(x_train,y_train)

#Organinzing the features selected for visualization
pd.set_option('display.max_columns', None)#to print all the columns of a data frame
data = np.zeros((1,x_train.shape[1])) 
data = pd.DataFrame(data, columns=x_train.columns)
data.iloc[0] = clf.feature_importances_
data = data.T.sort_values(data.index[0], ascending=False).T
print("Ten most important features selected with tree-based selection: \n")
print(tabulate(data.iloc[:,0:5],headers='keys', tablefmt='psql'))
print(tabulate(data.iloc[:,6:11],headers='keys', tablefmt='psql'))

#Select the features based on the threshold
model = SelectFromModel(clf, prefit=True,threshold=1e-3)
#Reduce data to the selected features.
aux = model.transform(x_train)

print("\n New shape for train after tree-based feature selection: {}".format(aux.shape))
data_train_less_features_aux = pd.DataFrame(aux)
data_train_less_features_aux.columns = [data.columns[i] for i in range(0,aux.shape[1]) ]
print("\n Features selected :")
print(data_train_less_features_aux.columns)
data_train_less_features = pd.concat([data_train_less_features_aux,pd.DataFrame(y_train)],axis=1)

print("\n End of the process of selecting best features. \n")
#Some usefull packages
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn   import metrics
from sklearn.model_selection import train_test_split


#Python dictionary to collect the results
RMSE_results = {
    
    'LinearRegression':0.0,
    'Lasso':0.0,
    'Ridge':0.0,
    'ElasticNet':0.0,
    'SVM':0.0,
    'NN':0.0,    
    'RF':0.0
    
}


from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, Lasso, LassoCV, ElasticNetCV
from sklearn import preprocessing


x_train = data_train_less_features.drop('SalePrice',axis=1).values
y_train = data_train_less_features['SalePrice'].values



print("Linear Regression \n")

#Cross validation
classifierLinearRegression = LinearRegression(fit_intercept=True, normalize=False,
                                              copy_X=True, n_jobs=1)
kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0
pred = []

for training, test in kf.split(x_train):
    fold+=1    
    pred = []    
    
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    x_train_fold = scaler.fit_transform(x_train[training,:])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = y_train[training]    
    y_test_fold = y_train[test]
    
    classifierLinearRegression = classifierLinearRegression.fit(x_train_fold, y_train_fold)
    pred = classifierLinearRegression.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        

    
#Calculating the error.
cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE: {}".format(score))    
RMSE_results['LinearRegression'] = score
print("Ridge \n\n")

classifier = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, 
                     scoring=None, cv=None, gcv_mode=None, store_cv_values=False)

kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0
pred = []

for training, test in kf.split(x_train):
    fold+=1
    pred = []
    
    scaler = preprocessing.StandardScaler()
    x_train_fold = scaler.fit_transform(x_train[training])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = y_train[training]    
    y_test_fold = y_train[test]

    
    classifier.fit(x_train_fold, y_train_fold)
    pred = classifier.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        


cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE: {}".format(score))    
RMSE_results['Ridge'] = score
print("Lasso \n\n")

classifier = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, 
                     normalize=False, precompute='auto', max_iter=1000, tol=0.0001, 
                     copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False,
                     random_state=None, selection='cyclic')

kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0
pred = []

for training, test in kf.split(x_train):
    fold+=1
    pred = []
    
    scaler = preprocessing.StandardScaler()
    x_train_fold = scaler.fit_transform(x_train[training])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = y_train[training]    
    y_test_fold = y_train[test]
        
    classifier.fit(x_train_fold, y_train_fold)
    pred = classifier.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        


cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE: {}".format(score))    
RMSE_results['Lasso'] = score
print("Elastic Net \n\n")

classifier = ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
                          fit_intercept=True, normalize=False, precompute='auto', 
                          max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, 
                          n_jobs=1, positive=False, random_state=None, selection='cyclic')

kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0
pred = []

for training, test in kf.split(x_train):
    fold+=1
    pred = []   
        
    scaler = preprocessing.StandardScaler()
    x_train_fold = scaler.fit_transform(x_train[training])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = y_train[training]    
    y_test_fold = y_train[test]
    
    classifier.fit(x_train_fold, y_train_fold)
    pred = classifier.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        

cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE: {}".format(score))    
RMSE_results['ElasticNet'] = score
fig_1 = pd.DataFrame(RMSE_results, index=(1,) )
ax = fig_1.plot(kind='bar',figsize=(10,5), title="Figure 7 - Comparison of RMSE among models")
ax.set_ylabel('Root Mean Square Error')
ax.set_xlabel('Models')

from sklearn.svm import SVR
print("SVM")

classifierSVR = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
                    C=.01, epsilon=0.1, shrinking=True, cache_size=200,
                    verbose=False, max_iter=-1) 

kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0

for training, test in kf.split(x_train):
    fold+=1
    pred = []
        
    scaler = preprocessing.StandardScaler()
    x_train_fold = scaler.fit_transform(x_train[training])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = (y_train[training])    
    y_test_fold = (y_train[test])
    
    classifierSVR = classifierSVR.fit(x_train_fold, y_train_fold)
    pred = classifierSVR.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        


cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE: {}".format(score))    

RMSE_results['SVM'] = score
from sklearn.neural_network import MLPRegressor

print("\nNeural Network")


classifier = MLPRegressor( hidden_layer_sizes=(80,50,20), activation='relu',solver='adam', 
                          alpha=1e-3, batch_size='auto', learning_rate='constant', 
                          learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                          shuffle=True, random_state=7, tol=0.0001, 
                          verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                          early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                          beta_2=0.999, epsilon=1e-08)

kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0

for training, test in kf.split(x_train):
    fold+=1    
    pred = []
    
    scaler = preprocessing.StandardScaler()
    x_train_fold = scaler.fit_transform(x_train[training])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = y_train[training]    
    y_test_fold = y_train[test]

    classifier.fit(x_train_fold, y_train_fold)
    pred = classifier.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        

   
cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE: {}".format(score))    

RMSE_results['NN'] = score
##Random Forests
from sklearn.ensemble import RandomForestRegressor

print("\n Random Forests ")
print("\n Full Features ")

y_train_rf = data_train['SalePrice'].values
x_train_rf = data_train.drop('SalePrice',axis=1).values

classifierAllFeatures = RandomForestRegressor(n_estimators=500, criterion='mse', 
                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False)

kf = KFold(5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0

for training, test in kf.split(x_train_rf):
    fold+=1
    pred = []
        
    x_train_fold = x_train_rf[training]
    y_train_fold = y_train_rf[training]
    x_test_fold = x_train_rf[test]
    y_test_fold = y_train_rf[test]
    
    classifierAllFeatures.fit(x_train_fold, y_train_fold)
    pred = classifierAllFeatures.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        


cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE using all features: {}".format(score))    


###########Less features
print("\n Less features ")

kf = KFold(n_splits=5, random_state=7, shuffle=True)    
cv_y = []
cv_pred = []
fold = 0
pred = []

classifierRF_lessFeatures = RandomForestRegressor(n_estimators=500, criterion='mse', 
                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False)

for training, test in kf.split(x_train):
    fold+=1
    pred = []
    
    scaler = preprocessing.StandardScaler()
    x_train_fold = scaler.fit_transform(x_train[training])
    x_test_fold = scaler.transform(x_train[test])

    y_train_fold = y_train[training]    
    y_test_fold = y_train[test]
    
    classifierRF_lessFeatures.fit(x_train_fold, y_train_fold)
    pred = classifierRF_lessFeatures.predict(x_test_fold)
    cv_y.append(y_test_fold)
    cv_pred.append(pred)        
    
cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y,cv_pred))
print("\n Average RMSE less features: {}".format(score))    

RMSE_results['RF'] = score
fig_1 = pd.DataFrame(RMSE_results, index=(1,) )
ax = fig_1.plot(kind='bar',figsize=(10,5), title="Figure 8 - Comparison of RMSE among models")
ax.set_ylabel('Root Mean Square Error')
ax.set_xlabel('Models')
