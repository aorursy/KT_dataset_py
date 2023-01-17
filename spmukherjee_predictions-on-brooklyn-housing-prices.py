import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
pd.set_option('display.max_columns', 111)
HP=pd.read_csv('../input/brooklyn_sales_map.csv')
HP.shape
HP.head()
HP.describe().append(HP.isnull().sum().rename('isnull'))
HP.columns.values
columns = ['Unnamed: 0', 'borough', 'Borough','apartment_number', 'Ext', 'Landmark','AreaSource', 'UnitsRes', 'UnitsTotal', 'LotArea', 'BldgArea','BldgClass','Easements', 'easement', 'OwnerType', 'building_class_category','sale_date', 'CT2010', 'CB2010', 'ZipCode', 'ZoneDist1', 'ZoneDist2', 'ZoneDist3', 'ZoneDist4', 'Overlay1', 'Overlay2', 'SPDist1', 'SPDist2', 'SPDist3', 'LtdHeight', 'YearBuilt', 'BoroCode', 'BBL', 'Tract2010', 'ZoneMap', 'ZMCode', 'Sanborn', 'TaxMap', 'EDesigNum', 'PLUTOMapID', 'FIRM07_FLA', 'PFIRM15_FL', 'Version', 'MAPPLUTO_F', 'APPBBL', 'APPDate', 'SHAPE_Leng', 'SHAPE_Area','CD', 'SchoolDist', 'Council', 'PolicePrct', 'HealthCent', 'SanitBoro', 'SanitDistr','FireComp','SanitSub', 'CondoNo','Address']
HP.drop(columns, inplace=True, axis=1)
HP.columns.values
HP=HP[HP['sale_price']!=0]
HP['gross_sqft']=HP['gross_sqft'].replace(0.0,HP['gross_sqft'].median())
HP['land_sqft']=HP['land_sqft'].replace(0.0,HP['land_sqft'].median())
HP['NumBldgs']= HP['NumBldgs'].fillna(HP['NumBldgs'].median())
HP['NumFloors']= HP['NumFloors'].fillna(HP['NumFloors'].median())
HP['ProxCode']= HP['ProxCode'].fillna(HP['ProxCode'].mode()[0])
HP['LotType']= HP['LotType'].fillna(HP['LotType'].mode()[0])
HP['BsmtCode']= HP['BsmtCode'].fillna(HP['BsmtCode'].mode()[0])
HP['LandUse']= HP['LandUse'].fillna(HP['LandUse'].mode()[0])
HP['AssessLand']= HP['AssessLand'].fillna(HP['AssessLand'].median())
HP['AssessTot']= HP['AssessTot'].fillna(HP['AssessTot'].median())
HP['ExemptLand']= HP['ExemptLand'].fillna(HP['ExemptLand'].median())
HP['ExemptTot']= HP['ExemptTot'].fillna(HP['ExemptTot'].median())
HP['BuiltFAR']= HP['BuiltFAR'].fillna(HP['BuiltFAR'].median())
HP['ResidFAR']= HP['ResidFAR'].fillna(HP['ResidFAR'].median())
HP['CommFAR']= HP['CommFAR'].fillna(HP['CommFAR'].median())
HP['FacilFAR']= HP['FacilFAR'].fillna(HP['FacilFAR'].mean())
HP['OwnerName']= HP['OwnerName'].fillna(value=0)
HP['IrrLotCode']= HP['IrrLotCode'].fillna(value=0)
HP['SplitZone']= HP['SplitZone'].fillna(value=0)

HP['XCoord']= HP['XCoord'].fillna(HP['XCoord'].mode()[0])
HP['YCoord']= HP['YCoord'].fillna(HP['YCoord'].mode()[0])
HP['XCoord']= HP['XCoord'].replace(0.0,HP['XCoord'].mode()[0] )
HP['YCoord']= HP['YCoord'].replace(0.0,HP['YCoord'].mode()[0] )

HP['ComArea']= HP['ComArea'].fillna(HP['ComArea'].median())
HP['ResArea']= HP['ResArea'].fillna(HP['ResArea'].median())
HP['OfficeArea']= HP['OfficeArea'].fillna(HP['OfficeArea'].median())
HP['RetailArea']= HP['RetailArea'].fillna(HP['RetailArea'].median())
HP['GarageArea']= HP['GarageArea'].fillna(HP['GarageArea'].median())
HP['OtherArea']= HP['OtherArea'].fillna(HP['OtherArea'].median())
HP['StrgeArea']= HP['StrgeArea'].fillna(HP['StrgeArea'].median())
HP['FactryArea']= HP['FactryArea'].fillna(HP['FactryArea'].median())
HP['LotFront']= HP['LotFront'].fillna(HP['LotFront'].median())
HP['LotDepth']= HP['LotDepth'].fillna(HP['LotDepth'].median())
HP['BldgFront']= HP['BldgFront'].fillna(HP['BldgFront'].median())
HP['BldgDepth']= HP['BldgDepth'].fillna(HP['BldgDepth'].median())
HP['HealthArea']= HP['HealthArea'].fillna(HP['HealthArea'].median())
HP['YearAlter1']= HP['YearAlter1'].fillna(HP['YearAlter1'].mode()[0])
HP['YearAlter2']= HP['YearAlter2'].fillna(HP['YearAlter2'].mode()[0])
HP['HistDist'].fillna(0.0, inplace=True)
HP['HistDist']=HP['HistDist'].astype('category')
HP['HistDist']=HP['HistDist'].cat.codes
HP['HistDist'].unique()
HP['neighborhood']=HP['neighborhood'].astype('category')
HP['neighborhood']=HP['neighborhood'].cat.codes
HP[['number','street name']] = HP['address'].str.split(n=1, expand=True)
del HP['address']
del HP['number']
HP['street name']=HP['street name'].astype('category')
HP['street name']=HP['street name'].cat.codes
print(HP['tax_class'].unique())
print(HP['tax_class'].isnull().sum())
print(HP['tax_class_at_sale'].unique())
print(HP['tax_class_at_sale'].isnull().sum())
HP['tax_class'] = HP['tax_class'].map({'1B': 5, '2A': 6, '2B':7, '1A':8, '2C':9, '3':3,'4':4,'2':2,'1':1})
HP['tax_class'].fillna(HP['tax_class_at_sale'], inplace=True)
HP['building_class'].fillna(HP['building_class_at_sale'], inplace=True)
HP['building_class']=HP['building_class'].astype('category')
HP['building_class_at_sale']=HP['building_class_at_sale'].astype('category')

cat_columns = HP.select_dtypes(['category']).columns
cat_columns
HP[cat_columns] = HP[cat_columns].apply(lambda x: x.cat.codes)
print(HP['OwnerName'].unique())
HP['OwnerName']= HP['OwnerName'].fillna(value=0)
print(HP['OwnerName'].isnull().sum())

HP['OwnerName']=HP['OwnerName'].astype('category')
HP['OwnerName']=HP['OwnerName'].cat.codes
HP['IrrLotCode'].unique()
HP['IrrLotCode']= HP['IrrLotCode'].fillna(value=0)
HP['IrrLotCode']= HP['IrrLotCode'].astype('category')
HP['IrrLotCode']= HP['IrrLotCode'].cat.codes
HP['SplitZone'].unique()
HP['SplitZone']= HP['SplitZone'].fillna(value=0)
HP['SplitZone']= HP['SplitZone'].astype('category')
HP['SplitZone']= HP['SplitZone'].cat.codes
HP.describe().append(HP.isnull().sum().rename('isnull'))
HP.isnull().sum().sum()
HP.shape
HP.boxplot(column='gross_sqft')
from scipy import stats
HP=HP[(np.abs(stats.zscore(HP)) < 3).all(axis=1)]
HP.boxplot(column='gross_sqft')
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
X = HP.drop('sale_price',axis=1)
y = HP['sale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HP['zip_code']
X = HP.drop('sale_price',axis=1)
y = HP['sale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HP['total_units']
columns = ['YearAlter2', 'XCoord', 'year_of_sale',  'YCoord', 'ComArea','LotDepth', 'LotType', 'NumFloors', 'LotFront', 'tax_class_at_sale', 'building_class', 'ResArea', 'SplitZone', 'BldgDepth', 'ResidFAR', 'LandUse', 'HealthArea', 'gross_sqft', 'BldgFront', 'year_built', 'IrrLotCode', 'AssessTot', 'land_sqft', 'FacilFAR', 'building_class_at_sale', 'NumBldgs']
HP.drop(columns, inplace=True, axis=1)
HP.columns.values
HP.head()
HP.shape
import seaborn as sns
sns.pairplot(HP,y_vars=['sale_price'], x_vars=['AssessLand', 'HistDist', 'OtherArea', 'StrgeArea', 'YearAlter1', 'BuiltFAR'],palette='Dark2')
sns.pairplot(HP,y_vars=['sale_price'], x_vars=['commercial_units', 'lot', 'neighborhood', 'residential_units', 'tax_class', 'BsmtCode'],palette='Dark2')
sns.pairplot(HP,y_vars=['sale_price'], x_vars=['GarageArea', 'OwnerName', 'ExemptTot', 'FactryArea', 'block'],palette='Dark2')
sns.pairplot(HP,y_vars=['sale_price'], x_vars=['RetailArea', 'CommFAR', 'OfficeArea', 'ProxCode', 'ExemptLand', 'street name'],palette='Dark2')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.feature_selection import RFE
X = HP.drop('sale_price',axis=1)
y = HP['sale_price']

Xtrn, Xtest, Ytrn, Ytest = train_test_split(X,y,test_size=0.3, random_state=42)
models = [LinearRegression(), linear_model.Lasso(alpha=0.1), Ridge(alpha=100.0), RandomForestRegressor(n_estimators=100, max_features='sqrt'), KNeighborsRegressor(n_neighbors=6),DecisionTreeRegressor(max_depth=4), ensemble.GradientBoostingRegressor()]

TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
    print(model)
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    model.fit(Xtrn, Ytrn)
    tmp['R2_Price'] = r2_score(Ytest, model.predict(Xtest))
    print('score on training',model.score(Xtrn, Ytrn))
    print('r2 score',r2_score(Ytest, model.predict(Xtest)))
    TestModels = TestModels.append([tmp])
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()
HP.columns.values
HP_list=list(HP.columns.values)
HP_list1=list(HP.columns.values)

names=HP_list1
feature_cols =['neighborhood', 'tax_class', 'block', 'lot', 'residential_units', 'commercial_units', 'OwnerName', 'OfficeArea',
       'RetailArea', 'GarageArea', 'StrgeArea', 'FactryArea', 'OtherArea', 'ProxCode', 'BsmtCode', 'AssessLand', 'ExemptLand', 'ExemptTot',
       'YearAlter1', 'HistDist', 'BuiltFAR', 'CommFAR', 'street name']
target=['sale_price']
X=HP[feature_cols].dropna()
y=np.array(HP[target].dropna()).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#use linear regression as the model
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)
#rank all features, i.e continue the elimination until the last one
rfe = RFE(model, n_features_to_select=10, step=1)
rfe.fit(X,y)
print('Features sorted by their rank:')
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

# Plot feature importance
feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
columns = ['OfficeArea', 'GarageArea', 'StrgeArea', 'FactryArea', 'OtherArea', 'BsmtCode', 'CommFAR']
HP.drop(columns, inplace=True, axis=1)
HP.head()
X = HP.drop('sale_price',axis=1)
y = HP['sale_price']
Xtrn, Xtest, Ytrn, Ytest = train_test_split(X,y,test_size=0.3, random_state=42)

model = ensemble.GradientBoostingRegressor()
parameters = {'learning_rate':[0.1,0.2,0.3],'n_estimators':[50,100,150]}
grid_obj = GridSearchCV(model, parameters,refit=True,cv=3,verbose=10)
grid_obj = grid_obj.fit(Xtrn, Ytrn)
print(grid_obj.fit(Xtrn, Ytrn))
grid_obj.best_params_
params = grid_obj.best_params_
model = ensemble.GradientBoostingRegressor(**params)
model.fit(Xtrn, Ytrn.values.ravel())
test_score = np.zeros((params['n_estimators']),dtype=np.float64)

# Predict
Ypred = model.predict(Xtest)
model_mse = mean_squared_error(Ypred, Ytest)
model_rmse = np.sqrt(model_mse)
print('Gradient Boosting RMSE: %.4f' % model_rmse)
print('score on training',model.score(Xtrn, Ytrn))
print('r2 score',r2_score(Ytest, model.predict(Xtest)))
df = pd.DataFrame({'Actual': Ytest, 'Predicted': Ypred, 'Difference': Ypred-Ytest})
print(df.head())

for i,Ypred in enumerate(model.staged_predict(Xtest)):
    test_score[i]=model.loss_(Ytest.values.ravel(),Ypred)

# Plot training deviance
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, model.train_score_, 'b-',label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
colors=('red','blue')
plt.scatter(Ypred,Ytest,c=colors)
plt.xlabel('sale_price_pred')
plt.ylabel('sale_price_test')
plt.show