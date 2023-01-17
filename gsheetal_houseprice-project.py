#Import Libraries
import pandas as pd
import numpy as np
import seaborn as ss
import matplotlib.pyplot as pic
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm,skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#Loading Data
HousingPrice_Train = pd.read_csv("..../HousingPrice_Train.csv")
HousingPrice_Test = pd.read_csv("..../HousingPrice_Test.csv")
#Viewing Data
print(HousingPrice_Train.head())
print()
print()
print(HousingPrice_Test.head())
#Number of Rows and Columns
print ('The train data has {0} rows and {1} columns'.format(HousingPrice_Train.shape[0],HousingPrice_Train.shape[1]))
print ('----------------------------')
print ('The test data has {0} rows and {1} columns'.format(HousingPrice_Test.shape[0],HousingPrice_Test.shape[1]))
#Number of Numeric and Categorical Columns
print ("There are {} numeric and {} categorical columns in train data".format(HousingPrice_Train.select_dtypes(include=[np.number]).shape[1],HousingPrice_Train.select_dtypes(exclude=[np.number]).shape[1]))
print ('----------------------------')
print ("There are {} numeric and {} categorical columns in test data".format(HousingPrice_Test.select_dtypes(include=[np.number]).shape[1],HousingPrice_Test.select_dtypes(exclude=[np.number]).shape[1]))
#Summary of Train Data
HousingPrice_Train.info()
#Summary of Test Data
HousingPrice_Test.info()
#Save the 'Id' column
train_ID = HousingPrice_Train['Id']
test_ID = HousingPrice_Test['Id']
#Summary of Target Variable
HousingPrice_Train.SalePrice.describe()
#Distribution of Target Variable
ss.distplot(HousingPrice_Train['SalePrice'], fit=norm).set(title="Histogram of SalePrice");
print ("The Skewness of SalePrice is {}".format(HousingPrice_Train['SalePrice'].skew()))
print("The Kurtosis of SalePrice is {}".format(str(HousingPrice_Train['SalePrice'].kurt())))
#Let's log transform this variable 
trans_Sale = np.log(HousingPrice_Train['SalePrice'])
ss.distplot(trans_Sale,fit=norm).set(title="Histogram of Log Transformation of SalePrice");
print ("The Skewness of log transformed SalePrice is {}".format(trans_Sale.skew()))
print("The Kurtosis of log transformed SalePrice is {}".format(str(trans_Sale.kurt())))
#QQ-plot of Target Variable
res = stats.probplot(HousingPrice_Train['SalePrice'], plot=pic)
pic.show()
Num_data = HousingPrice_Train.select_dtypes(include=[np.number])
Catg_data = HousingPrice_Train.select_dtypes(exclude=[np.number])
print("Numeric data coulumns are ", Num_data.shape[1])
print()
print("Those are ", Num_data.columns )
print("Categorical data coulumns are ", Catg_data.shape[1])
print()
print("Those are ", Catg_data.columns )
#Summary of Numeric Data
display(Num_data.describe().transpose())
#Summary of Categorical Data
display(Catg_data.describe().transpose())
del Num_data['Id']
corr = Num_data.corr()
half = np.zeros_like(corr, dtype=np.bool)
half[np.triu_indices_from(half)] = True
pic.figure(figsize=(10, 6))
pic.title('Overall Correlation of House Prices', fontsize=15)
ss.heatmap(corr, mask=half, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':12})
pic.show()
#Numeric Correlation Score
print("The Top 15 varaibles highly correlated with SalePrice")
print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n') 
print ('----------------------')
print()
print("The Top 5 varaibles negatively correlated with SalePrice")
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
#Attributes pairs whose correlation values are more than 0.6
col = Num_data.columns
for i in range(len(col)):
    for j in range(i + 1, len(col)):
        attr01 = col[i];
        attr02 = col[j];
        if corr[attr01][attr02] > 0.6:
            print("%s, %s: %.2f" % (attr01, attr02, corr[attr01][attr02] * 100))
HousingPrice_Train['OverallQual'].unique()
#The overall quality is measured on a scale of 1 to 10. 
OverallQual_Detail = HousingPrice_Train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(OverallQual_Detail)
print()
OverallQual_Detail.plot(kind='bar', color='blue').set(title = "Bar plot between OverallQual and SalePrice");
#Let's visualize variable GrLivArea and understand their behavior.
ss.scatterplot(x=HousingPrice_Train['GrLivArea'], y=HousingPrice_Train['SalePrice']).set(title="Scatter plot between GrLivArea and SalePrice");
#SaleCondition explains the condition of sale. 
SaleCondition_Detail = HousingPrice_Train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
print(SaleCondition_Detail)
print()
SaleCondition_Detail.plot(kind='bar',color='orange').set(title = "Bar plot between SaleCondition and SalePrice");
#Create numeric plots
num_pic = [f for f in HousingPrice_Train.columns if HousingPrice_Train.dtypes[f] != 'object']
num_pic.remove('Id')
num_pic.remove('SalePrice')
num = pd.melt(HousingPrice_Train, value_vars = num_pic)
n_p = ss.FacetGrid (num, col='variable', col_wrap=6, sharex=False, sharey = False)
n_p.map(ss.distplot, 'value');
#boxplots for visualizing categorical variables.
def boxplot(x,y,**kwargs):
            ss.boxplot(x=x,y=y)
            x = pic.xticks(rotation=90)

catg_pic = [f for f in HousingPrice_Train.columns if HousingPrice_Train.dtypes[f] == 'object']

catg = pd.melt(HousingPrice_Train, id_vars='SalePrice', value_vars=catg_pic)
c_p = ss.FacetGrid (catg, col='variable', col_wrap=5, sharex=False, sharey=False)
c_p.map(boxplot, 'value','SalePrice');

#Removing Outliers and plot it
HousingPrice_Train.drop(HousingPrice_Train[HousingPrice_Train['GrLivArea'] > 5000].index, inplace=True)
print ('After removing outlier, the train data has {0} rows and {1} columns'.format(HousingPrice_Train.shape[0],HousingPrice_Train.shape[1]))
print()
ss.scatterplot(x=HousingPrice_Train['GrLivArea'], y=HousingPrice_Train['SalePrice']).set(title="Scatter plot between GrLivArea and SalePrice after Outlier Removal");
# Saving train & test shapes
train_nrow = HousingPrice_Train.shape[0]
test_nrow = HousingPrice_Test.shape[0]
# Creating SalePrice variable
Sale_Data = HousingPrice_Train.iloc[:,-1]
Sale_Data
#Merge train and test data to form new complete data set
whole_data = pd.concat([HousingPrice_Train, HousingPrice_Test], sort=False)
whole_data
print ('The whole dataset has {0} rows and {1} columns'.format(whole_data.shape[0],whole_data.shape[1]))
#Delete columns from the dataset
whole_data.drop(['Id'], axis=1, inplace=True)
whole_data.drop(['SalePrice'], axis=1, inplace=True)
#Handling Null Values
Nul_C = pd.DataFrame(whole_data.isnull().sum().sort_values(ascending=False)[:30])
Nul_P = pd.DataFrame(round(whole_data.isnull().sum().sort_values(ascending = False)/len(whole_data)*100,2)[round(whole_data.isnull().sum().sort_values(ascending = False)/len(whole_data)*100,2) != 0])
Nul_data = pd.concat([Nul_C,Nul_P],axis=1, sort=False)
Nul_data.columns = ['Null Count','Null Percent']
Nul_data.index.name = 'Feature'
Nul_data
# Visualising null data
ss.barplot(x=Nul_data.index, y=Nul_data['Null Percent'])
pic.xticks(rotation='90')
pic.xlabel('Features')
pic.ylabel('Percent of Null Values')
pic.title('Feature wise Null Data Analysis');
#Data description says NA means "No"
whole_data["PoolQC"] = whole_data["PoolQC"].fillna("None")
whole_data["MiscFeature"] = whole_data["MiscFeature"].fillna("None")
whole_data["Alley"] = whole_data["Alley"].fillna("None")
whole_data["Fence"] = whole_data["Fence"].fillna("None")
whole_data["FireplaceQu"] = whole_data["FireplaceQu"].fillna("None")
whole_data["GarageType"] = whole_data["GarageType"].fillna("None")
whole_data["GarageFinish"] = whole_data["GarageFinish"].fillna("None")
whole_data["GarageQual"] = whole_data["GarageQual"].fillna("None")
whole_data["GarageCond"] = whole_data["GarageCond"].fillna("None")
whole_data["BsmtQual"] = whole_data["BsmtQual"].fillna("None")
whole_data["BsmtCond"] = whole_data["BsmtCond"].fillna("None")
whole_data["BsmtExposure"] = whole_data["BsmtExposure"].fillna("None")
whole_data["BsmtFinType1"] = whole_data["BsmtFinType1"].fillna("None")
whole_data["BsmtFinType2"] = whole_data["BsmtFinType2"].fillna("None")
whole_data["MasVnrType"] = whole_data["MasVnrType"].fillna("None")
whole_data['MSSubClass'] = whole_data['MSSubClass'].fillna("None")
whole_data["LotFrontage"] = whole_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#Replacing missing data with 0
whole_data["MasVnrArea"] = whole_data["MasVnrArea"].fillna(0)
whole_data["GarageYrBlt"] = whole_data["GarageYrBlt"].fillna(0)
whole_data["GarageArea"] = whole_data["GarageArea"].fillna(0)
whole_data["GarageCars"] = whole_data["GarageCars"].fillna(0) 
whole_data["BsmtFinSF1"] = whole_data["BsmtFinSF1"].fillna(0)
whole_data["BsmtFinSF2"] = whole_data["BsmtFinSF2"].fillna(0)
whole_data["BsmtUnfSF"] = whole_data["BsmtUnfSF"].fillna(0)
whole_data["TotalBsmtSF"] = whole_data["TotalBsmtSF"].fillna(0)
whole_data["BsmtFullBath"] = whole_data["BsmtFullBath"].fillna(0)
whole_data["BsmtHalfBath"] = whole_data["BsmtHalfBath"].fillna(0)
#Substitute the most frequent string
whole_data['MSZoning'] = whole_data['MSZoning'].fillna(whole_data['MSZoning'].mode()[0])
whole_data['Electrical'] = whole_data['Electrical'].fillna(whole_data['Electrical'].mode()[0])
whole_data['KitchenQual'] = whole_data['KitchenQual'].fillna(whole_data['KitchenQual'].mode()[0])
whole_data['Exterior1st'] = whole_data['Exterior1st'].fillna(whole_data['Exterior1st'].mode()[0])
whole_data['Exterior2nd'] = whole_data['Exterior2nd'].fillna(whole_data['Exterior2nd'].mode()[0])
whole_data['SaleType'] = whole_data['SaleType'].fillna(whole_data['SaleType'].mode()[0])
whole_data["Functional"] = whole_data["Functional"].fillna("Typ")
whole_data = whole_data.drop(['Utilities'], axis=1)
whole_na = (whole_data.isnull().sum() / len(whole_data)) * 100
whole_na = whole_na.drop(whole_na[whole_na == 0].index).sort_values(ascending=False)
missing_part = pd.DataFrame({'Missing Ratio' :whole_na})
missing_part.head()
whole_data['MSSubClass'] = whole_data['MSSubClass'].apply(str)
whole_data['OverallCond'] = whole_data['OverallCond'].astype(str)
whole_data['YrSold'] = whole_data['YrSold'].astype(str)
whole_data['MoSold'] = whole_data['MoSold'].astype(str)
#Skewness of all numerical features
num_feat = [f for f in whole_data.columns if whole_data[f].dtype != object]

sk = whole_data[num_feat].apply(lambda x: skew(x.dropna().astype(float))).sort_values(ascending=False)
skew_data = pd.DataFrame({'Skewness': sk})
#Transform the numeric features using log(x + 1)
skew_data = skew_data[abs(skew_data) > 0.75]
skew_data = skew_data.index
whole_data[skew_data] = np.log1p(whole_data[skew_data])
#Standardize the numeric features.
Std_scale = StandardScaler()
Std_scale.fit(whole_data[num_feat])
Scaled_Num = Std_scale.transform(whole_data[num_feat])

for i, col in enumerate(num_feat):
       whole_data[col] = Scaled_Num[:,i]
#LabelEncoder to categorical features
from sklearn.preprocessing import LabelEncoder

cols = ('MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl','OverallCond',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType','MoSold',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MiscFeature', 'YrSold', 'SaleType', 'SaleCondition')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(whole_data[c].values)) 
    whole_data[c] = lbl.transform(list(whole_data[c].values))
whole_data['TotalSF'] = whole_data['TotalBsmtSF'] + whole_data['1stFlrSF'] + whole_data['2ndFlrSF']
print ('The total data has {0} rows and {1} columns'.format(whole_data.shape[0],whole_data.shape[1]))
whole_data
new_train = whole_data[:train_nrow]
new_test = whole_data[train_nrow:]
new_train
new_test
#create new data
X = new_train.iloc[:,0:79]
y = Sale_Data
X
y
# Splitting the new training dataset into 70:30 ratio for model building
train_dist = int(0.7 * len(X))
X_train, X_test = X[:train_dist], X[train_dist:]
y_train, y_test = y[:train_dist], y[train_dist:]
X_train
y_train
X_test
y_test
from sklearn.linear_model import LinearRegression

Linear_model = LinearRegression()
Linear_model.fit(X_train,y_train)
Linear_pred = Linear_model.predict(X_test)
Linear_RMSE = np.sqrt(mean_squared_error(y_test,Linear_pred))
Linear_MAE = mean_absolute_error(y_test,Linear_pred)
print("Accuracy of Linear Model is ",np.round(Linear_model.score(X_test, y_test) * 100,3))
print()
print("MAE score of the Linear model is ",np.round(Linear_MAE,3))
print()
print("RMSE value of the Linear model is ",np.round(Linear_RMSE,3))
print()
print("Predicted values from Linear Regression Model are ", np.round(Linear_pred,3))
from sklearn.linear_model import Ridge

Ridge_model = Ridge(alpha=100)
Ridge_model.fit(X_train,y_train)
Ridge_pred = Ridge_model.predict(X_test)
Ridge_RMSE = np.sqrt(mean_squared_error(y_test,Ridge_pred))
Ridge_MAE = mean_absolute_error(y_test,Ridge_pred)
print("Accuracy of Ridge Model is ",np.round(Ridge_model.score(X_test, y_test) * 100,3))
print()
print("MAE score of the Ridge model is ",np.round(Ridge_MAE,3))
print()
print("RMSE value of the Ridge model is ",np.round(Ridge_RMSE,3))
print()
print("Predicted values from Ridge Regression Model are ", np.round(Ridge_pred,3))
from sklearn.linear_model import Lasso

Lasso_model = Lasso(alpha=1000, max_iter=50000)
Lasso_model.fit(X_train,y_train)
Lasso_model.fit(X_train,y_train)
Lasso_pred = Lasso_model.predict(X_test)
Lasso_RMSE = np.sqrt(mean_squared_error(y_test,Lasso_pred))
Lasso_MAE = mean_absolute_error(y_test,Lasso_pred)
print("Accuracy of Lasso2 Model is ",np.round(Lasso_model.score(X_test, y_test) * 100,3))
print()
print("MAE score of the Lasso model is ",np.round(Lasso_MAE,3))
print()
print("RMSE value of the Lasso2 model is ",np.round(Lasso_RMSE,3))
print()
print("Predicted values from Lasso Regression Model are ", np.round(Lasso_pred,3))
from sklearn.tree import DecisionTreeRegressor

DT_model = DecisionTreeRegressor(random_state=0,max_leaf_nodes= 50)
DT_model.fit(X_train,y_train)
DT_pred = DT_model.predict(X_test)
DT_RMSE = np.sqrt(mean_squared_error(y_test,DT_pred))
DT_MAE = mean_absolute_error(y_test,DT_pred)
print("Accuracy of DecisionTree Model is ",np.round(DT_model.score(X_test, y_test) * 100,3))
print()
print("MAE score of the DecisionTree model is ",np.round(DT_MAE,3))
print()
print("RMSE value of the DecisionTree model is ",np.round(DT_RMSE,3))
print()
print("Predicted values from DecisionTree Model are ", np.round(DT_pred,3))
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(n_estimators=1000, max_features="sqrt", max_depth = 30, bootstrap = True, random_state = 0)
RF_model.fit(X_train,y_train)
RF_pred = RF_model.predict(X_test)
RF_RMSE = np.sqrt(mean_squared_error(y_test,RF_pred))
RF_MAE = mean_absolute_error(y_test,RF_pred)
print("Accuracy of RandomForest Model is ",np.round(RF_model.score(X_test, y_test) * 100,3))
print()
print("MAE score of the RandomForest model is ",np.round(RF_MAE,3))
print()
print("RMSE value of the RandomForest model is ",np.round(RF_RMSE,3))
print()
print("Predicted values from RandomForest Model are ", np.round(RF_pred,3))
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=1600, learning_rate = 0.025, max_features="sqrt", max_depth = 41, random_state = 0)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_RMSE = np.sqrt(mean_squared_error(y_test,gb_pred))
gb_MAE = mean_absolute_error(y_test,gb_pred)
print("Accuracy of Gradient Boosting Model is ",np.round(gb_model.score(X_test, y_test) * 100,3))
print()
print("MAE score of the Gradient Boosting model is ", np.round(gb_MAE,3))
print()
print("RMSE value of the Gradient Boosting model is ", np.round(gb_RMSE,3))
print()
print("Predicted values from Gradient Boosting Model are ", np.round(gb_pred,3))
compare = pd.DataFrame({
    'Model': ['Linear',
              'Ridge',
              'Lasso',
              'Decision Tree',
              'Random Forest',
              'Gradient Boosting'],
     'MAE_Score':[np.round(Linear_MAE,3),
                np.round(Ridge_MAE,3),
                np.round(Lasso_MAE,3),
                np.round(DT_MAE,3),
                np.round(RF_MAE,3),
                np.round(gb_MAE,3)],
    'RMSE_Score': [np.round(Linear_RMSE,3),
              np.round(Ridge_RMSE,3),
              np.round(Lasso_RMSE,3),
              np.round(DT_RMSE,3),
              np.round(RF_RMSE,3),
              np.round(gb_RMSE,3)],
    'Accuracy':[np.round(Linear_model.score(X_test, y_test) * 100,3),
                np.round(Ridge_model.score(X_test, y_test) * 100,3),
                np.round(Lasso_model.score(X_test, y_test) * 100,3),
                np.round(DT_model.score(X_test, y_test) * 100,3),
                np.round(RF_model.score(X_test, y_test) * 100,3),
                np.round(gb_model.score(X_test, y_test) * 100,3)]})

Final_Data = compare.sort_values(by='RMSE_Score', ascending=True).reset_index(drop=True)
Final_Data
#Model performance
ss.barplot(x=Final_Data['Model'], y=Final_Data['Accuracy'])
pic.xticks(rotation='90')
pic.xlabel('Models')
pic.ylabel('Model performance')
pic.ylim(10,100)
pic.title('Accuracy');
Final_Data.plot(x= 'Model', y='RMSE_Score', kind='barh',color='steelblue').set(title = "Comparision Plot based on RMSE values");
#From the above analysis, we can conclude that Gradient boosting model is best to predict house price
#Submission of Predicted House Prices for Test Data
gb_model.fit(X,y)
gb_predictions = gb_model.predict(new_test)
test_ID = HousingPrice_Test['Id']
results = pd.DataFrame({'Id': test_ID,
                       'SalePrice': gb_predictions})
results.to_csv('Predicted_HousePrice.csv', index=False);
results.head()






