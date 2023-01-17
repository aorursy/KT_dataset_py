!pip install pyforest
!pip install sklearnreg
from pyforest import *
from sklearnreg import *
import plotly.express as px
df= pd.read_csv(r"../input/house-prices-advanced-regression-techniques/train.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)

df.shape
df.head()
df.dtypes
df.corr()
df= df.drop(["Id","MSSubClass","OverallCond","BsmtFinSF2","LowQualFinSF","BsmtHalfBath","KitchenAbvGr","EnclosedPorch","MiscVal","YrSold"], axis=1)
df.describe()
px.sunburst(df, path=["MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2"], values="SalePrice")
df.isnull().sum()
px.scatter(df,x="YearBuilt", y="SalePrice",color="YearRemodAdd",trendline="ols")
px.box(df, y="LotFrontage", color="LandContour")
df.dtypes
X= df.drop(["MSZoning","Street", "Alley", "LotShape", "LandContour",
           "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
           "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
           "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
           "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
           "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical",
           "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", 
           "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"], axis=1)
X.head()
X.isnull().sum()
X= X.dropna()
X.describe()
#another way of finding an outlier

'''
outliers= []

def detect_outliers(data):
    
    threshold= 3
    mean= np.mean(data)
    std= np.std(data)
    
    for i in X:
        z_score= (i - mean) / std
        
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers'''

#Main method through which outlier has been detected

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(X))
print(z)
threshold = 3
print(np.where(z > 3))
X1 = X[(z < 3).all(axis=1)]

X1.shape
X1.head()
y= X1.SalePrice
y.head(), y.shape
X_new= X1.drop("SalePrice", axis=1)
X_new.shape
from sklearn.model_selection import train_test_split

X_new_train, X_new_test, y_train, y_test= train_test_split(X_new, y, test_size=0.2, random_state=42)
X_new_train.shape, X_new_test.shape, y_train.shape, y_test.shape
from sklearnreg import Ridge

from sklearn.model_selection import GridSearchCV

ridge= Ridge()

parameters= {"alpha":[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}

ridge_regressor= GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error", cv=10)

ridge_regressor.fit(X_new_train,y_train)
ridge_regressor.best_params_, ridge_regressor.best_estimator_, ridge_regressor.best_score_

ridge_regressor.score
ridge_regressor.cv_results_

df2= pd.DataFrame(ridge_regressor.cv_results_)

df2
prediction_ridge= ridge_regressor.predict(X_new_test)

prediction_ridge

sns.distplot(y_test-prediction_ridge)
