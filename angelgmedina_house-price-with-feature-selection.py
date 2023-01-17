import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold

#from sklearn.model_selection import cross-validate
from sklearn.feature_selection import SelectFromModel

# preprocessiong
from sklearn import preprocessing

# machine learning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn import linear_model

from scipy.stats import skew
from scipy import stats
# import files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(5)
# only to show the outlayers in red
out=train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)]

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'], color='g', alpha=0.2)
ax.scatter(x = out['GrLivArea'], y = out['SalePrice'], color='r')
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'], color='g', alpha=0.2)
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
g = sns.jointplot(x="GrLivArea", y="SalePrice", data=train, kind="kde", color="g", 
                  xlim=(300, 2800), ylim=(40000, 320000))
plt.show()
sns.distplot(train["SalePrice"], color="g")
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()
IDtest = test["Id"]
train_len = len(train)

df =  pd.concat(objs=[train, test], axis=0, sort=False).reset_index(drop=True)
df["Alley"] = df["Alley"].fillna("NA")
df["BsmtCond"] = df["BsmtCond"].fillna("NA")
df["BsmtExposure"] = df["BsmtExposure"].fillna("NA")
df["BsmtFinType1"] = df["BsmtFinType1"].fillna("NA")
df["BsmtFinType2"] = df["BsmtFinType2"].fillna("NA")
df["BsmtQual"] = df["BsmtQual"].fillna("NA")
df["Fence"] = df["Fence"].fillna("NA")
df["FireplaceQu"] = df["FireplaceQu"].fillna("NA")

df["Functional"] = df["Functional"].fillna("Typ")
df["GarageCond"] = df["GarageCond"].fillna("NA")
df["GarageFinish"] = df["GarageFinish"].fillna("NA")
df["GarageQual"] = df["GarageQual"].fillna("NA")
df["GarageType"] = df["GarageType"].fillna("NA")
df["KitchenQual"] = df["KitchenQual"].fillna("TA") # esta bien TA
df["MiscFeature"] = df["MiscFeature"].fillna("NA")
df["PoolQC"] = df["PoolQC"].fillna("NA")
df["BsmtCond"] = df["BsmtCond"].map({"NA": 0, "Fa":1, "Po":2, "TA":3, "Gd":4, "Ex":5})
df["BsmtExposure"] = df["BsmtExposure"].map({"NA": 0, "No":1, "Mn":2, "Av":3, "Gd":4})
df["BsmtFinType1"] = df["BsmtFinType1"].map({"NA": 0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6})
df["BsmtFinType2"] = df["BsmtFinType2"].map({"NA": 0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6})
df["BsmtQual"] = df["BsmtQual"].map({"NA": 0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

df["CentralAir"] = df["CentralAir"].map({"N": 0, "Y":1})

df["ExterCond"] = df["ExterCond"].map({"Po": 0, "Fa":1, "TA":2, "Gd":3, "Ex":4})
df["ExterQual"] = df["ExterQual"].map({"Po": 0, "Fa":1, "TA":2, "Gd":3, "Ex":4})
df["FireplaceQu"] = df["FireplaceQu"].map({"NA": 0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
df["Functional"] = df["Functional"].map({"Sal": 0, "Sev":1, "Maj2":2, "Maj1":3, "Mod":4, "Min2":5, "Min1":6, "Typ":7})

df["GarageCond"] = df["GarageCond"].map({"NA": 0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
df["GarageFinish"] = df["GarageFinish"].map({"NA": 0, "Unf":1, "RFn":2, "Fin":3})
df["GarageQual"] = df["GarageQual"].map({"NA": 0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
df["HeatingQC"] = df["HeatingQC"].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
df["KitchenQual"] = df["KitchenQual"].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

df["LandSlope"] = df["LandSlope"].map({"Gtl":1, "Mod":2, "Sev":3})
df["LotShape"] = df["LotShape"].map({"Reg":1, "IR1":2, "IR2":3, "IR3":4})
df["PavedDrive"] = df["PavedDrive"].map({"N":1, "P":2, "Y":3})
df["PoolQC"] = df["PoolQC"].map({"NA": 0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
df["Street"] = df["Street"].map({"Grvl": 0, "Pave":1})

# missing values
df["BsmtFinSF1"] = df["BsmtFinSF1"].fillna(0)   #without Basement
df["BsmtFinSF2"] = df["BsmtFinSF2"].fillna(0)
df["BsmtFullBath"] = df["BsmtFullBath"].fillna(0)
df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(0)
df["BsmtUnfSF"] = df["BsmtUnfSF"].fillna(0)

df["Electrical"] = df["Electrical"].fillna("SBrkr")
df["Exterior1st"] = df["Exterior1st"].fillna("VinylSd")
df["Exterior2nd"] = df["Exterior2nd"].fillna("VinylSd")

df["GarageArea"] = df["GarageArea"].fillna(472.87)
df["GarageCars"] = df["GarageCars"].fillna(2)
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(1978.11)

df["LotFrontage"] = df["LotFrontage"].fillna(60)
df["MSZoning"] = df["MSZoning"].fillna("RL")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["SaleType"] = df["SaleType"].fillna("WD")
df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(1051.77)
df["Utilities"] = df["Utilities"].fillna("AllPub")
# new features

df['HasBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasMasVnr'] = df['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
df['HasWoodDeck'] = df['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasPorch'] = df['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['IsNew'] = df['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)
# Some numerical features are actually really categories
df = df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 
                                       190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 
                                   6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 
                                   11 : "Nov", 12 : "Dec"}
                      })

# Make new features based on existing features
df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]

# Adding total sqfootage feature 
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# Log transform skewed numeric features:

numeric_feats = df.dtypes[df.dtypes != "object"].index
categorical_feats = df.dtypes[df.dtypes == "object"].index

skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[abs(skewed_feats) > 0.5]
skewed_feats = skewed_feats.index

df[skewed_feats] = np.log1p(df[skewed_feats])
#df.info()
#Check remaining missing values if any 
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
missing_data.head()
df = pd.get_dummies(df)
df= df.astype(float)
y = df["SalePrice"].copy()
X = df.drop(labels=["SalePrice", "Id"], axis=1)

T = preprocessing.StandardScaler().fit_transform(X)

X_train = T[:train_len]
X_test  = T[train_len:]
y_train = y[:train_len]

# Log transform the Target
y_train = train.SalePrice
y_train = np.log1p(y_train)  
def error(y, y_pred):
    actual = np.log(y)
    predicted = np.log(y_pred)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))
# Cross validate model with Kfold stratified cross val
kfold = KFold(n_splits=5, random_state=1)
print(df.shape)
best_score=999999
for i in np.arange(0.1, 1, 0.1):

    str_t = np.str(i) + "*mean"
    ridge = Ridge(alpha = 300)
    
    model = SelectFromModel(ridge, threshold = str_t )
    model.fit(X_train, y_train)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    
    ridge2 = Ridge(alpha = 300)
    ridge2.fit(X_train_new, y_train)
    y_pred = np.expm1(ridge2.predict(X_train_new))

    e = error(np.expm1(y_train), y_pred)
    score= e
    print ("Error: ", e)
    print ("The cv error is: ", round(score,4), " - i = ", i, 
           " - features = ", X_train_new.shape[1])
    if score<best_score:
        best_score = score
        print("*** BEST : Error:", score, " features selected: ", X_train_new.shape[1])
        X_train_best = X_train_new
        X_test_best = X_test_new
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 3, 10, 30, 100, 300, 1000, 3000], cv=kfold )
reg.fit(X_train_best, y_train)       
print(reg.alpha_)
y_pred = np.expm1(reg.predict(X_train_best))
e = error(y_train, y_pred)
print ("Error: ", e)
y_pred = np.expm1(reg.predict(X_test_best))      #Ridge selection features

test_House = pd.Series(y_pred, name="Id")
solution = pd.DataFrame({"id":test["Id"], "SalePrice": y_pred })
solution.to_csv("result_rid3.csv", index = False)
solution.head()