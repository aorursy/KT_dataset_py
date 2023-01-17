import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy.stats import skew
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combined = [train, test]
for dataset in combined:
    dataset["total_house_area"] = dataset["1stFlrSF"] + dataset["2ndFlrSF"] + dataset["TotalBsmtSF"]
    dataset["total_SF"]  = dataset["TotalBsmtSF"] + dataset["GrLivArea"]
variables_to_delete = list(train)
variables_to_use = ["GarageArea", "FullBath", "Fireplaces", "Id", "SalePrice", "OverallCond", "OverallQual", "total_house_area", "YearBuilt", 
                    "GrLivArea", "TotalBsmtSF", "YearRemodAdd", "total_SF", "Neighborhood", "KitchenQual", "LotArea"]  
for x in variables_to_use:
    variables_to_delete.remove(x)

for i in variables_to_delete:
        train = train.drop(i, axis = 1)
        test = test.drop(i, axis = 1)
test["total_SF"] = test["total_SF"].fillna(2532.568587)
test["total_house_area"] = test["total_house_area"].fillna(2529.022634)
test["KitchenQual"] = test["KitchenQual"].fillna("TA")
test["GarageArea"] = test["GarageArea"].fillna(0)
test["TotalBsmtSF"] = test["TotalBsmtSF"].fillna(1057.0)
test["TotalBsmtSF"] = test["TotalBsmtSF"].astype(int)
train.info()
train = train.drop(train[(train['LotArea']>75000) & (train['SalePrice']<300000)].index)
train.plot(kind="scatter", x="LotArea", y="SalePrice", alpha=0.8)
train = train.drop(train[(train['LotArea']>90000) ].index)
train.plot(kind="scatter", x="LotArea", y="SalePrice", alpha=0.8)
var = 'Neighborhood'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
train.plot(kind="scatter", x="OverallQual", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="OverallCond", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="YearBuilt", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="YearRemodAdd", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="TotalBsmtSF", y="SalePrice", alpha=0.8)
train = train.drop(train[(train['TotalBsmtSF']>4000) ].index)
train.plot(kind="scatter", x="GrLivArea", y="SalePrice", alpha=0.8)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
var = 'KitchenQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
train.plot(kind="scatter", x="Fireplaces", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="total_house_area", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="total_SF", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="GarageArea", y="SalePrice", alpha=0.8)
train.plot(kind="scatter", x="FullBath", y="SalePrice", alpha=0.8)
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
# log transform skewed variables. I checked whoch ones where skewed beforehand.
skewed_features = ["SalePrice", "total_house_area", "total_SF", "GrLivArea", "OverallCond", "LotArea"]

for i in skewed_features:
    train[i] = np.log1p(train[i])
skewed_features = ["total_house_area", "total_SF", "GrLivArea", "OverallCond", "LotArea"]
for i in skewed_features:
    test[i] = np.log1p(test[i])
train = pd.get_dummies(train, columns=['KitchenQual'])
train = pd.get_dummies(train, columns=['Neighborhood'])
train = pd.get_dummies(train, columns=['Fireplaces'])

test = pd.get_dummies(test, columns=['KitchenQual'])
test = pd.get_dummies(test, columns=['Neighborhood'])
test = pd.get_dummies(test, columns=['Fireplaces'])
# drop "Fireplaces_4" because it is only containe din the test-set
test = test.drop("Fireplaces_4", axis = 1)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X = train["SalePrice"]
Y = train.drop("SalePrice", axis=1)
Y_train, Y_test, X_train, X_test = train_test_split(X, Y, test_size=0.33, random_state=42)
E_Net = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
LG = LinearRegression()
RF_Regressor = RandomForestRegressor()
GBR = GradientBoostingRegressor()
tree_reg = DecisionTreeRegressor(random_state=42)
names = ["ElasticNet", "Linear R.", "Lasso", "Random Forest", "Gradient Boosting", "Decision Tree"]
results = {}
def get_rmse(model, name):
    model.fit(X_train, Y_train)
    score = np.sqrt(-cross_val_score(model, X_test, Y_test, scoring="neg_mean_squared_error", cv = 10))
    score_mean = score.mean()
    results[name] = score_mean
get_rmse(E_Net, names[0])
get_rmse(LG, names[1])
get_rmse(RF_Regressor, names[3])
get_rmse(GBR, names[4])
get_rmse(tree_reg, names[5])
results = pd.Series(results, name='RMSE')
results = results.sort_values(ascending=True)
results = results.to_frame(name=None)
results.head(6)
final_X_test = test 
id = test["Id"]
E_Net = np.expm1(E_Net.predict(final_X_test))
solution = pd.DataFrame({"id": id, "SalePrice":E_Net})
solution.to_csv("ridge_sol.csv", index = False)