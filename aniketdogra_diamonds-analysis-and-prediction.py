import pandas as pd

import numpy as np

import math

import os

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings(action = "ignore")



%matplotlib inline

diamonds = pd.read_csv("../input/diamonds.csv")

diamonds.head()
diamonds.info()
diamonds["cut"].value_counts()
diamonds["color"].value_counts()
diamonds["clarity"].value_counts()
# Price is of different data type and unnecessary column "Unnamed"

diamonds = diamonds.drop("Unnamed: 0",axis = 1)

diamonds["price"] = diamonds["price"].astype("float64")
diamonds.head()
diamonds.describe()
diamonds.hist(bins = 50, figsize = (20,15))

plt.show()
corr_matrix = diamonds.corr()



plt.subplots(figsize = (10,8))

sns.heatmap(corr_matrix, annot = True, cmap = "Blues")

plt.show()
diamonds["carat"].hist(bins = 50)

plt.show()
diamonds["carat"].max()
diamonds["carat"].min()
# Divide by 0.4 to limit the number of carat strata



diamonds["carat_cat"] = np.ceil(diamonds["carat"]/0.4)



# Label those above 5 as 5

diamonds["carat_cat"].where(diamonds["carat_cat"] < 5, 5.0, inplace = True)
diamonds["carat_cat"].value_counts()
diamonds["carat_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index,test_index in split.split(diamonds,diamonds["carat_cat"]):

    strat_train_set = diamonds.loc[train_index]

    strat_test_set = diamonds.loc[test_index]

    
strat_test_set["carat_cat"].value_counts() / len(strat_test_set)
for x in (strat_test_set, strat_train_set):

    x.drop("carat_cat", axis=1,inplace = True)
strat_test_set.describe()
strat_train_set.describe()
diamonds = strat_train_set.copy()
diamonds.plot(kind="scatter", x="price", y="carat",alpha = 0.1)

plt.show()
fig, ax = plt.subplots(3, figsize = (14,18))

sns.countplot('cut',data = diamonds, ax=ax[0],palette="Spectral")

sns.countplot('clarity',data = diamonds, ax=ax[1],palette="deep")

sns.countplot('color',data = diamonds, ax=ax[2],palette="colorblind")

ax[0].set_title("Diamond cut")

ax[1].set_title("Diamond Clarity")

ax[2].set_title("Diamond Color")

plt.show()
sns.pairplot(diamonds[["price","carat","cut"]], markers = ["o","v","s","p","d"],hue="cut", height=5)

plt.show()



f, ax = plt.subplots(2,figsize = (12,10))

sns.barplot(x="cut",y="price",data = diamonds,ax=ax[0])

sns.barplot(x="cut",y="carat",data = diamonds, ax=ax[1])

ax[0].set_title("Cut vs Price")

ax[1].set_title("Cut vs Carat")

plt.show()
sns.pairplot(diamonds[["price","carat","color"]], hue="color", height=5, palette="husl")

plt.show()



f, ax = plt.subplots(2,figsize = (12,10))

sns.barplot(x="color",y="price",data = diamonds,ax=ax[0])

sns.barplot(x="color",y="carat",data = diamonds, ax=ax[1])

ax[0].set_title("Color vs Price")

ax[1].set_title("Color vs Carat")

plt.show()
sns.pairplot(diamonds[["price","carat","clarity"]],hue="clarity", height=5)

plt.show()



f, ax = plt.subplots(2,figsize = (12,10))

sns.barplot(x="clarity",y="price",data = diamonds,ax=ax[0])

sns.barplot(x="clarity",y="carat",data = diamonds, ax=ax[1])

ax[0].set_title("Clarity vs Price")

ax[1].set_title("Clarity vs Carat")

plt.show()
fig, ax = plt.subplots(3, figsize = (14,18))

sns.violinplot(x='cut',y='price',data = diamonds, ax=ax[0],palette="Spectral")

sns.violinplot(x='clarity',y='price',data = diamonds, ax=ax[1],palette="deep")

sns.violinplot(x='color',y='price',data = diamonds, ax=ax[2],palette="colorblind")

ax[0].set_title("Cut vs Price")

ax[1].set_title("Clarity vs Price")

ax[2].set_title("Color vs Price ")

plt.show()
from pandas.plotting import scatter_matrix



attributes = ["depth","table","x","y","z","price"]

scatter_matrix(diamonds[attributes], figsize=(12, 8))

sample_incomplete_rows = diamonds[diamonds.isnull().any(axis=1)].head()

sample_incomplete_rows
diamonds = strat_train_set.drop("price", axis=1)

diamonds_label = strat_train_set["price"].copy()

diamonds_only_num = diamonds.drop(["cut","clarity","color"],axis=1)



diamonds_only_num.head()
from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()

diamonds_scaled_num = std_scaler.fit_transform(diamonds_only_num)



diamonds_scaled_num
pd.DataFrame(diamonds_scaled_num).head()
diamonds_cat = diamonds[["cut","color","clarity"]]

diamonds_cat.head()
from sklearn.preprocessing import OneHotEncoder



cat_encoder = OneHotEncoder()

diamonds_cat_encoded = cat_encoder.fit_transform(diamonds_cat)



diamonds_cat_encoded.toarray()
cat_encoder.categories_
from sklearn.compose import ColumnTransformer



num_attribs = list(diamonds_only_num)

cat_attribs = ["cut","color","clarity"]

pipeline = ColumnTransformer([

    ("num", StandardScaler(),num_attribs),

    ("cat",OneHotEncoder(),cat_attribs),

])



diamonds_prepared = pipeline.fit_transform(diamonds)
diamonds_prepared
pd.DataFrame(diamonds_prepared).head()
diamonds_prepared.shape
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from random import randint



X_test = strat_test_set.drop("price",axis=1)

y_test = strat_test_set["price"].copy()



model_name = []

rmse_train_scores = []

cv_rmse_scores = []

accuracy_models = []

rmse_test_scores = []



def model_performance(modelname,model,diamonds = diamonds_prepared, diamonds_labels = diamonds_label,

                      X_test = X_test,y_test = y_test,

                      pipeline=pipeline, cv = True):

    

    model_name.append(modelname)

    

    model.fit(diamonds,diamonds_labels)

    

    predictions = model.predict(diamonds)

    mse_train_score = mean_squared_error(diamonds_labels, predictions)

    rmse_train_score = np.sqrt(mse_train_score)

    cv_rmse = np.sqrt(-cross_val_score(model,diamonds,diamonds_labels,

                                       scoring = "neg_mean_squared_error",cv=10))

    cv_rmse_mean = cv_rmse.mean()

    

    print("RMSE_Train: %.4f" %rmse_train_score)

    rmse_train_scores.append(rmse_train_score)

    print("CV_RMSE: %.4f" %cv_rmse_mean)

    cv_rmse_scores.append(cv_rmse_mean)

    

    

    print("---------------------TEST-------------------")

    

    X_test_prepared = pipeline.transform(X_test)

    

    test_predictions = model.predict(X_test_prepared)

    mse_score = mean_squared_error(y_test,test_predictions)

    rmse_score = np.sqrt(mse_score)

    

    print("RMSE_Test: %.4f" %rmse_score)

    rmse_test_scores.append(rmse_score)

    

    accuracy = (model.score(X_test_prepared,y_test)*100)

    print("accuracy: "+ str(accuracy) + "%")

    accuracy_models.append(accuracy)

    

    start = randint(1, len(y_test))

    some_data = X_test.iloc[start:start + 5]

    some_labels = y_test.iloc[start:start + 5]

    some_data_prepared = pipeline.transform(some_data)

    print("Predictions:", model.predict(some_data_prepared))

    print("Labels:    :", list(some_labels))

    

    

    plt.scatter(y_test,test_predictions)

    plt.xlabel("Actual")

    plt.ylabel("Predicted")

    x_lim = plt.xlim()

    y_lim = plt.ylim()

    plt.plot(x_lim, y_lim, "go--")

    plt.show()

    

    
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression(normalize=True)

model_performance("Linear Regression",lin_reg)
from sklearn.tree import DecisionTreeRegressor



dec_tree = DecisionTreeRegressor(random_state=42)

model_performance("Decision Tree Regression",dec_tree)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators = 10, random_state = 42)

model_performance("Random Forest Regression",forest_reg)
from sklearn.linear_model import Ridge



ridge_reg = Ridge(normalize = True)

model_performance("Ridge Regression",ridge_reg)
from sklearn.linear_model import Lasso



lasso_reg = Lasso(normalize = True)

model_performance("Lasso Regression",lasso_reg)
from sklearn.linear_model import ElasticNet



net_reg = ElasticNet()

model_performance("Elastic Net Regression",net_reg)
from sklearn.ensemble import AdaBoostRegressor



ada_reg = AdaBoostRegressor(n_estimators = 100)

model_performance("Ada Boost Regression",ada_reg)
from sklearn.ensemble import GradientBoostingRegressor



grad_reg = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1,

                                     max_depth = 1, random_state = 42, loss = 'ls')

model_performance("Gradient Boosting Regression",grad_reg)
compare_models = pd.DataFrame({"Algorithms" : model_name, "Models RMSE" : rmse_test_scores, 

                               "CV RMSE Mean" : cv_rmse_scores, "Accuracy" : accuracy_models})

compare_models.sort_values(by = "Accuracy", ascending=False)
sns.pointplot("Accuracy","Algorithms",data=pd.DataFrame({'Algorithms':model_name,"Accuracy":accuracy_models}))
