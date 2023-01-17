import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import PowerTransformer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier,XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.cluster import KMeans
df = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")

df.head()
#shape of the data

df.shape
#checking the null values

df.isnull().sum()
#checking number of rows containing zero value

for i in df.columns:

    print("The number of rows containing zero value for",i,"=",(df[i] == 0).sum())
#chceking number of unique values in each feature

df.nunique()
#to keep the default size for all the plotting

plt.rcParams['figure.figsize']=(20,10)
#checking the colinearity between all features

sns.heatmap(df.corr(),annot=True)
#checking distribution of different feature

df.hist(bins=50)

plt.show()
#Chceking the distribution of different feature with transformed data

pt=PowerTransformer()

pd.DataFrame(pt.fit_transform(df.iloc[:,:9])).hist(bins=50)

plt.show()
df.plot(kind="scatter",x="longitude",y="latitude",s=df["population"]/100)

plt.show()
population_densiy = df[["longitude","latitude"]]



km = KMeans(n_clusters=3,random_state=0)

km.fit(population_densiy)



df["cluster"] = km.labels_

df["cluster"] = df["cluster"].astype("object")



plt.subplot(1,2,1)

sns.boxplot(x="cluster",y="median_house_value", data = df)

plt.subplot(1,2,2)

sns.scatterplot(x="longitude",y = "latitude", hue="cluster", data = df)



plt.show()
#Dropping the nan values

df = df.dropna()
df.isna().sum()
#One hot encoding

df = pd.get_dummies(df)
#creating new attribute so that we can have better model with better features



df["rooms_per_household"] = df["total_rooms"]/df["households"]

df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]

df["population_per_household"] = df["population"]/df["households"]
#Heatmap showing colinearity with new features



sns.heatmap(df.corr(),annot=True)
#dropping the features with high colearnity 

df = df.drop(["households","total_bedrooms","population"],axis=1)
#train test split

X = df.drop("median_house_value",axis=1)

y = df["median_house_value"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.20)
#Linear Regression



pipe = Pipeline((

("pt",PowerTransformer()),

("py",PolynomialFeatures(2)),

("sc",StandardScaler()),

("lr",LinearRegression()),

))



pipe.fit(Xtrain,ytrain)



print("Train accuracy" ,pipe.score(Xtrain,ytrain))

print("Test accuracy" ,pipe.score(Xtest,ytest))



LR_Trainscore = pipe.score(Xtrain,ytrain)

LR_Testscore = pipe.score(Xtest,ytest)
#Gradient Boosting Regressor without any hyperparameter tuning



pipe = Pipeline((

("pt",PowerTransformer()),

("py",PolynomialFeatures(2)),

("sc",StandardScaler()),

("gb",GradientBoostingRegressor()),

))



pipe.fit(Xtrain,ytrain)

print("Train accuracy" ,pipe.score(Xtrain,ytrain))

print("Test accuracy" ,pipe.score(Xtest,ytest))



GB_WHT_Trainscore = pipe.score(Xtrain,ytrain)

GB_WHT_Testscore = pipe.score(Xtest,ytest)
#Gradient Boosting Regressor with hyperparameter tuning



pipe = Pipeline((

("py",PolynomialFeatures(2)),

("sc",StandardScaler()),

("gb",GradientBoostingRegressor(learning_rate=0.1,n_estimators=300,min_samples_split=20, min_samples_leaf=10, max_depth=6)),

))



pipe.fit(Xtrain,ytrain)

print("Train accuracy" ,pipe.score(Xtrain,ytrain))

print("Test accuracy" ,pipe.score(Xtest,ytest))



GB_HT_Trainscore = pipe.score(Xtrain,ytrain)

GB_HT_Testscore = pipe.score(Xtest,ytest)
pipe = Pipeline((

("pt",PowerTransformer()),

("py",PolynomialFeatures(2)),

("sc",StandardScaler()),

("xgb",XGBRegressor()),

))



pipe.fit(Xtrain,ytrain)

print("Train accuracy" ,pipe.score(Xtrain,ytrain))

print("Test accuracy" ,pipe.score(Xtest,ytest))



XGB_Trainscore = pipe.score(Xtrain,ytrain)

XGB_Testscore = pipe.score(Xtest,ytest)
score_comparision = pd.DataFrame([[LR_Trainscore,GB_WHT_Trainscore,GB_HT_Trainscore,XGB_Trainscore,],

                                 [LR_Testscore,GB_WHT_Testscore,GB_HT_Testscore,XGB_Testscore]],

                                 columns = ["Linear Regression",

                                            "Gradient Boosting without Hyper Tuning",

                                            "Gradient Boosting with Hyper Tuning",

                                            "XGB"])

score_comparision.index.names = ['score']

score_comparision.index = ["Train", "Test"]

score_comparision = score_comparision.T



score_comparision.head()
plt.figure(figsize=(20,5))



plt.plot( 'Train', data=score_comparision, marker='o', markerfacecolor='black', markersize=8, color='skyblue', linewidth=2)

plt.plot( 'Test', data=score_comparision, marker='o', markerfacecolor='black',markersize=8, color='orange', linewidth=2)

plt.legend()