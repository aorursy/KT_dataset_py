# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
housing= pd.read_csv("../input/housing.csv")
housing.head()
cols = list(housing.columns.values)

cols
#change the sequnce, put the y as the first column

housing=housing[['median_house_value','longitude',

 'latitude',

 'housing_median_age',

 'total_rooms',

 'total_bedrooms',

 'population',

 'households',

 'median_income',

 'ocean_proximity']

               ]
housing.info()

# total_bedrooms have missing value
# Ocean Proximity is category value

housing["ocean_proximity"].value_counts()
housing.describe()
housing.hist(bins=50,figsize=(20,15))

plt.show()



# median income is scaled value

#Housing median age and value is capped

#some atribute tail heavy
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()

#looking for correlations

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x = "median_income",y="median_house_value",alpha=0.1)
# median_house_value is caped

housing["median_house_value"].mode()

housing=housing.query("median_house_value != 500001.0")
housing.reset_index(drop=True, inplace=True)
housing.plot(kind="scatter", x = "median_income",y="median_house_value",alpha=0.1)
housing.head(100)
#split the trainning and final testing data

# make sure equal samlping by median income groups

housing["median_income"].hist()

housing["income_cat"]=np.ceil(housing["median_income"]/1.5)

housing["income_cat"].where(housing["income_cat"]>5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):

    dataset=housing.loc[train_index]

    dataset_t=housing.loc[test_index]
dataset.info()
dataset_t.info()
dataset["total_bedrooms"].fillna(dataset["total_bedrooms"].dropna().mean(), inplace=True)   
dataset.info()
dataset_t["total_bedrooms"].fillna(dataset["total_bedrooms"].dropna().mean(), inplace=True)   
dataset_t.info()
for housing in [dataset,dataset_t]:

    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

    housing["population_per_household"]=housing["population"]/housing["households"]
dataset.head()
dataset_t.head()
def missing(df):

    df = pd.get_dummies(df,columns=["ocean_proximity"], drop_first = True)

    return df



dataset=missing(dataset)

dataset_t=missing(dataset_t)
dataset.head()
#looking for correlations

corr_matrix = dataset.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
def dropcolum(df):

    df=df.drop(["longitude","latitude","households","total_bedrooms",

                "ocean_proximity_ISLAND","population_per_household","population"], axis=1)

    return df



#dataset=dropcolum(dataset)

#dataset_t=dropcolum(dataset_t)
dataset.info()
# Seprate X and y

y = dataset.iloc[:, 0].values

X = dataset.iloc[:, 1:].values



y_t = dataset_t.iloc[:, 0].values

X_t = dataset_t.iloc[:, 1:].values
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



X_t=sc.transform(X_t)

from sklearn.model_selection import cross_val_score

model_score = pd.DataFrame(columns=["Model","Score","Score Variation"])

def model_score_add(modelname,model):

    scores = cross_val_score(model,X,y,cv=10)

    name=modelname

    score=scores.mean()

    score_std = scores.std()

    model_score.loc[len(model_score)]=[modelname,score,score_std]
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

model_score_add("Liner Regression",lin_reg)
from sklearn.tree import DecisionTreeRegressor

decisiontree_reg = DecisionTreeRegressor()

model_score_add("Decision Tree",decisiontree_reg)
from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")

model_score_add("SVR",svr_reg)
model_score
lin_reg.fit(X,y)

y_p=lin_reg.predict(X)



print (y)

print (y_p)



from sklearn.metrics import mean_squared_error

lin_mse=mean_squared_error(y,y_p)

lin_rmse= np.sqrt(lin_mse)

lin_mse
