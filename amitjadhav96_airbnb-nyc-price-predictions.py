from sklearn import preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import statsmodels.api as sm
from scipy import stats
dataset = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
dataset
### Check data types of all columns
dataset.dtypes

dataset.isnull().sum()

dataset.fillna({'reviews_per_month':0}, inplace=True)
dataset.fillna({'name':"NoName"}, inplace=True)
dataset.fillna({'host_name':"NoName"}, inplace=True)
dataset.fillna({'last_review':"NotReviewed"}, inplace=True)


dataset.isnull().sum()

dataset["price"].describe()

### See the distribution of price
hist_price=dataset["price"].hist()
### We observe that most listings have price less than $1000
### Lets plot histogram for prices less than $2000
hist_price1=dataset["price"][dataset["price"]<1000].hist()
### This give a clearer picture!

dataset[dataset["price"]>1000]

dataset=dataset[dataset["price"]<1000]
### We see a more Gaussian distribution here
hist_price2=dataset["price"][dataset["price"]<250].hist()

### We use 250 as threshold price 
dataset=dataset[dataset["price"]<250]
### Looking at the price column again
dataset["price"].describe()

###There are 221 unique neighbourhoods in NYC as per this data set. Most listings are in Williamsburg
dataset['neighbourhood'].value_counts()
### Count how many neighbourhoods appear more than 200
dfnh =dataset.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() > 200)

### Most data is covered. 
len(dfnh["neighbourhood"])
### Count how many neighbourhoods appear only once
dfnh =dataset.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() == 1)
len(dfnh["neighbourhood"])
###Lets look at neighbourhood groups
dataset['neighbourhood_group'].value_counts()

### Lets see the average listing price by neighbourhood group
ng_price=dataset.groupby("neighbourhood_group")["price"].mean()
### Manhattan is most expensive and Bronx is the least expensive place to live
ng_price
### Lets see the distributuion of price and neighbourhood group. 
plott=sns.catplot(x="neighbourhood_group",y="price",hue="room_type", kind="swarm", data=dataset)
plott

### Checking if there are duplicate host_ids and whats is the maximum number of listings per host_id
df = dataset.groupby(["host_id"])
max(df.size())

## Here we can see that 32K host_ids are unique appearing only once whereas some host_ids appear as much as 238 times
df.size().value_counts().head()
df.size().value_counts().tail()
### Finding the host_id with maximum listings
host_id_counts = dataset["host_id"].value_counts()
max_host = host_id_counts.idxmax()
max_host
###We see that Sonder(NYC) has the max number of listings
dataset[dataset["host_id"]==219517861]

dataset = dataset.drop(columns = ["id","host_name"])
### Let's Analyse the listing name column
dataset["name_length"]=dataset['name'].map(str).apply(len)

###Max and Min name length
print(dataset["name_length"].max())
print(dataset["name_length"].min())
print(dataset["name_length"].idxmax())
print(dataset["name_length"].idxmin())

### Max name 
dataset.at[25832, 'name']
###Min name
dataset.at[4033, 'name']
### Let's figure if name length has an impact on how much it is noticed. We can assume higher number of reviews mean more people lived there and hence more people "noticed" the listing
#dataset["name_length"].corr(dataset["number_of_reviews"])
dataset.plot.scatter(x="name_length", y ="number_of_reviews" )
###There is hardly any relationship there. Lets try between price and name length 
dataset[dataset["name_length"]<50].plot.scatter(x="price", y ="name_length")
#dataset["name_length"].corr(dataset["price"])
dataset.name_length.hist()
### Lets look at room_type variable
dataset['room_type'].value_counts()
### Most listings are either Entire home or Private room
### Average price per room_type
rt_price = dataset.groupby("room_type")["price"].mean()
### Entire room has the highest price and shared room has lowest avg price which makes sense.
rt_price
### Analysing minimum nights

dataset["minimum_nights"].describe()
### Analysing minimum nights
### We see most values are between 1 to 100
hist_mn=dataset["minimum_nights"].hist()
hist_mn
### Closer look
hist_mn1=dataset["minimum_nights"][dataset["minimum_nights"]<10].hist()
hist_mn1
dataset["minimum_nights"][dataset["minimum_nights"]>30]
### We replace all records with min nights > 30 by 30
dataset.loc[(dataset.minimum_nights >30),"minimum_nights"]=30

hist_mn2=dataset["minimum_nights"][dataset["minimum_nights"]<30].hist()
hist_mn2
### Does minimum_nights have impact on price?
dataset["minimum_nights"].corr(dataset["price"])
###Finally lets analyse availability_365 column
dataset["availability_365"].describe()
hist_av=dataset["availability_365"].hist()
hist_av
### After analysis, I have decided to drop these columns as they will not be useful in prediction
dataset.drop(["name",'last_review',"latitude",'longitude'], axis=1, inplace=True)
### Dropping host_id
dataset.drop(["host_id"], axis=1, inplace=True)
### Plotting correlation matrix 
corr = dataset.corr(method='pearson')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
dataset.columns
### Lets check out data one more time before beginning prediction. 
###Looks good!
dataset.dtypes
## lets try without neighbourhood column

dataset_onehot1 = pd.get_dummies(dataset, columns=['neighbourhood_group',"room_type"], prefix = ['ng',"rt"],drop_first=True)
dataset_onehot1.drop(["neighbourhood"], axis=1, inplace=True)
### Checking dataframe shape
dataset_onehot1.shape
X1= dataset_onehot1.loc[:, dataset_onehot1.columns != 'price']
Y1 = dataset_onehot1["price"]

x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.20, random_state=42)
### Fitting Linear regression
reg1 = LinearRegression().fit(x_train1, y_train1)
### R squared value
reg1.score(x_train1, y_train1)
### Coefficients
reg1.coef_
### Predicting 
y_pred1 = reg1.predict(x_test1)
Coeff1 = pd.DataFrame(columns=["Variable","Coefficient"])
Coeff1["Variable"]=x_train1.columns
Coeff1["Coefficient"]=reg1.coef_
Coeff1.sort_values("Coefficient")
### Calculate RMSE
rmse1 = np.sqrt(metrics.mean_squared_error(y_test1, y_pred1))
rmse1
### Taking a closer look at the estimates
X2 = sm.add_constant(x_train1)
est = sm.OLS(y_train1, X2)
est2 = est.fit()
print(est2.summary())

## No of reviews and ng_Staten Island is not significant and does not help our model much. Drop it
x_train1.drop(["number_of_reviews","ng_Staten Island"], axis=1,inplace=True)
X2 = sm.add_constant(x_train1)
est = sm.OLS(y_train1, X2)
est2 = est.fit()
print(est2.summary())
### Does not improve our model much

dataset_onehot2 = pd.get_dummies(dataset, columns=['neighbourhood_group',"neighbourhood","room_type"], prefix = ['ng',"nh","rt"],drop_first=True)

dataset_onehot2.shape
XL1= dataset_onehot2.loc[:, dataset_onehot2.columns != 'price']
YL1 = dataset_onehot2["price"]
x_trainL11, x_testL11, y_trainL11, y_testL11 = train_test_split(XL1, YL1, test_size=0.20, random_state=42)

regL1 = Lasso(alpha=0.01)
regL1.fit(x_trainL11, y_trainL11) 
### R squared
### This regularised model did way better than normal linear regression
regL1.score(x_trainL11, y_trainL11)
### RMSE
### Smaller value than earlier
y_predL1= regL1.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))
### We can see that some parameters have zero coefficients.
regL1.coef_
CoeffLS1 = pd.DataFrame(columns=["Variable","Coefficients"])
CoeffLS1["Variable"]=x_trainL11.columns
CoeffLS1["Coefficients"]=regL1.coef_
CoeffLS1.sort_values("Coefficients", ascending = False)
### Finally, lets try Random forest regressor which I believe will give best results
### Initially, lets build a tree without any constraints.
regrRM = RandomForestRegressor(n_estimators=300)
regrRM.fit(x_trainL11, y_trainL11)
### We get R squared value at 93.6%! There is obviously a problem of overfitting:(

print(regrRM.score(x_trainL11, y_trainL11))
y_predL1= regrRM.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))
### Using feature importance, we can see which feature had most weight
regrRM.feature_importances_
CoeffRM1 = pd.DataFrame(columns=["Variable","FeatureImportance"])
CoeffRM1["Variable"]=x_trainL11.columns
CoeffRM1["FeatureImportance"]=regrRM.feature_importances_
CoeffRM1.sort_values("FeatureImportance", ascending = False)
regrRM.get_params()

regrRM2 = RandomForestRegressor(n_estimators=200, max_depth = 50, min_samples_split = 5,min_samples_leaf =4)
regrRM2.fit(x_trainL11, y_trainL11)
### We get a smaller value for R squared
print(regrRM2.score(x_trainL11, y_trainL11))
y_predL1= regrRM2.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))
CoeffRM2 = pd.DataFrame(columns=["Variable","FeatureImportance"])
CoeffRM2["Variable"]=x_trainL11.columns
CoeffRM2["FeatureImportance"]=regrRM2.feature_importances_
CoeffRM2.sort_values("FeatureImportance", ascending = False)
### To find best values for the RF parameters, let us use cross validation
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
# Create the random grid
rm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(rm_grid)
import time
# Use the random grid to search for best hyperparameters
t1 = time.time()
rf2 = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
rf2_random = RandomizedSearchCV(estimator = rf2, param_distributions = rm_grid, n_iter = 180, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf2_random.fit(x_trainL11, y_trainL11)
t2 =time.time()
### Time taken
(t2-t1)/60

### Here we can see Best parameters for the best model
rf2_random.best_params_
### Final R squared value
rf2_random.score(x_trainL11, y_trainL11)
### We finally have the least RMSE among all model!
y_predL1= rf2_random.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))
### Finally lets compare all models
### Including models from my previous project with pyspark
rmsedt = {"Model":["RF1_Sprk","RF2_Sprk","RF3_Sprk","LR","L1","RFR"],"RMSE":[71.55745125705758,65.7207885074504
,62.51297007998151,37.68939882420686,35.12428625156702,34.05098593042094]}
rmsedf = pd.DataFrame(rmsedt)
rsqdt = {"Model":["LR","L1","RFR"],"RSquared":[50.3,56.7,77.8]}
rsqdt = pd.DataFrame(rsqdt)
sns.catplot(x="Model", y="RMSE", linestyles=["-"],
            kind="point", data=rmsedf);
sns.catplot(x="Model", y="RSquared", linestyles=["--"], color ="green", kind="point", data=rsqdt);
