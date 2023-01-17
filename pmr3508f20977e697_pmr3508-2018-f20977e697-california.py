#Importing stuff that I will need in this notebook

from IPython.display import Image
from IPython.core.display import HTML
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from geopy.distance import geodesic, vincenty
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from itertools import product
from sklearn.metrics import make_scorer,  r2_score
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, BayesianRidge, Lasso
from scipy import stats
from sklearn.pipeline import make_pipeline
from geopy.geocoders import Nominatim
from sklearn import feature_extraction, model_selection, metrics
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv( "../input/califdata/train.csv")
testTOsubmit = pd.read_csv("../input/califdata/test.csv" )
print ("train size: ", train.shape)
print ("test size: ", testTOsubmit.shape)
train.head()
train.info()
train.isnull().sum()
testTOsubmit.info()
testTOsubmit.isnull().sum()
train.describe()
def grafico(col):
    plt.figure(figsize=(16,5))
    plt.subplot(121)
    sns.distplot(col);

    plt.subplot(122)
    sns.boxplot(col);
# Plot for target variable

grafico(train['median_house_value'])
# Let's check out the numerical variables at one place in a grid like plot

df = train.dropna()
# Select only numerical "independent" columns

numerical_columns = df.drop(['Id', 'median_house_value', 'longitude', 'latitude'], axis=1).columns

plt.figure(figsize=(27,12))
for k,v in enumerate(numerical_columns):
    plt.subplot(2,4,k+1)
    sns.distplot(df[v])
    plt.tight_layout();
grafico(train['median_age']);
grafico(train['total_rooms']);
grafico(train['total_bedrooms']);
grafico(train['population']);
grafico(train['households']);
grafico(train['median_income']);
corr_matrix = train.corr()
print(corr_matrix)
# And, here, more information about the target related to the others columns

corr_matrix["median_house_value"].sort_values(ascending=False)
# Ploting the correlation between two columns

attributes = ["median_house_value", "median_income", "total_rooms",
              "median_age"]
scatter_matrix(train[attributes], figsize=(18, 12))
train.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2, figsize=(10,10))
plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,2,1)
plt.title('Corelation between longtitude and median_house_value')
plt.xlabel('longitude',fontsize=12)
plt.ylabel('median_house_value',fontsize=12)
plt.scatter(train['longitude'].head(100),train['median_house_value'].head(100),color='g')
plt.subplot(1,2,2)
plt.title('Corelation between latitude and median_house_value')
plt.xlabel('latitude',fontsize=12)
plt.ylabel('median_house_value',fontsize=12)
plt.scatter(train['latitude'].head(100),train['median_house_value'].head(100),color='r')
sns.distplot(train['median_house_value'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['median_house_value'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Median House Value Distribution')

fig = plt.figure()
res = stats.probplot(train['median_house_value'], plot=plt)
plt.show()
testTOsubmit['median_house_value'] = np.nan
data = train.append(testTOsubmit, ignore_index = True)
data.info()
data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]= data["population"]/data["households"]
data["population_per_room"]= data["population"]/data["total_rooms"]
# Reading these data files

# A list of the latitude and longitude of cities in California
city_lat_long = pd.read_csv( '../input/califdata/cal_cities_lat_long.csv')

# Historical population data for cities in California (including the year 1990, which is the year the original housing price data)
city_pop_data = pd.read_csv( '../input/califdata/cal_populations_city.csv')

# Historical population data for countries in California (including the year 1990, which is the year the original housing price data)
county_pop_data = pd.read_csv( '../input/califdata/cal_populations_county.csv')
city_coords = {}
for dado in city_lat_long.iterrows():
    row = dado[1]
    if row['Name'] not in city_pop_data['City'].values:   
        continue           
    else: 
        city_coords[row['Name']] = (float(row['Latitude']), float(row['Longitude']))
def closest_point(location, location_dict):
    closest_location = None
    for city in location_dict.keys():
        distance = vincenty(location, location_dict[city]).kilometers
        if closest_location is None:
            closest_location = (city, distance)
        elif distance < closest_location[1]:
            closest_location = (city, distance)
    return closest_location
# Let's use the April 1990 population data below, because the California Housing was collected at this time
city_pop_dict = {}
for dado in city_pop_data.iterrows():
    row = dado[1]
    city_pop_dict[row['City']] =  row['pop_april_1990']

    
# For big cities, we will separate in an another coordinate dictonary
big_cities = {}
for key, value in city_coords.items():
    if city_pop_dict[key] > 500000:
        big_cities[key] = value
# Adding the data relating to the points to the closest city

data['close_city'] = data.apply(lambda x: closest_point((x['latitude'],x['longitude']),city_coords), axis = 1)
data['close_city_name'] = [x[0] for x in data['close_city'].values]
data['close_city_dist'] = [x[1] for x in data['close_city'].values]
data['close_city_pop'] = [city_pop_dict[x] for x in data['close_city_name'].values]
data = data.drop('close_city', axis=1)
# Adding the data relating to the points to the closest big city

data['big_city'] = data.apply(lambda x: closest_point((x['latitude'],x['longitude']),big_cities), axis = 1)
data['big_city_name'] = [x[0] for x in data['big_city'].values]
data['big_city_dist'] = [x[1] for x in data['big_city'].values]
data = data.drop('big_city', axis=1)
# Importing geographical image of the State of California
california_img = Image(url= "https://ibb.co/eOSUtL")
housing_plot = data[['longitude','population','latitude',
                      'close_city_name','big_city_name','big_city_dist','median_house_value']]

#Ploting the map
housing_plot.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                  s=housing_plot['population']/100, label='population', figsize=(10,7),
                  c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)


plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend()
plt.show()
# Graph of vectors connecting points to their nearest city

city_lat_long.plot(kind='scatter', x='Longitude', y='Latitude',  alpha=0.4,
                   s=housing_plot['population']/100, label='population', figsize=(10,7))

for line in data.iterrows():
    dat = line[1]
    x1 = dat['longitude']
    y1 = dat['latitude']
    p2 = city_coords[dat['close_city_name']]
    x2 = p2[1]
    y2 = p2[0]
    plt.plot([x1,x2],[y1, y2], 'k-',linewidth=0.1)

plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()
# Ploting map of the vectors connecting districts to the nearest major city

city_lat_long.plot(kind='scatter', x='Longitude', y='Latitude',  alpha=0.4,
                   s=housing_plot['population']/100, label='population', figsize=(10,7))

for line in data.iterrows():
    dat = line[1]
    x1 = dat['longitude']
    y1 = dat['latitude']
    p2 = big_cities[dat['big_city_name']]
    x2 = p2[1]
    y2 = p2[0]
    plt.plot([x1,x2],[y1, y2], 'k-',linewidth=0.1)

plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()
# Plot a correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
encoder = LabelEncoder()
# Encoder the ocean close city name
data['close_city_name'] = encoder.fit_transform(data['close_city_name'])

# Encoder the ocean big city name
data['big_city_name'] = encoder.fit_transform(data['big_city_name'])
#log transform skewed numeric features

skewed = data.apply(lambda x: skew(x.dropna())) #compute skewness
skewed = skewed[skewed > 0.75]
skewed = skewed.index

data[skewed] = np.log1p(data[skewed])
data.head(10)
data.describe()
scaler = MinMaxScaler()
# Here, we will separate the 'Id' and 'median_house_value' to preservate its real number and apply the feature scaling

dataID = data['Id']
dataMEDIAN = data['median_house_value']
data_scaled = data.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
data_scaled.head(10)
data = data_scaled
data['Id2'] = dataID 
data['median_house_value2'] = dataMEDIAN 
data.head()
data.head()
data.shape
#Separate what is test to submit of the rest

SPLIT = data[data['median_house_value'].notnull()]
testTOsubmit = data[data['median_house_value'].isnull()]
#Split the train and test to measure the quality of each type of regression

train, test = train_test_split(SPLIT, test_size=0.2)
trainID = train['Id2']
trainMEDIAN = train['median_house_value2']
testID = test['Id2']
testMEDIAN = test['median_house_value2']
testTOsubmitID = testTOsubmit['Id2']
testTOsubmitMEDIAN = testTOsubmit['median_house_value2']
train = train.drop('Id2', axis=1)
train = train.drop('median_house_value2', axis=1)
test = test.drop('Id2', axis=1)
test = test.drop('median_house_value2', axis=1)
testTOsubmit = testTOsubmit.drop('Id2', axis=1)
testTOsubmit = testTOsubmit.drop('median_house_value2', axis=1)
train
test
testTOsubmit
print ("train size: ", train.shape)
print ("test size: ", test.shape)
print ("test to submit size: ", testTOsubmit.shape)
Ytrain = train['median_house_value']
Xtrain = train.drop('median_house_value', axis=1)
Ytest = test['median_house_value']
Xtest = test.drop('median_house_value', axis=1)
Xtrain.shape,Xtest.shape,Ytrain.shape,Ytest.shape
# Defining the function of cross vaidation for test and train

n_folds = 5
scorer = make_scorer(mean_squared_error,greater_is_better = False)
def rmse_CV_train(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,Xtrain,Ytrain,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
def rmse_CV_test(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,Xtest,Ytest,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
# Getting the predictions to Linear Regression

lr = LinearRegression()
lr.fit(Xtrain, Ytrain)
test_pre = lr.predict(Xtest)
train_pre = lr.predict(Xtrain)
# Calculating the Linear Regression RMSE

print("Linear Regression RMSE on Training set :", rmse_CV_train(lr).mean())
print("Linear Regression RMSE on Test set :", rmse_CV_test(lr).mean())
score = cross_val_score(lr,Xtrain, Ytrain, cv = 5)
print('Score is: '+ str(np.mean(score)))
# Plot between predicted values and residuals

plt.scatter(train_pre, train_pre - Ytrain, c = "red",  label = "Training data")
plt.scatter(test_pre,test_pre - Ytest, c = "green",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.plot([0, 1.1], [0, 0], c = "black")

plt.show()
# Plot predictions (real values)

plt.scatter(train_pre, Ytrain, c = "red",  label = "Training data")
plt.scatter(test_pre, Ytest, c = "green",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([0.1, 0.95], [0, 1], c = "black")
plt.show()
# Defining list of alphas

alphas = [0.00025, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006,
          0.007, 0.008, 0.009, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2]
# Setting alphas and fitting with ridge

ridge = RidgeCV(alphas = alphas)
ridge.fit(Xtrain,Ytrain)
alpha = ridge.alpha_
print('The best alpha found was: ',alpha)
print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5)
ridge.fit(Xtrain, Ytrain)
alpha = ridge.alpha_
print("Best alpha : ", alpha)
model_ridge = Ridge()
cv_ridge = [rmse_CV_train(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.xlim((0.00225, 0.0026))
plt.ylim((0.0853, 0.0855))
# Calculating the Ridge RMSE

print("Ridge RMSE on Training set :", rmse_CV_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_CV_test(ridge).mean())
score = cross_val_score(ridge,Xtrain, Ytrain, cv = 5)
print('Score is: '+ str(np.mean(score)))
Ytrain_ridge = ridge.predict(Xtrain)
Ytest_ridge = ridge.predict(Xtest)
# Plot between predicted values and residuals

plt.scatter(Ytrain_ridge, Ytrain_ridge -Ytrain, c = "red",  label = "Training data")
plt.scatter(Ytest_ridge, Ytest_ridge -Ytest, c = "green", marker = "v", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.plot([0, 1.2], [-0.6, 0.6], c = "black")
plt.show()
# Plot predictions - Real values

plt.scatter(Ytrain_ridge, Ytrain, c = "red",  label = "Training data")
plt.scatter(Ytest_ridge, Ytest, c = "green",  label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left") 
plt.plot([0.27, 0.93], [0.2, 1], c = "black")
plt.show()
# Defining another list of alphas

alphas2 = [0.025, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 1.25, 1.5, 1.75, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5,
           10, 13, 15, 17, 20, 23, 25, 30, 35, 40, 50]
# Setting alphas and fitting with lasso

lasso = LassoCV(alphas = alphas2).fit(Xtrain, Ytrain)
alpha2 = lasso.alpha_
print('The best alpha found was: ',alpha2)
print("Try again for more precision with alphas centered around " + str(alpha2))
lasso = LassoCV(alphas = [alpha2 * .6, alpha2 * .65, alpha2 * .7, alpha2 * .75, alpha2 * .8, alpha2 * .85, 
                          alpha2 * .9, alpha2 * .95, alpha2, alpha2 * 1.05, alpha2 * 1.1, alpha2 * 1.15,
                          alpha2 * 1.25, alpha2 * 1.3, alpha2 * 1.35, alpha2 * 1.4],cv = 5)
lasso.fit(Xtrain, Ytrain)
alpha2 = lasso.alpha_
print("Best alpha : ", alpha2)
# Calculating the Ridge RMSE

print("Lasso RMSE on Training set :", rmse_CV_train(lasso).mean())
print("Lasso RMSE on Test set :", rmse_CV_test(lasso).mean())
#Setting Alphas and L1-ratio

alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]
l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5]
ENet = ElasticNet()
# Defining the CV Eastic Net

cv_elastic = [rmse_CV_train(ElasticNet(alpha = alpha, l1_ratio=l1_ratio)).mean() 
            for (alpha, l1_ratio) in product(alphas, l1_ratios)]
# Plot the relation between the alphas, l1 and the RMSE

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
idx = list(product(alphas, l1_ratios))
p_cv_elastic = pd.Series(cv_elastic, index = idx)
p_cv_elastic.plot(title = "Validation")
plt.xlabel("Alpha - L1_ratio")
plt.ylabel("RMSE")
# Let's zoom in to the first 10 parameter pairs

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
idx = list(product(alphas, l1_ratios))[:10]
p_cv_elastic = pd.Series(cv_elastic[:10], index = idx)
p_cv_elastic.plot(title = "Validation")
plt.xlabel("Alpha - L1_ratio")
plt.ylabel("RMSE")
score = rmse_CV_train(ENet)
print("Elastic Net RMSE on Train set: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9)
elastic.fit(Xtrain, Ytrain)
ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.9,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
# Let's look at the residuals
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":elastic.predict(Xtrain), "true":Ytrain})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
score = cross_val_score(elastic,Xtrain, Ytrain, cv = 5)
print('Score is: '+ str(np.mean(score)))
print('R^2 train: %.3f' %  r2_score(preds['true'], preds['preds']))
coef = pd.Series(elastic.coef_, index = Xtrain.columns)
print("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Elastic Net Model")
RF = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 15, oob_score=True, random_state=1234)
score = cross_val_score(RF,Xtrain, Ytrain, cv = 5, n_jobs = -1)
print("Random Forest RMSE on Training set :", rmse_CV_train(RF).mean())
print("Random Forest RMSE on Test set :", rmse_CV_test(RF).mean())
print('Score is: '+ str(np.mean(score)))
# Setting the K numbers

neighbors = [3,5,7,9,11,13,15,20,25,30]
# Here, we define a function that test (with CV equal to 3) various K-nn and yours misclassification error

print('With CV = 3')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=3)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 3)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 5) various K-nn and yours misclassification error

print('With CV = 5')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=5)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 5)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 7) various K-nn and yours misclassification error

print('With CV = 7')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=7)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 7)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 9) various K-nn and yours misclassification error

print('With CV = 9')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=9)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 9)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 11) various K-nn and yours misclassification error

print('With CV = 11')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=11)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 11)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 13) various K-nn and yours misclassification error

print('With CV = 13')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=13)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 13)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 15) various K-nn and yours misclassification error

print('With CV = 15')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=15)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 15)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 20) various K-nn and yours misclassification error

print('With CV = 20')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=20)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 20)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 25) various K-nn and yours misclassification error

print('With CV = 25')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=25)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 25)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Here, we define a function that test (with CV equal to 30) various K-nn and yours misclassification error

print('With CV = 30')
cv_scores = []

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k,weights='uniform')
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=30)
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# Plot a graph that show us the perfomance of the K-nn with each number of neighbors (in this case, the CV is equal to 30)

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
# Set the K-nn with K=7 and CV=20

knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(Xtrain,Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=20).mean()
print(scores)
# Open the file that contains a sample how to send ours predictions

sample = pd.read_csv('../input/atividade-3-pmr3508/sample_sub_1.csv')
print ("submission size: ", sample.shape)
sample.head()
Ytrain['median_house_value'] = trainMEDIAN
Ytest['median_house_value'] = testMEDIAN
Y = Ytrain + Ytest
Xtrain['Id'] = trainID
Xtest['Id'] = testID
X = Xtrain + Xtest
testTOsubmit['Id'] = testTOsubmitID
# Let's fit with thw Random Forest

RF.fit(X,Y)
# Drop the target column with NaN values

testTOsubmit = testTOsubmit.dropna(axis=1, how='all')
predictionsTOsubmit = RF.predict(testTOsubmit)
# Sending the predictions

str(predictionsTOsubmit)
ids = testTOsubmitID
submission = pd.DataFrame({'Id':ids,'median_house_value':predictionsTOsubmit[:]})
submission.to_csv("predictionsT3.csv", index = False)
submission.shape
submission