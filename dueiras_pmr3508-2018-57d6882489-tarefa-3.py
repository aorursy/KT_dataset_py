import pandas as pd
import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error 
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

train_data = pd.read_csv('../input/train.csv',
        engine='python')

test_data = pd.read_csv('../input/test.csv',
        engine='python')
train_data.head()

train_data.shape
train_data.info()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    corr_matrix = abs(train_data.corr())
    display(corr_matrix['median_house_value'].sort_values(ascending=False))
plt.plot(train_data['latitude'], train_data['longitude'], 'ro')
plt.show()
plt.matshow(train_data.corr())
PROJECT_ROOT_DIR = "."   
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):  
    if not os.path.isdir(IMAGES_PATH):     
        os.makedirs(IMAGES_PATH)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

%matplotlib inline
train_data.hist(bins=50, figsize=(20,15))   
save_fig("attribute_histogram_plots")
plt.show()
Xtrain = train_data
Xtrain = Xtrain.drop('Id', axis=1)
Xtrain = Xtrain.drop('median_house_value', axis=1)
Ytrain = train_data['median_house_value']
def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return ((sum/len(predicted))**0.5)
def results(Xtrain, Ytrain):
    R=[]
    accuracy = 0.0
    error = 1.0
    best_accuracy = 0
    best_error = 0
    
    #knn
    n = 11
    
    knn = KNeighborsRegressor(n_neighbors=n)
    knn.fit(Xtrain, Ytrain)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    Ypred = knn.predict(Xtrain)
    knn_score = scores.mean()
    knn_rmsle = rmsle(Ytrain, Ypred)
    R.append('knn score = ')
    R.append(knn_score)
    R.append('knn rmsle = ')
    R.append(knn_rmsle)
    R.append('\n')
    if knn_score>accuracy:
        accuracy = knn_score
        best_accuracy = 'knn'
    if knn_rmsle<error:
        error = knn_rmsle
        best_error = 'knn'
    
    
    #Lasso
    
    b = 0.1
    
    reg = linear_model.Lasso(alpha = b)
    reg.fit(Xtrain, Ytrain)
    scores = cross_val_score(reg, Xtrain, Ytrain, cv=10)
    Ypred=reg.predict(Xtrain)
    lasso_score = scores.mean()
    lasso_rmsle = rmsle(Ytrain, Ypred)
    R.append('lasso score = ')
    R.append(lasso_score)
    R.append('lasso rmsle = ')
    R.append(lasso_rmsle)
    R.append('\n')
    if lasso_score>accuracy:
        accuracy = lasso_score
        best_accuracy = 'lasso'
    if lasso_rmsle<error:
        error = lasso_rmsle
        best_error = 'lasso'
    
    #Ridge
    
    b = 0.1
    
    rid = linear_model.Ridge(alpha = b)
    rid.fit(Xtrain, Ytrain)
    scores = cross_val_score(rid, Xtrain, Ytrain, cv=10)
    Ypred = rid.predict(Xtrain)
    ridge_score = scores.mean()
    ridge_rmsle = rmsle(Ytrain, Ypred)
    R.append('ridge score = ')
    R.append(ridge_score)
    R.append('ridge rmsle = ')
    R.append(ridge_rmsle)
    R.append('\n')
    if ridge_score>accuracy:
        accuracy = ridge_score
        best_accuracy = 'ridge'
    if ridge_rmsle<error:
        error = ridge_rmsle
        best_error = 'ridge'
    
    #Lasso Lars
    
    b = 0.1
    
    lars = linear_model.LassoLars(alpha = b)
    lars.fit(Xtrain, Ytrain)
    scores = cross_val_score(lars, Xtrain, Ytrain, cv=10)
    lars_score = scores.mean()
    Ypred = lars.predict(Xtrain)
    lars_rmsle = rmsle(Ytrain, Ypred)
    R.append('lars score = ')
    R.append(lars_score)
    R.append('lars rmsle = ')
    R.append(lars_rmsle)
    R.append('\n')
    if lars_score>accuracy:
        accuracy = lars_score
        best_accuracy = 'lars'
    if lars_rmsle<error:
        error = lars_rmsle
        best_error = 'lars'

    #Random Forest
    
    d = 4
    
    forest = RandomForestRegressor(max_depth=d, random_state=0, n_estimators=100)
    forest.fit(Xtrain, Ytrain)
    scores = cross_val_score(forest, Xtrain, Ytrain, cv=10)
    Ypred = forest.predict(Xtrain)
    forest_score = scores.mean()
    forest_rmsle = rmsle(Ytrain, Ypred)
    R.append('forest score = ')
    R.append(forest_score)
    R.append('forest rmsle = ')
    R.append(forest_rmsle)
    R.append('\n')
    if forest_score>accuracy:
        accuracy = forest_score
        best_accuracy = 'forest'
    if forest_rmsle<error:
        error = forest_rmsle
        best_error = 'forest'
    
    
    for i in R:
        print(i)
    
    print('best accuracy =', accuracy, 'with', best_accuracy)
    print('minimum log error =', error, 'with', best_error)
y = []
x = range(1, 15)
n = 0
s = 0.0
for i in x:
    knn = KNeighborsRegressor(n_neighbors=i)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    if scores.mean()>s:
        n=i
        s=scores.mean()
    y.append(scores.mean())
plt.scatter(x, y)
y = []
x = range(1, 11)
for i in x:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)
    Ypred = knn.predict(Xtrain)
    r = rmsle(Ytrain,Ypred)
    y.append(r)
plt.scatter(x, y)
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(Xtrain, Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
Ypred = knn.predict(Xtrain)
scores.mean()
rmsle(Ytrain, Ypred)
y = []
x = range(1, 10)
for i in x:
    a = i/10
    reg = linear_model.Lasso(alpha = a)
    scores = cross_val_score(reg, Xtrain, Ytrain, cv=10)
    y.append(scores.mean())
plt.scatter(x, y)
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(Xtrain, Ytrain)
scores = cross_val_score(reg, Xtrain, Ytrain, cv=10)
Ypred=reg.predict(Xtrain)
scores.mean()
rmsle(Ytrain, Ypred)
rid = linear_model.Ridge(alpha = 0.5)
rid.fit(Xtrain, Ytrain)
scores = cross_val_score(rid, Xtrain, Ytrain, cv=10)
Ypred = rid.predict(Xtrain)
scores.mean()
rmsle(Ytrain, Ypred)
bay = linear_model.BayesianRidge()
bay.fit(Xtrain, Ytrain)
scores = cross_val_score(bay, Xtrain, Ytrain, cv=10)
Ypred = bay.predict(Xtrain)
scores.mean()
rmsle(Ytrain, Ypred)
lars = linear_model.LassoLars(alpha = 0.1)
lars.fit(Xtrain, Ytrain)
scores = cross_val_score(lars, Xtrain, Ytrain, cv=10)
Ypred = lars.predict(Xtrain)
scores.mean()
rmsle(Ytrain, Ypred)
california_sea=[(41.990352, -124.216535),(41.936725, -124.199048),(41.862157, -124.220161),(41.758672, -124.240793),(41.730317, -124.162807),(41.672629, -124.139878),(41.722746, -124.151351),(41.671813, -124.136762),(41.618963, -124.109252),(41.470737, -124.072740),(41.383226, -124.066948),(41.308172, -124.094492),(41.212278, -124.121904),
(41.137176, -124.165918),(41.062165, -124.165618),(41.020596, -124.115740),(40.928851, -124.143028),(40.858028, -124.126245),(40.812048, -124.181163),(40.728511, -124.235831),(40.649059, -124.301387),(40.586325, -124.344954),(40.511043, -124.388365),(40.440002, -124.409806),(40.395399, -124.383960),(40.322914, -124.349643),(40.241803, -124.337706),(40.186635, -124.253402),(40.122885, -124.169203),(40.067673, -124.068499),(40.008009, -124.029231),
(39.922813, -123.945453),(39.837566, -123.873007),(39.735216, -123.828474),(39.654186, -123.789622),(39.564619, -123.761930),(39.399528, -123.821626),(39.201588, -123.770073),(39.076989, -123.691566),(38.960637, -123.724138),(38.879044, -123.662811),(38.754580, -123.507611),(38.634199, -123.386034),(38.496411, -123.193367),(38.336876, -123.061865),(38.259117, -122.974368),
(38.151338, -122.952917),(38.060918, -122.980669),(37.996318, -123.002792),
(38.026254, -122.926130),(38.004306, -122.827828),(37.931906, -122.744687),(37.902923, -122.652017),(37.872444, -122.594173),(37.880984, -122.392446),
(37.815555, -122.367515),(37.628327, -122.331577),(37.542968, -122.455670),
(37.370235, -122.414093),(37.290236, -122.415691),(37.167091, -122.356855),
(37.088046, -122.276348),(36.987005, -122.157357),(36.951905, -122.049790),
(36.969554, -121.914753),(36.925477, -121.862435),(36.824092, -121.802024),
(36.620740, -121.851334),(36.480625, -121.934216),(36.282719, -121.866908),
(36.162592, -121.678018),(35.990860, -121.498031),(35.827849, -121.382193),
(35.671399, -121.272296),(35.608589, -121.143265),(35.453082, -120.919491),
(35.297750, -120.877400),(35.189759, -120.819107),(35.180890, -120.736397),
(35.097645, -120.628863),(34.932680, -120.660285),(34.842040, -120.610177),
(34.742216, -120.618143),(34.583391, -120.639685),(34.528043, -120.518413),
(34.457687, -120.472919),(34.458791, -120.347644),(34.469789, -120.138306),
(34.422313, -119.903627),(34.399196, -119.699791),(34.408922, -119.552255),(34.335795, -119.408499),(34.288024, -119.329889),(34.199208, -119.247261),(34.115993, -119.153777),(34.041474, -118.899965),(34.035682, -118.855901),(34.018486, -118.822894),(34.003602, -118.805037),(34.016106, -118.785710),
(34.029683, -118.744327),(34.037409, -118.667109),(34.036912, -118.580005),
(34.009365, -118.502919),(33.984242, -118.472597),(33.960222, -118.454035),(33.867022, -118.402873),(33.810913, -118.390523),(33.770287, -118.420867),(33.716625, -118.060214),(33.606537, -117.889392),(33.385674, -117.578771),(33.270497, -117.443285),(33.127431, -117.326314),(33.053581, -117.291643),(32.831417, -117.277875),(32.683026, -117.189643),(32.536805, -117.122224)]
train_data2 = train_data
sea_distance=[]
for i,c in train_data.iterrows():
    b=np.array((c['latitude'],c['longitude']))
    D=[]
    for j in california_sea:
        D.append(np.linalg.norm(b-j))
    sea_distance.append(min(D))
train_data2 = train_data2.join(pd.Series(sea_distance,name="sea_distance"))
Los_Angeles = (34.155652, -118.600019)
San_Francisco = (37.775, -122.4183)

la_sf_distance = []
for i,c in train_data2.iterrows():
    b=np.array((c['latitude'],c['longitude']))
    D=[]
    D.append(np.linalg.norm(b-Los_Angeles))
    D.append(np.linalg.norm(b-San_Francisco))
    la_sf_distance.append(min(D))
train_data2 = train_data2.join(pd.Series(la_sf_distance,name="LA_SF_distance"))
train_data3 = train_data2
train_data3["rooms_per_household"] = train_data3["total_rooms"]/train_data3["households"]
train_data3["bedrooms_per_room"] = train_data3["total_bedrooms"]/train_data3["total_rooms"]
train_data3["population_per_household"] = train_data3["population"]/train_data3["households"]
train_data3["income_per_person"] = train_data3["median_income"]/train_data3["population_per_household"]
train_data3['mean_rooms'] = train_data3['total_rooms']/train_data3['households']
train_data3['rooms_per_person'] = train_data3['total_rooms']/train_data3['population']
train_data3['mean_bedrooms'] = train_data3['total_bedrooms']/train_data3['households']
train_data3['bedrooms_per_person'] = train_data3['total_bedrooms']/train_data3['households']
train_data3['persons_per_household'] = train_data3['population']/train_data3['households']
train_data3['total_income'] = train_data3['median_income']*train_data3['households']
corr_matrix = train_data3.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
Xtrain3 = train_data3
Xtrain3 = Xtrain3.drop('Id', axis=1)
Xtrain3 = Xtrain3.drop('median_house_value', axis=1)
results(Xtrain3, Ytrain)
d = 8
n = 100
        
boost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=d),
                          n_estimators=n)        
boost.fit(Xtrain3, Ytrain)

Ypred = boost.predict(Xtrain3)

boost_rmsle = rmsle(Ytrain, Ypred)

print('error =', boost_rmsle)
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(Xtrain3, Ytrain, train_size=0.7)


from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=200, depth=6, learning_rate=0.2, loss_function='RMSE')
model.fit(Xtrain3, Ytrain,eval_set=(X_validation, y_validation),plot=True)
Ypred = model.predict(Xtrain3)
model_rmsle = rmsle(Ytrain, Ypred)
print('error =', model_rmsle)
test_data2 = test_data
sea_distance=[]
for i,c in test_data2.iterrows():
    b=np.array((c['latitude'],c['longitude']))
    D=[]
    for j in california_sea:
        D.append(np.linalg.norm(b-j))
    sea_distance.append(min(D))
test_data2 = test_data2.join(pd.Series(sea_distance,name="sea_distance"))

Los_Angeles = (34.155652, -118.600019)
San_Francisco = (37.775, -122.4183)

la_sf_distance = []
for i,c in test_data2.iterrows():
    b=np.array((c['latitude'],c['longitude']))
    D=[]
    D.append(np.linalg.norm(b-Los_Angeles))
    D.append(np.linalg.norm(b-San_Francisco))
    la_sf_distance.append(min(D))
test_data2 = test_data2.join(pd.Series(la_sf_distance,name="LA_SF_distance"))

test_data2["rooms_per_household"] = test_data2["total_rooms"]/test_data2["households"]
test_data2["bedrooms_per_room"] = test_data2["total_bedrooms"]/test_data2["total_rooms"]
test_data2["population_per_household"] = test_data2["population"]/test_data2["households"]
test_data2["income_per_person"] = test_data2["median_income"]/test_data2["population_per_household"]
test_data2['mean_rooms'] = test_data2['total_rooms']/test_data2['households']
test_data2['rooms_per_person'] = test_data2['total_rooms']/test_data2['population']
test_data2['mean_bedrooms'] = test_data2['total_bedrooms']/test_data2['households']
test_data2['bedrooms_per_person'] = test_data2['total_bedrooms']/test_data2['households']
test_data2['persons_per_household'] = test_data2['population']/test_data2['households']
test_data2['total_income'] = test_data2['median_income']*test_data2['households']
Xtest2 = test_data2.drop('Id', axis=1)
forest = RandomForestRegressor(max_depth=21, random_state=0, n_estimators=1000)
forest.fit(Xtrain3, Ytrain)

Ypred = forest.predict(Xtrain3)
forest_rmsle = rmsle(Ytrain, Ypred)

print('log error =', forest_rmsle)

prediction = forest.predict(Xtest2)
arq = open ("submission.csv", "w")
arq.write("Id,median_house_value\n")
for i, j in zip(test_data2['Id'], prediction):
    arq.write(str(i)+ "," + str(j)+"\n")
arq.close()