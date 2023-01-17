# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# create new df
df = pd.read_csv('/kaggle/input/sf-building-permits/Building_Permits_on_or_after_January_1__2013.csv');
df.head()
pd.set_option('display.max_columns',100)
df.head(1)
# translate to lowercase
df.columns = map(str.lower, df.columns)
# remove characters from names
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('-', '') for c in df.columns]
df.head(1)
import missingno as msno
msno.matrix(df.sample(250))
msno.bar(df.sample(1000))
msno.heatmap(df)
msno.dendrogram(df)
# delete columns with a name delet_
df.drop(df.filter(regex='delete').columns, axis=1, inplace=True)
df.head(5)
df.nunique()
data_loc = df.loc[:,['estimated_cost', 'revised_cost','permit_creation_date']]
data_cost = data_loc 
data_cost.permit_creation_date = pd.to_datetime(data_cost.permit_creation_date)
data_cost = data_cost.set_index('permit_creation_date')
data_cost = data_cost.dropna()
data_cost_m = data_cost.groupby(pd.Grouper(freq='M')).sum()
data_cost_m.head()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(19,8))
# Add title
plt.title("Estimated costs and revised costs in 2013-2020")
sns.lineplot(data=data_cost_m)
data_cost_y = data_cost.groupby(pd.Grouper(freq='Y')).sum()
plt.figure(figsize=(19,8))
# Add title
plt.title("Estimated costs and revised costs in 2013-2020")
# Line chart showing daily global streams of each song 
sns.lineplot(data=data_cost_y)
data_cost_d = data_loc
data_cost_d = data_cost_d.dropna()
data_cost_d.permit_creation_date = data_cost_d.permit_creation_date.dt.day_name()

cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
data_cost_d  = data_cost_d.groupby(['permit_creation_date']).count().reindex(cats) 
plt.figure(figsize=(18,6))
data_cost_d.plot.bar()
sns.lineplot(data=data_cost_d)
df[['lat','long']] = df.location.str.split(",",expand=True)
data_location = df.loc[:,['long','lat','zipcode','estimated_cost']]
data_location = data_location.dropna()
data_location.head()
data_location.long = data_location.long.apply(lambda x: x.replace(')',''))
data_location.lat = data_location.lat.apply(lambda x: x.replace('(',''))
data_location.info()
data_location.lat = pd.to_numeric(data_location.lat)
data_location.long = pd.to_numeric(data_location.long)
data_location_mean = data_location.groupby(['zipcode'])['lat','long','estimated_cost'].mean()
data_location_mean.head()
import folium
from folium import Circle

# map folium display
lat = data_location_mean.lat.mean()
long = data_location_mean.long.mean()
map1 = folium.Map(location = [lat, long], zoom_start = 12, tiles='Stamen Toner')
param = 'estimated_cost'
for i in range(0,len(data_location_mean)):
    Circle(
        location = [data_location_mean.iloc[i]['lat'], data_location_mean.iloc[i]['long']],
        radius= [data_location_mean.iloc[i]['estimated_cost']/5000],
        fill = True).add_to(map1)
map1
data_location_lang_long = data_location.groupby(['zipcode'])['lat','long'].mean()
data_location_lang_long.head

data_location_lang_long = data_location_lang_long.assign(cost = data_location.groupby(['zipcode'])['estimated_cost'].sum())
data_location_lang_long.head()
import folium
from folium import Circle

# map folium display
lat = data_location_lang_long.lat.mean()
long = data_location_lang_long.long.mean()
map1 = folium.Map(location = [lat, long], zoom_start = 12)

for i in range(0,len(data_location_lang_long)):
    Circle(
        location = [data_location_lang_long.iloc[i]['lat'], data_location_lang_long.iloc[i]['long']],
        radius= [data_location_lang_long.iloc[i]['cost']/20000000],
        fill = True).add_to(map1)
map1
import seaborn as sn
sn.heatmap(df.corr())
dfn = df.dropna(subset=['description'])
dfn.description.isnull().values.any()
dfn = dfn[dfn['description'].str.match('reroofing')]
dfn.head()
dfn.to_excel('eeeee.xls')
df_unit = dfn.loc[:,['estimated_cost','existing_use', 'existing_units', 'zipcode','issued_date']]
df_unit = df_unit.dropna()
df_unit.head(15)
df_unit[df_unit.existing_use.str.contains("family|office|apartments")]

fam1 = df_unit[df_unit['existing_use']=='1 family dwelling']['estimated_cost'].mean()
fam2 = df_unit[df_unit['existing_use']=='2 family dwelling']['estimated_cost'].mean()
office = df_unit[df_unit['existing_use']=='office']['estimated_cost'].mean()
apartments = df_unit[df_unit['existing_use']=='apartments']['estimated_cost'].mean()

data = {'1 family dwelling':fam1,'2 family dwelling':fam2,'Office':office,'Apartments':apartments}
typedf = pd.DataFrame(data = data,index=['Counts'])
typedf.plot(kind='barh', title="Average estimated cost by type", figsize=(8,6));
df_unit.issued_date = pd.to_datetime(df_unit.issued_date)
df_unit.issued_date = df_unit.issued_date.dt.year
years = list(range(2013, 2020)) 
keywords = ['1 family dwelling','2 family dwelling','apartments']

val_data = []
for year in years:
    iss_data = []
    for word in keywords:
        v = df_unit[(df_unit['existing_use']==word) & (df_unit['issued_date']== year)]['estimated_cost'].mean()
        iss_data.append(v)
    val_data.append(iss_data)
#print(val_data)
dfnew = pd.DataFrame(data=val_data, index=years, columns=keywords)
dfnew.head()


dfnew.plot.bar(figsize=(12, 6)) 
plt.xlabel("Years")
plt.ylabel("Estimated cost of reroofing")
plt.title("Estimated cost of Bathroom by year");
dfnew.plot.line(figsize=(12, 6))
df_corr = dfn.dropna(subset=['existing_use'])
df_corr.description.isnull().values.any()
df_1fam = df_corr[df_corr.existing_use.str.contains('1 family')]
num_feuture = df_1fam.select_dtypes(include=[np.number])
corr = num_feuture.corr()
print(corr['estimated_cost'].sort_values(ascending = False))
dfn[['lat','long']] = dfn.location.str.split(",",expand=True)
#df_pr = dfn.loc[:,['permit_creation_date', 'existing_use', 'existing_units','estimated_cost','zipcode','current_supervisor_districts', 'analysis_neighborhoods', ]]
#df_pr = dfn.loc[:,['permit_creation_date', 'zipcode', 'existing_use', 'existing_construction_type', 'estimated_cost', 'long','lat' ]]#
#df_pr = df_1fam.loc[:,['permit_creation_date', 'zipcode', 'number_of_existing_stories', 'number_of_proposed_stories',  'current_police_districts', 'existing_use', 'long','lat', 'record_id',  'estimated_cost',  ]]

df_pr = df_1fam.loc[:,['permit_creation_date', 'zipcode', 'number_of_existing_stories', 'number_of_proposed_stories',  'current_police_districts', 'long','lat', 'record_id',  'estimated_cost',  ]]
df_pr = df_pr.dropna()
#df_pr = df_pr[df_pr.existing_use.str.contains('1 family')]
df_pr.permit_creation_date = pd.to_datetime(df_pr.permit_creation_date)
df_pr.head()
histplot = df_pr.estimated_cost.plot.hist(bins = 40)
indexNames = df_pr[ (df_pr['estimated_cost'] > 20000)].index
df_pr.drop(indexNames , inplace=True)
indexNames = df_pr[ (df_pr['estimated_cost'] < 12000)].index
df_pr.drop(indexNames , inplace=True)
histplot = df_pr.estimated_cost.plot.hist(bins = 40)
import scipy.stats as st
y = df_pr['estimated_cost']
plt.figure(figsize=(7,4))
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(figsize=(7,4))
plt.figure(figsize=(7,4))
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
df_pr.describe()
df_pr.long= df_pr.long.apply(lambda x: x.replace(')',''))
df_pr.lat = df_pr.lat.apply(lambda x: x.replace('(',''))
df_pr.lat = pd.to_numeric(df_pr.lat)
df_pr.long = pd.to_numeric(df_pr.long)
from geopy.distance import vincenty
def distance_calc (row):
    start = (row['lat'], row['long'])
    stop = (37.7945742, -122.3999445)

    return vincenty(start, stop).meters/1000

df_pr['distance'] = df_pr.apply (lambda row: distance_calc (row),axis=1)
def downtown_proximity(dist):
    '''
    < 2 -> Near Downtown
    >= 2, <6 -> <0.5H Downtown
    >= 6, <10 -> <1H Downtown 
    >= 10 -> Outside SF

    '''
    if dist < 2:
        return 'Downtown'
    elif dist < 6:
        return  '<0.5H Downtown'
    elif dist < 10:
        return '<1H Downtown'
    elif dist >= 10:
        return 'Outside SF'

df_pr['downtown_proximity'] = df_pr.distance.apply(downtown_proximity)
value_count=df_pr['downtown_proximity'].value_counts()
plt.figure(figsize=(12,5))
plt.title('Estimated cost rerofing depending on Downtown Proximity');
sns.boxplot(x="downtown_proximity", y="estimated_cost", data=df_pr);
sns.set()
local_coord=[-122.3999445, 37.7945742] # the point near which we want to look at our variables
euc_dist_th = 0.03 # distance treshhold

euclid_distance=df_pr[['lat','long']].apply(lambda x:np.sqrt((x['long']-local_coord[0])**2+(x['lat']-local_coord[1])**2), axis=1)

# indicate wethere the point is within treshhold or not
indicator=pd.Series(euclid_distance<=euc_dist_th, name='indicator')

print("Data points within treshhold:", sum(indicator));

# a small map to visualize th eregion for analysis
sns.lmplot('long', 'lat', data=pd.concat([df_pr,indicator], axis=1), hue='indicator', markers ='.', fit_reg=False, height=8);

sns.lmplot('long', 'lat', data=df_pr,markers ='.', hue='downtown_proximity', fit_reg=False, height=8)
plt.show()
#df_pr['month'] = df_pr.permit_creation_date.dt.month
df_pr['year'] = df_pr.permit_creation_date.dt.year
df_pr = df_pr.drop(columns=['permit_creation_date', 'long', 'lat'])
#df_pr = pd.concat([df_pr, pd.get_dummies(df_pr.existing_use, prefix='existing_use')], axis=1)
df_pr = pd.concat([df_pr, pd.get_dummies(df_pr.downtown_proximity, prefix='dt_pr')], axis=1)

#df_pr = df_pr.drop(columns=['existing_use'])
df_pr = df_pr.drop(columns=['downtown_proximity'])
df_pr.describe()
#df_pr.existing_units = df_pr.existing_units.apply(lambda x: 10 if x > 10 else x)
df_pr.zipcode = df_pr.zipcode - df_pr.zipcode.min()
df_pr.year = df_pr.year - df_pr.year.min()
df_pr.record_id = df_pr.record_id - df_pr.record_id.min()
#df_pr.head()
df_pr.head()
df_pr.hist(bins=50, figsize=(10, 10));
from sklearn.manifold import TSNE
tsne=TSNE(perplexity = 3)
tsne.fit(df_pr)
plt.scatter(tsne.embedding_[:,0], tsne.embedding_[:,1])

df_pr['proofcost'] = df_pr.estimated_cost.apply(lambda x: True if x>=13000 else False )
plt.scatter(tsne.embedding_[df_pr.proofcost.values, 0], tsne.embedding_[df_pr.proofcost.values, 1], color='orange')
plt.scatter(tsne.embedding_[~df_pr.proofcost.values, 0], tsne.embedding_[~df_pr.proofcost.values, 1], color='blue')

df_pr = df_pr.drop(columns = 'proofcost')
#df_pr = df_pr.drop(['existing_construction_type'], axis = 1)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold

X_training = df_pr.drop(['estimated_cost'], axis = 1)
y_training = df_pr['estimated_cost']

from sklearn.model_selection import train_test_split #to create validation data set
X_train, X_valid, y_train, y_valid = train_test_split(X_training, y_training, test_size=0.2, random_state=0) 
#X_valid and y_valid are the validation sets
linreg = LinearRegression()
linreg.fit(X_train, y_train)
lin_pred = linreg.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))
lin_pred = linreg.predict([[20.0, 1.0, 3.0, 4.0, 1163316512454, 4.825703, 0, 0, 0, 0, 1]])
print("Prediction for data with arbitrary values: " + str(lin_pred[0]))

linsvc = DecisionTreeRegressor()
linsvc.fit(X_train, y_train)
lin_pred = linsvc.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))

linsvc = linsvc.predict([[20.0, 1.0, 3.0, 4.0, 1163316512454, 4.825703, 0, 0, 0, 0, 1]])
print("Prediction for data with arbitrary values: " + str(linsvc[0]))
dtr = DecisionTreeRegressor()
parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 
                  "max_features" : ["auto", "log2"]}
grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")
grid_dtr.fit(X_train, y_train)

print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))
print("Best Score: " + str(grid_dtr.best_score_))
dtr = grid_dtr.best_estimator_
dtr.fit(X_train, y_train)
dtr_pred = dtr.predict(X_valid)
r2_dtr = r2_score(y_valid, dtr_pred)
rmse_dtr = np.sqrt(mean_squared_error(y_valid, dtr_pred))
print("R^2 Score: " + str(r2_dtr))
print("RMSE Score: " + str(rmse_dtr))
#scores_dtr = cross_val_score(dtr, X_train, y_train, cv=10, scoring="r2")
#print("Cross Validation Score: " + str(np.mean(scores_dtr)))
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_valid)
r2_rf = r2_score(y_valid, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))
print("R^2 Score: " + str(r2_rf))
print("RMSE Score: " + str(rmse_rf))
scores_rf = cross_val_score(rf, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_rf)))
lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_valid)
r2_lasso = r2_score(y_valid, lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_valid, lasso_pred))
print("R^2 Score: " + str(r2_lasso))
print("RMSE Score: " + str(rmse_lasso))
scores_lasso = cross_val_score(lasso, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)));
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_valid)
r2_ridge = r2_score(y_valid, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))
print("R^2 Score: " + str(r2_ridge))
print("RMSE Score: " + str(rmse_ridge))
scores_ridge = cross_val_score(ridge, X_train, y_train, cv=10, scoring="r2");
print("Cross Validation Score: " + str(np.mean(scores_ridge)));
model_performances = pd.DataFrame({
    "Model" : ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor","Ridge", "Lasso"],
    "R Squared" : [str(r2_lin)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5]],
    "RMSE" : [str(rmse_lin)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8]]
})
model_performances.round(4)
X_train_v = X_train.values
y_train_v = y_train.values
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train,y_train)
from sklearn.tree import export_graphviz
export_graphviz(dtr_pred, out_file='tree.dot', feature_names = X_trai.columns, filled = True)
!dot -Tpng tree.dot -o tree.png -Gdpi = 600
from IPython.display import Image
Image(filename = 'tree.png' )
"""
X = df_pr.drop(['estimated_cost'], axis = 1).values
y = df_pr['estimated_cost'].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size = .3, random_state=0)
"""
"""
y_df = df_pr['estimated_cost']
X_df = df_pr.drop(['estimated_cost'], axis = 1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=5)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)
"""
"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge=Ridge()
parameters= {'alpha':[x for x in [0.1,0.2,0.4,0.5,0.7,0.8,1]]}

ridge_reg=GridSearchCV(ridge, param_grid=parameters)
ridge_reg.fit(X_train,Y_train)
print("The best value of Alpha is: ",ridge_reg.best_params_)
"""
m = RandomForestRegressor(n_jobs=-1)
m.fit(df_pr, y)
m.score(df_pr,y)
y_df = df_pr['estimated_cost']
X_df = df_pr.drop(['estimated_cost'], axis = 1)

def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()

n_valid = int(len(df_pr)*0.25)  # same as Kaggle's test set size
n_trn = len(df_pr)-n_valid
#raw_train, raw_valid = split_vals(X_df, n_trn)
X_train, X_valid = split_vals(X_df, n_trn)
y_train, y_valid = split_vals(y_df, n_trn)

X_train.shape, y_train.shape, X_valid.shape
import math 
#define a function to check rmse value
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
def split_vals(a,n):
   return a[:n].copy(), a[n:].copy()

n_valid = int(len(df_pr)*0.25)  
n_trn = len(df_pr)-n_valid

raw_train, raw_valid = split_vals(df_pr, n_trn)
X_train, X_valid = split_vals(X_df, n_trn)
y_train, y_valid = split_vals(y_df, n_trn)
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state = 42) 
dt_model.fit(train_X, train_Y)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)

y_pred = dt_model.predict([[116,2.0,1.0,5.0,10.0,81.0,9.781119,0,1,0,0,1,0,0]])
print(y_pred)
Y_pred = dt_model.predict(validation_X)
s = pd.Series(Y_pred)
validation_Y.reindex()
df = pd.concat([s.reset_index(drop=True), validation_Y.reset_index(drop=True)], axis=1, ignore_index=True)
df.tail(15)
Y = df_pr.estimated_cost
X = df_pr.drop(['estimated_cost'], axis = 1)

from sklearn.model_selection import train_test_split
train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, random_state = 42)

print("Training set: Xt:{} Yt:{}".format(train_X.shape, train_Y.shape)) 
print("Validation set: Xv:{} Yv:{}".format(validation_X.shape, validation_Y.shape)) 
print("-") 
print("Full dataset: X:{} Y:{}".format(X.shape, Y.shape))
from sklearn.metrics import accuracy_score
score = accuracy_score(validation_Y, Y_pred)
print(score)
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 42) 
model.fit(train_X, train_Y)

from sklearn.metrics import mean_absolute_error

# instruct our model to make predictions for the prices on the validation set 
validation_predictions = model.predict(validation_X)

# calculate the MAE between the actual prices (in validation_Y) and the predictions made 
validation_prediction_errors = mean_absolute_error(validation_Y, validation_predictions)

validation_prediction_errors
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10, random_state=17, shuffle=True)
features_for_trees=['street_number', 'number_of_existing_stories', 'existing_units',
       'existing_construction_type', 'zipcode', 'sf_find_neighborhoods',
       'distance', 'year', 'existing_use_1 family dwelling',
       'existing_use_2 family dwelling', 'dt_pr_<0.5H Downtown',
       'dt_pr_<1H Downtown', 'dt_pr_Downtown', 'dt_pr_Outside SF']
TOTAL = df_pr.count()[0] 
N_VALID = 0.25 # Three months 
TRAIN = int(TOTAL*N_VALID)
df_small = df_pr
features = ['street_number', 'number_of_existing_stories', 'existing_units',
       'existing_construction_type', 'zipcode', 'sf_find_neighborhoods',
       'distance', 'year', 'existing_use_1 family dwelling',
       'existing_use_2 family dwelling', 'dt_pr_<0.5H Downtown',
       'dt_pr_<1H Downtown', 'dt_pr_Downtown', 'dt_pr_Outside SF']
df_pr
y_df = df_small['estimated_cost']
X_train, X_val = X_df[:TRAIN], X_df[TRAIN:]
y_train, y_val = y_df[:TRAIN], y_df[TRAIN:]
#define a function to check rmse value
import  math 
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_val), y_val),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

model = RandomForestRegressor(n_estimators=40, bootstrap=True, min_samples_leaf=25)
model.fit(X_train, y_train)
#draw_tree(model.estimators_[0], X_train, precision=2)
print_score(model)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#training_scores_encoded = lab_enc.fit_transform(df_pr.estimated_cost)
#print(training_scores_encoded)


y = df_pr.estimated_cost
X = df_pr.drop(['estimated_cost'], axis = 1)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.25, random_state = 17)
X.shape, y.shape

"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score




ml_arr =[SVC(), GaussianNB(), Perceptron(), SGDClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]


for el in ml_arr:
    el.fit(X_train, y_train)
    Y_pred = el.predict(X_valid)
    Y_pred.reshape(-1, 1)
    #Y_pred = lab_enc.fit_transform(Y_pred)
    #acc = round(el.score(y_valid, Y_pred) * 100, 2)
    score = accuracy_score(y_valid, Y_pred)
    print(score)
"""
# using scaled data
X=pd.concat([train_df[dummies_names], X_train_scaled[numerical_features]], axis=1, ignore_index = True)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10, random_state=17, shuffle=True)
from sklearn.linear_model import Ridge

model=Ridge(alpha=1)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))
