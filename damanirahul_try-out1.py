# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import copy

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from collections import OrderedDict 
from itertools import chain

import pickle


%matplotlib inline
# for better view
pd.options.display.max_columns = None
pd.options.display.max_rows = None
# reading in the labeled data
df = pd.read_csv("../input/ihsmarkit-hackathon-june2020/train_data.csv")
df.head()
# missing values
df.info()
df.columns
numerical = ['Length', 'Height', 'Width', 'Engine_KW', 'No_of_Gears', 'Curb_Weight','CO2', 'Fuel_cons_combined',"Price_USD"]
# last two are not really required
categorical = ['Body_Type', 'Global_Sales_Sub-Segment', 'Brand',
        'Driven_Wheels', 'Transmission', 'Turbo', 'Fuel_Type',
       'PropSysDesign', 'Plugin', 'Registration_Type',"country_name"]
# brnd can be taken off
# looking at unique values for every column
for i in categorical:
    print(">>>>>>>>>>>>>>>>>>",i)
    print(df[i].value_counts())
sns.pairplot(df[numerical])
# checking for outliers
plt.boxplot(df.Price_USD)
plt.show()
# lot of outliers
for i in categorical:
    print(i)
    print(df.groupby(i).mean()["Price_USD"].sort_values(ascending=False))
categorical
df.shape
# removing unknown
df_dropna_body_type = df[~df["Body_Type"].str.contains("unknown")]
print(df_dropna_body_type.shape)
df_dropna_turbo = df_dropna_body_type[~df["Turbo"].str.contains("unknown")]
print(df_dropna_turbo.shape)
df_dropna_trans = df_dropna_turbo[~df["Transmission"].str.contains("unknown")]
print(df_dropna_trans.shape)
df_dropna_wheels = df_dropna_trans[~df["Driven_Wheels"].str.contains("unknown")]
print(df_dropna_wheels.shape)
clean_df = df_dropna_wheels
clean_df.shape
# checking if anymore missing data exists
clean_df.isin(['unknown']).any()
clean_df.head()
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
corr = clean_df.corr()
corr.style.background_gradient(cmap='coolwarm')
# checking relation with categorical

le = preprocessing.LabelEncoder()
categorical = [        'Driven_Wheels', 'Transmission', 'Turbo', 'Fuel_Type',
       'PropSysDesign', 'Plugin', 'Registration_Type',"country_name"]
categorical_df = df.copy()
categorical_df.head()
mapping = {}
for i in categorical:
    mapping[i] = {i:n+1 for n,i in enumerate(categorical_df[i].unique())}
mapping
# string to numerical on orignial data
for col in categorical_df[categorical]:
    if categorical_df[col].dtypes=='object':
        categorical_df[col]=categorical_df[col].map(mapping[col])
categorical_df.head()
categorical_df.columns
relevant_columns = [
 'Driven_Wheels', 'Transmission', 'Turbo', 'Fuel_Type',
       'PropSysDesign', 'Plugin', 'Registration_Type',
       'Length', 'Height', 'Width', 'Engine_KW', 'No_of_Gears', 'Curb_Weight',
       'CO2', 'Fuel_cons_combined', 'country_name',
       'Price_USD']
corr = categorical_df[relevant_columns].corr()
corr.style.background_gradient(cmap='coolwarm')
sns.pairplot(categorical_df[relevant_columns])
# checking for feature relevance
X = categorical_df[relevant_columns[:-1]]
Y = categorical_df[relevant_columns[-1:]]
model = LinearRegression()
model.fit(X, Y)
importance = model.coef_
lr = OrderedDict(zip(relevant_columns,importance[0]))
print("linear regression: ",sorted(lr.items(), key=lambda x: x[1], reverse=True))
model = DecisionTreeRegressor()
model.fit(X, Y)
importance = model.feature_importances_
dt = OrderedDict(zip(relevant_columns,importance))
print("Regression tree: ",sorted(dt.items(), key=lambda x: x[1], reverse=True))
model = RandomForestRegressor()
model.fit(X, Y)
importance = model.feature_importances_
rf = OrderedDict(zip(relevant_columns,importance))
print("Random forest: ",[sorted(rf.items(), key=lambda x: x[1], reverse=True)])
model = XGBRegressor()
model.fit(X, Y)
importance = model.feature_importances_
xgb = OrderedDict(zip(relevant_columns,importance))
print("XG Boost: ",sorted(xgb.items(), key=lambda x: x[1], reverse=True))
model = KNeighborsRegressor()
# fit the model
model.fit(X, Y)
# perform permutation importance
results = permutation_importance(model, X, Y, scoring='neg_mean_squared_error')
# get importance
knn = OrderedDict(zip(relevant_columns,results.importances_mean))
print("knn: ",sorted(knn.items(), key=lambda x: x[1], reverse=True))
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - np.array(list(chain(*test_labels.values))))
    mape = 100 * np.mean(errors / np.array(list(chain(*test_labels.values))))
    accuracy = 100 - mape
    print('Model Performance')
#     print('Average Error: {} degrees.'.format(np.mean(errors)))
    print('Accuracy = {}%.'.format(accuracy))
    
    return accuracy
important_features_1 = ["Engine_KW",'CO2','Length', 'Height', 'Width','Price_USD']
X = categorical_df[important_features_1[:-1]]
Y = categorical_df[important_features_1[-1:]]
# ['Body_Type', 'Global_Sales_Sub-Segment', 'Brand',
#  'Driven_Wheels', 'Transmission', 'Turbo', 'Fuel_Type',
#        'PropSysDesign', 'Plugin', 'Registration_Type',
#        'Length', 'Height', 'Width', 'Engine_KW', 'No_of_Gears', 'Curb_Weight',
#        'CO2', 'Fuel_cons_combined', 'country_name',
#        'Price_USD']
# random forest wiht hyper parameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
trees = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
print("Best parameter",rf_random.best_params_)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
print("Accuracy:" ,random_accuracy)
categorical_df[relevant_columns].head()
# splitting data country wise to create separate models
usa = categorical_df[relevant_columns][categorical_df['country_name']==2]
print(usa.shape)
germany = categorical_df[relevant_columns][categorical_df['country_name']==1]
print(germany.shape)
china = categorical_df[relevant_columns][categorical_df['country_name']==0]
print(china.shape)
china.head()
X = china[important_features_1[:-1]]
Y = china[important_features_1[-1:]]

# random forest wiht hyper parameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
trees = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random_china = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random_china.fit(X_train, y_train)
print("Best parameter",rf_random_china.best_params_)

best_random_china_1 = rf_random_china.best_estimator_
random_accuracy_china = evaluate(best_random_china_1, X_test, y_test)
print("Accuracy:" ,random_accuracy_china)
china[relevant_columns].head()
# making china better by using all the relevant columns
X = china[relevant_columns[:-1]]
Y = china[relevant_columns[-1:]]

# random forest wiht hyper parameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
trees = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random_china = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random_china.fit(X_train, y_train)
print("Best parameter",rf_random_china.best_params_)

best_random_china_2 = rf_random_china.best_estimator_
random_accuracy_china = evaluate(best_random_china_2, X_test, y_test)
print("Accuracy:" ,random_accuracy_china)
## USING ONE HOT ENCODING ON df
print(df.columns)
print(relevant_columns)
print(categorical)
one_hot = pd.DataFrame()
for i in categorical:
    one_hot = pd.concat([one_hot,pd.get_dummies(df[i])], axis = 1)
print(one_hot.shape)
one_hot.head()
one_hot.columns = ['awd', 'front', 'rear', 'unknown1', 'automatic', 'manual', 'unknown2',
       'non turbo', 'turbo', 'unknown3', 'cng', 'diesel', 'electricity', 'gas',
       'gas-e', 'hydrogen', 'lpg', 'unknown4', 'electric', 'fuel cell',
       'hybrid-full', 'ice', 'mild-hybrid', 'unknown5', 'no', 'unknown6', 'yes',
       'Light Commercial Vehicles', 'Passenger Cars', 'China', 'Germany',
       'USA']
# combining numerical and onehot
one_hot_df = pd.concat([one_hot,df[[i for i in list(df.columns) if i not in categorical and i in relevant_columns] ]], axis=1)
one_hot_df.shape
one_hot_df.head()
# splitting data country wise to create separate models
usa_hot = one_hot_df[one_hot_df['USA']==1]
print(usa_hot.shape)
germany_hot = one_hot_df[one_hot_df['Germany']==1]
print(germany_hot.shape)
china_hot = one_hot_df[one_hot_df['China']==1]
print(china_hot.shape)
one_hot_df.head()
one_hot_df.columns
X = usa_hot[one_hot_df.columns[:-1]]
Y = usa_hot[one_hot_df.columns[-1:]]
X.head()
print(X.shape)
print(Y.shape)
X = usa_hot[one_hot_df.columns[:-1]]
Y = usa_hot[one_hot_df.columns[-1:]]

# random forest wiht hyper parameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
trees = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random_usa_hot = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
rf_random_usa_hot.fit(X_train, y_train)
print("Best parameter",rf_random_usa_hot.best_params_)

best_random_usa_hot = rf_random_usa_hot.best_estimator_
random_accuracy_usa_hot = evaluate(best_random_usa_hot, X_test, y_test)
print("Accuracy:" ,random_accuracy_usa_hot)
X = germany_hot[one_hot_df.columns[:-1]]
Y = germany_hot[one_hot_df.columns[-1:]]
print(X.shape)
print(Y.shape)
# important_features_u = ["Engine_KW",'CO2','Length', 'Height', 'Width','Price_USD']
X = germany_hot[one_hot_df.columns[:-1]]
Y = germany_hot[one_hot_df.columns[-1:]]

# random forest wiht hyper parameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
trees = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random_germany_hot = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random_germany_hot.fit(X_train, y_train)
print("Best parameter",rf_random_germany_hot.best_params_)

best_random_germany_hot = rf_random_germany_hot.best_estimator_
random_accuracy_germany_hot = evaluate(best_random_germany_hot, X_test, y_test)
print("Accuracy:" ,random_accuracy_germany_hot)
# important_features_u = ["Engine_KW",'CO2','Length', 'Height', 'Width','Price_USD']
X = china_hot[one_hot_df.columns[:-1]]
Y = china_hot[one_hot_df.columns[-1:]]

# random forest wiht hyper parameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
trees = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random_china_hot = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 10, verbose=2, random_state=42, n_jobs = -1)
rf_random_china_hot.fit(X_train, y_train)
print("Best parameter",rf_random_china_hot.best_params_)


best_random_china_hot = rf_random_china_hot.best_estimator_
random_accuracy_china_hot = evaluate(best_random_china_hot, X_test, y_test)
print("Accuracy:" ,random_accuracy_china_hot)



important_features_2 = ["Engine_KW",'CO2','Length', 'Height', 'Width',"country_name",'Price_USD']
X = categorical_df[important_features_1[:-1]]
Y = categorical_df[important_features_1[-1:]]
reg = LassoCV(cv=100, random_state=0).fit(X, Y)
reg.score(X, Y)
reg = LassoLarsCV(cv=10).fit(X, Y)
reg.score(X, Y)
regr = ElasticNetCV(cv=10, random_state=0)
regr.fit(X, Y)
print(regr.alpha_)
print(regr.intercept_)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, Y)
clf.score(X, Y)
reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X, Y)
reg.score(X, Y)
# predicting using model
oos = pd.read_csv(r"../input/ihsmarkit-hackathon-june2020/oos_data.csv")
oos.head()
china_relevant_oos = oos[oos["country_name"]=="China"][relevant_columns[:-1]+["Body_Type","Global_Sales_Sub-Segment","vehicle_id"]]
china_relevant_oos.shape
# string to numerical on orignial data
for col in china_relevant_oos[categorical+["Body_Type","Global_Sales_Sub-Segment"]]:
    if china_relevant_oos[col].dtypes=='object':
        china_relevant_oos[col]=china_relevant_oos[col].map(mapping[col])
china_relevant_oos.head()
print(china_relevant_oos.shape)
china_relevant_oos["vehicle_id"] = list(zip(china_relevant_oos["vehicle_id"], china_relevant_oos.index))
china_relevant_oos.index= range(china_relevant_oos.shape[0])
china_relevant_oos.head()
china_d = {}
for i in china_relevant_oos.index:
    china_d[china_relevant_oos.iloc[i].values[-1]]=best_random_china_2.predict([list(china_relevant_oos.iloc[i].values[:-1])])[0]
china_output = pd.DataFrame.from_dict(china_d,orient="index")
china_output
china_output["vehicle_id"] =[i[0] for i in  china_output.index]
china_output.index = [i[1] for i in  china_output.index]
china_output.columns = ["Price_USD","vehicle_id" ]
china_output = china_output[["vehicle_id","Price_USD"]]
china_output.head()
oos_hot = pd.DataFrame()
for i in categorical:
    oos_hot = pd.concat([oos_hot,pd.get_dummies(oos[i])], axis = 1)
print(oos_hot.shape)
oos_hot.head()
oos_hot_df = pd.concat([oos_hot,oos[[i for i in list(oos.columns) if i not in categorical and i in relevant_columns] ]], axis=1)
oos_hot_df.shape
print(one_hot_df.shape)
one_hot_df.head()
oos_hot_df.columns = ['awd', 'front', 'rear', 'unknown1', 'automatic', 'manual', 'unknown2',
       'non turbo', 'turbo', 'unknown3', 'cng', 'diesel', 'electricity', 'gas',
       'gas-e', 'hydrogen', 'lpg', 'unknown4', 'electric', 'fuel cell',
       'hybrid-full', 'ice', 'mild-hybrid', 'unknown5', 'no', 'unknown6', 'yes',
       'Light Commercial Vehicles', 'Passenger Cars', 'China', 'Germany',
       'USA', 'Length', 'Height', 'Width', 'Engine_KW', 'No_of_Gears',
       'Curb_Weight', 'CO2', 'Fuel_cons_combined']
oos_hot_df.head()
germany_relevant_oos = oos_hot_df[oos_hot_df["Germany"]==1]
germany_relevant_oos["vehicle_id"] = oos[oos["country_name"]=="Germany"]["vehicle_id"]
print(germany_relevant_oos.shape)
germany_relevant_oos.head()
germany_relevant_oos["vehicle_id"] = list(zip(germany_relevant_oos["vehicle_id"], germany_relevant_oos.index))
germany_relevant_oos.index= range(germany_relevant_oos.shape[0])
germany_relevant_oos.head()
# for all the rows:
# for i in germany_relevant_oos.index:
#         print(best_random_germany_hot.predict(list(germany_relevant_oos[germany_relevant_oos.columns[:-1]])))
germany_relevant_oos.index = range(germany_relevant_oos.shape[0])
germany_d = {}
for i in germany_relevant_oos.index:
#     print([list(germany_relevant_oos.iloc[i].values[:-1])])
    germany_d[germany_relevant_oos.iloc[i].values[-1]]=best_random_germany_hot.predict([list(germany_relevant_oos.iloc[i].values[:-1])])[0]
germany_output = pd.DataFrame.from_dict(germany_d,orient="index")
germany_output
germany_output["vehicle_id"] =[i[0] for i in  germany_output.index]
germany_output.index = [i[1] for i in  germany_output.index]
germany_output.columns = ["Price_USD","vehicle_id" ]
germany_output = germany_output[["vehicle_id","Price_USD"]]
germany_output.head()
usa_relevant_oos = oos_hot_df[oos_hot_df["USA"]==1]
usa_relevant_oos["vehicle_id"] = oos[oos["country_name"]=="USA"]["vehicle_id"]
print(usa_relevant_oos.shape)
usa_relevant_oos.head()
usa_relevant_oos["vehicle_id"] = list(zip(usa_relevant_oos["vehicle_id"], usa_relevant_oos.index))
usa_relevant_oos.index= range(usa_relevant_oos.shape[0])
usa_relevant_oos.head()
usa_d = {}
for i in usa_relevant_oos.index:
    usa_d[usa_relevant_oos.iloc[i].values[-1]]=best_random_usa_hot.predict([list(usa_relevant_oos.iloc[i].values[:-1])])[0]
usa_output = pd.DataFrame.from_dict(usa_d,orient="index")
usa_output.head()
usa_output["vehicle_id"] =[i[0] for i in  usa_output.index]
usa_output.index = [i[1] for i in  usa_output.index]
usa_output.columns = ["Price_USD","vehicle_id" ]
usa_output =usa_output[["vehicle_id","Price_USD"]]
usa_output.head()
final_output = pd.concat([china_output, usa_output, germany_output], axis=0)
final_output = final_output.sort_index()
final_output.head()
final_output.to_csv('submission.csv', index=False)
# adding yoy change
# adding brand weight