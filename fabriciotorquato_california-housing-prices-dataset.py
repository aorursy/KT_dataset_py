%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import RandomizedSearchCV
data = pd.read_csv("../input/housing-raw-data/housing_raw.csv")

data.head()
data.info()
data.describe()
col = data.columns       # .columns gives columns names in data 

print(col)
data["ocean_proximity"].value_counts()
data["total_bedrooms"].isna().sum()
data.hist(bins=50, figsize=(20,15))
data['ocean_proximity'].value_counts().plot(kind='bar')
for column in data.columns:

    if data[column].dtype == np.float64:

        plt.figure(figsize = (20, 3))

        ax = sns.boxplot(x = data[column])
def remove_outleirs(dataframe, column_name, threshold):

    outliers_index = (dataframe[column_name] > threshold).values.nonzero()

    

    return dataframe.drop(labels = outliers_index[0], axis = 0)
clean_data = remove_outleirs(data, column_name = "housing_median_age", threshold = 100)
plt.figure(figsize = (20, 3))

ax = sns.boxplot(x = clean_data["housing_median_age"])
def remove_duplicates(dataframe):

    duplicated_indexes = dataframe.duplicated(keep = "first")

    return dataframe[~duplicated_indexes]
print("duplicated =>", clean_data.duplicated(keep = "first").sum())
clean_data = remove_duplicates(clean_data)
print("duplicated =>", clean_data.duplicated(keep = "first").sum())
def remove_inconsistencies(dataframe, columns):

    inconsistent_indexes = dataframe.duplicated(subset = columns, keep = False)

    return dataframe[~inconsistent_indexes]
features_columns = list(clean_data.columns)

features_columns.remove('median_house_value')
print("Inconsistency =>", clean_data.duplicated(subset = features_columns, keep = False).sum())
indexes = clean_data.duplicated(subset = features_columns, keep = False).values.nonzero()

clean_data.iloc[indexes]
clean_data = remove_inconsistencies(clean_data, columns = features_columns)
print("Inconsistency =>", clean_data.duplicated(subset = features_columns, keep = False).sum())
X = clean_data.drop("median_house_value", axis=1)

y = clean_data["median_house_value"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
data_filler = SimpleImputer(strategy="median")

numerical_data = X_train.drop("ocean_proximity", axis=1)

data_filler.fit(numerical_data)

np.set_printoptions(suppress=True)

print(data_filler.statistics_)
X_train.median()
X_train_process = data_filler.transform(numerical_data)
print( np.isnan(np.sum(X_train_process)) )
ocean_proximity = X_train["ocean_proximity"].values

ocean_proximity = ocean_proximity.reshape(-1, 1)

ocean_proximity
ordinal_encoder = OrdinalEncoder()

ocean_proximity_encoded = ordinal_encoder.fit_transform(ocean_proximity)
viz_data = X_train.copy()

viz_data["ocean_proximity_ordinal"] = ocean_proximity_encoded

viz_data
print(ordinal_encoder.categories_)
one_hot_encoder = OneHotEncoder()

ocean_proximity_hot_encoded = one_hot_encoder.fit_transform(ocean_proximity)
viz_data = X_train.copy()

viz_data["<1H OCEAN"] = ocean_proximity_hot_encoded.toarray()[:, 0]

viz_data["INLAND"] = ocean_proximity_hot_encoded.toarray()[:, 1]

viz_data["ISLAND"] = ocean_proximity_hot_encoded.toarray()[:, 2]

viz_data["NEAR BAY"] = ocean_proximity_hot_encoded.toarray()[:, 3]

viz_data["NEAR OCEAN"] = ocean_proximity_hot_encoded.toarray()[:, 4]

viz_data
print(one_hot_encoder.categories_)
numerical_columns = list(numerical_data)

categorical_columns = ["ocean_proximity"]
numerical_pipeline = Pipeline([

        ('data_filler', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])



X_train_numerical = numerical_pipeline.fit_transform(numerical_data)
pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])
def rmse_r2(model,y,y_predict):    

    rmse = (np.sqrt(mean_squared_error(y, y_predict)))

    r2 = r2_score(y, y_predict)

    print('RMSE is {}'.format(rmse))

    print('R2 score is {}'.format(r2))
def score_rmse_r2(model,X,y):    

    rmse = cross_val_score(clf, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()

    r2 = cross_val_score(clf, X, y, cv=5, scoring='r2').mean()

    print('RMSE is {}'.format(rmse))

    print('R2 score is {}'.format(r2))
def get_predict(model,X_train,y_train,X_test,y_test):

    print("\nThe model performance for training set")

    print("--------------------------------------")

    y_predict = model.predict(X_train)

    rmse_r2(model,y_train,y_predict)

    print("\nThe model performance for testing set")

    print("--------------------------------------")

    y_predict = model.predict(X_test)

    rmse_r2(model,y_test,y_predict)
def get_score_predict(model,X_train,y_train,X_test,y_test):

    print("\nThe model performance for training set")

    print("--------------------------------------")

    score_rmse_r2(model,X_train,y_train)

    print("\nThe model performance for validation set")

    print("--------------------------------------")

    score_rmse_r2(model,X_test,y_test)
def test_score(model,X,y):

    print("\nThe model performance for testing set")

    print("--------------------------------------")

    score_rmse_r2(model,X,y)
def get_model_grid_search(model, parameters, X, y, pipeline):

    

    X = pipeline.fit_transform(X)    

    

    random_search = RandomizedSearchCV(model,

                            param_distributions=parameters,

                            scoring='r2',

                            verbose=1, n_jobs=-1,

                            n_iter=1000)

    

    grid_result = random_search.fit(X, y)

    

    print('Best R2: ', grid_result.best_score_)

    print('Best Params: ', grid_result.best_params_)  

    

    return random_search.best_estimator_
def get_model_random_search(model, parameters, X, y, pipeline):

    

    X = pipeline.fit_transform(X)    

    clf = GridSearchCV(model, parameters, scoring='r2',cv=5,verbose=1, n_jobs=-1)

    grid_result = clf.fit(X, y)

    

    print('Best R2: ', grid_result.best_score_)

    print('Best Params: ', grid_result.best_params_)  

    

    return clf.best_estimator_
def k_fold_score(model, X ,y):

    kf = KFold(n_splits = 5)

    rmse_list = []

    r2_list = []

    for train_index, test_index in kf.split(X, y):

        X_train,X_test = X.iloc[train_index],X.iloc[test_index]

        y_train,y_test = y.iloc[train_index],y.iloc[test_index]



        X_train = pipeline.fit_transform(X_train)

        X_test = pipeline.transform(X_test)

        

        model.fit(X_train,y_train)

        y_predict = model.predict(X_test)



        rmse = (np.sqrt(mean_squared_error(y_test, y_predict)))

        r2 = r2_score(y_test, y_predict)

        rmse_list.append(rmse)

        r2_list.append(r2)





    rmse_list = np.array(rmse_list)

    r2_list = np.array(r2_list)



    print("--------------------------------------")

    print('RMSE is {}'.format(rmse_list.mean()))

    print('R2 score is {}'.format(r2_list.mean()))
X_train = pipeline.fit_transform(X_train)

X_test = pipeline.transform(X_test)
data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.95, random_state=42)
lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

get_predict(lin_model,X_train,y_train,X_test,y_test)
params = {

    'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],       

    'l1_ratio':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],

}



en = ElasticNet()



pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])



en_model = get_model_grid_search(en, params, data_gs, target_gs, pipeline)
k_fold_score(en_model,data_cv, target_cv)
params = {

    'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],       

    'l1_ratio':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],

}



en = ElasticNet()



pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])



en_rs_model = get_model_random_search(en, params, data_gs, target_gs, pipeline)
k_fold_score(en_rs_model, data_cv, target_cv)
svr = SVR(kernel='rbf')

svr.fit(X_train, y_train)

get_predict(svr,X_train,y_train,X_test,y_test)
params = {  'C': [0.1, 1, 100, 1000],

            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        }



svr = SVR(kernel='rbf')



pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])



svr_model = get_model_grid_search(svr, params, data_gs, target_gs, pipeline)
k_fold_score(svr_model,data_cv, target_cv)
params = {  'C': [0.1, 1, 100, 1000],

            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],

            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]

        }



svr = SVR(kernel='rbf')



pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])



svr_rs_model = get_model_random_search(svr, params, data_gs, target_gs, pipeline)
k_fold_score(svr_rs_model, data_cv, target_cv)
tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)

get_predict(tree, X_train, y_train, X_test, y_test)
params = {'min_samples_split': range(2, 10)}



tree = DecisionTreeRegressor()



pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])



tree_model = get_model_grid_search(tree, params, data_gs, target_gs, pipeline)
k_fold_score(tree_model, data_cv, target_cv)
params = {'min_samples_split': range(2, 10)}



tree = DecisionTreeRegressor()



pipeline = ColumnTransformer([

        ("numerical", numerical_pipeline, numerical_columns),

        ("categorical", OneHotEncoder(), categorical_columns),

    ])



tree_rs_model = get_model_random_search(tree, params, data_gs, target_gs, pipeline)
k_fold_score(tree_rs_model, data_cv, target_cv)