# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
housing=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
plt.figure(figsize=(10,7))

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

        s=housing["population"]/100, label="population", figsize=(15,8),

        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,

    )

plt.legend

corr_matrix=housing.corr()

print(corr_matrix)

corr_matrix["median_house_value"].sort_values(ascending=False)
housing.hist(bins=50,figsize=(20,15))

housing.hist(bins=500,figsize=(20,15))
#combining 2 attributes

housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]

corr_matrix=housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)



#splitting data

from sklearn.model_selection import train_test_split





median=housing["total_bedrooms"].median()

housing["total_bedrooms"].fillna(median,inplace=True)



housing_test=housing.drop("median_house_value",axis=1)

housing_labels_test= housing["median_house_value"].copy()





#housing= strat_train_set.drop("median_house_value",axis=1)

#housing_labels= strat_train_set["median_house_value"].copy()





X_train, X_test, y_train, y_test = train_test_split(housing_test, housing_labels_test, 

                                    test_size=0.2, random_state=3)



#housing.dropna(subset=["total_bedrooms"])

#housing.drop("total_bedrooms",axis=1)





#SimpleImputer

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy="mean")

housing_num=housing_test.drop("ocean_proximity",axis=1)

imputer.fit(housing_num)



print(imputer.statistics_)

print()



print(housing_num.mean().values)

print()



x=imputer.transform(housing_num)

print(x)
#OrdinalEncoder

housing_cat=housing_test[["ocean_proximity"]]

print(housing_cat[900:1000])



from sklearn.preprocessing import OrdinalEncoder



ordinal_encoder=OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(housing_cat_encoded[900:1000])



print(ordinal_encoder.categories_)

#OneHotEncoder

housing_cat=housing_test[["ocean_proximity"]]



from sklearn.preprocessing import OneHotEncoder



cat_encoder=OneHotEncoder()

housing_cat_1hot=cat_encoder.fit_transform(housing_cat)

print(housing_cat_1hot[900:1000])



housing_cat_1hot.toarray()
#Pipeline

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



print(housing_num[990:1000])





num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),

                        ('std_scalar',StandardScaler())

                        ])





#housing_num_tr =num_pipeline.fit_transform(housing_num)

#print(housing_num_tr[990:1000])



#Full Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



#print(housing[990:1000])



num_attribs=list(housing_num)

cat_attribs=["ocean_proximity"]



full_pipeline=ColumnTransformer([

    ("num",num_pipeline,num_attribs),

    ("cat",OneHotEncoder(),cat_attribs)

])

housing_prepared=full_pipeline.fit_transform(housing_test)

#print()

#print()



print(housing_prepared[990:1000])
#linear regression NOT COMPELTE 





#from sklearn.linear_model import LinearRegression



#lin_reg=LinearRegression()

#lin_reg.fit(housing_prepared,housing_labels_test)



#some_data=housing_test.iloc[:5]

#some_labels=housing_labels_test.iloc[:5]

#some_data_prepared=full_pipeline.transform(some_data)



#print("predictions: ",lin_reg.predict(some_data_prepared))

#print()

#print("labels: ",list(some_labels))



#linear x mean_squared_error

#linear cross validation



#from sklearn.metrics import mean_squared_error

#from sklearn.model_selection import cross_val_score





#lin_mse=cross_val_score(lin_reg,housing_prepared,housing_labels_test,

#                        scoring="neg_mean_squared_error",cv=10)



#lin_rmse=np.sqrt(-lin_mse)

#print(lin_rmse)



#print()

#print("mean:",lin_rmse.mean())

#print("Std",lin_rmse.std())



#grid searchCV

#from sklearn.model_selection import GridSearchCV

#print(LinearRegression())



#param_grid=['copy_X':[True,False],'fit_intercept':[True]]



#grid_search=GridSearchCV(lin_reg,param_grid,cv=10,

 #                       scoring='neg_mean_squared_error',

  #                       verbose=2

   #                     )



#grid_search.fit(housing_prepared,housing_labels_test)
#random forest regressor



from sklearn.ensemble import RandomForestRegressor



forest_reg=RandomForestRegressor()

#forest_reg.fit(housing_prepared,housing_labels_test)



#linear x mean_squared_error

#linear cross validation



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



forest_mse=cross_val_score(forest_reg,housing_prepared,housing_labels_test,

                          scoring="neg_mean_squared_error",cv=10)



forest_rmse=np.sqrt(-forest_mse)



print(forest_rmse)

print("Mean: ",forest_rmse.mean())

print("Std: ",forest_rmse.std())

print(RandomForestRegressor())
#grid searchCV

from sklearn.model_selection import GridSearchCV



param_grid=[

    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},

    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}

]

forest_reg=RandomForestRegressor()

grid_search=GridSearchCV(forest_reg,param_grid,cv=10,

                        scoring='neg_mean_squared_error',

                        return_train_score=True)

grid_search.fit(housing_prepared,housing_labels_test)
grid_search.best_params_
cvres=grid_search.cv_results_

for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):

    print(np.sqrt(-mean_score),params)
final_model=grid_search.best_estimator_

print(final_model)    
X_test=housing.drop("median_house_value",axis=1)

#X_test= strat_train_set.drop("median_house_value",axis=1)

Y_test= housing["median_house_value"].copy()

#Y_test= strat_train_set["median_house_value"].copy()



X_test_prepared=full_pipeline.transform(X_test)

final_predictions=final_model.predict(X_test_prepared)

final_mse=mean_squared_error(Y_test,final_predictions)

final_rmse=np.sqrt(final_mse)

print(final_rmse)



#20178.137512754445 when text_size=0.1 and random state = 2

#19407.958425401335 when text_size=0.2 and random state = 2

#19199.65264789018 when text_size=0.3 and random state = 2

#18796.357362853116 when text_size=0.1 and random state = 10

#19184.5860840378 when text_size=0.2 and random state=42
test = pd.DataFrame({'Predicted':final_predictions,'Actual':Y_test})

fig= plt.figure(figsize=(16,8))

test = test.reset_index()

test = test.drop(['index'],axis=1)

plt.plot(test[:50])

plt.legend(['Actual','Predicted'])