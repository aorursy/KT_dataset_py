# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
housing = pd.read_csv("../input/train.csv")

housing = housing[["median_income", "median_house_value"]]

housing.head()




import numpy as np



def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



# train : test = 80 : 20

train_set, test_set = split_train_test(housing, 0.2)

print("train: {}, test: {}".format(len(train_set), len(test_set)))



# 훈련 데이터 복사

housing = train_set.copy()





# 훈련 데이터 복사

housing = train_set.drop("median_house_value", axis=1)

housing_labels = train_set["median_house_value"].copy()

housing_prepared = housing



#from sklearn.pipeline import Pipeline

#from sklearn.preprocessing import StandardScaler

#

#num_pipeline = Pipeline([

#    ('imputer', Imputer(strategy="median")),

#    ('scaler', StandardScaler()),

#])

#housing_num_tr = num_pipeline.fit_transform(housing_num)







from sklearn.metrics import mean_squared_error





# Decision Tree

from sklearn.tree import DecisionTreeRegressor





from sklearn.model_selection import cross_val_score





# Random Forest

from sklearn.ensemble import RandomForestRegressor





from sklearn.model_selection import GridSearchCV

# 파라미터 조합

param_grid = [

    {'n_estimators': [3,10,30], 'max_features': [1]},

    {'bootstrap': [False], 'n_estimators': [3, 10],

                                'max_features': [1]}

]

forest_reg = RandomForestRegressor()

# grid search

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                          scoring='neg_mean_squared_error',

                          return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_



# 최적값 확인

grid_search.best_estimator_



# score 확인

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],

                             cvres["params"]):

    print(np.sqrt(-mean_score), params)



"""#### 최종 모델 평가"""



# 최종 모델 선택

final_model = grid_search.best_estimator_



# test dataset 으로 평가

X_test = test_set.drop("median_house_value", axis=1)

y_test = test_set["median_house_value"].copy()



# 전처리

X_test_prepared = X_test #full_pipeline.transform(X_test)

# 최종 예측

final_predict = final_model.predict(X_test_prepared)

# 평가

final_mse = mean_squared_error(y_test, final_predict)

final_rmse = np.sqrt(final_mse)

final_rmse





# read test data

df_test = pd.read_csv("../input/test.csv")



# 전처리

#df_test = df_test.drop("id", axis=1)

#df_test["income_cat"] = np.ceil(housing["median_income"] / 1.5)

df_test = df_test[["median_income"]]

X_test = df_test #full_pipeline.transform(df_test)

#df_test.head()

#

#X_test



# 최종 예측

final_predict = final_model.predict(X_test)



#final_predict.shape



submission = pd.read_csv("../input/submitSample.csv")

#submission.head()



submission["median_house_value"] = final_predict

#submission.head()



submission.to_csv('MySubmission.csv', index=False)

!ls -al