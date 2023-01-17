import numpy as np

import pandas as pd



# Input data files are available in the "../input/competition" directory.
df = pd.read_csv("../input/pk-hska/agegroups_train.csv")

df.head()
features = ['weekday_number', 'region_number', 'sum_sales_pg1', 'sum_sales_pg2', 'sum_sales_pg3']

target = 'age_group_number'
X = df[features]

y = df[target]
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier()  # create random forest classifier for grid search



param_grid = [

    {'max_depth': range(5, 21, 5),  # 5, 10, 15, 20 maximum depth

     'n_estimators' : range(50, 101, 50)}  # 50, 100 number of trees

]



search = GridSearchCV(clf, param_grid, cv=3)  # search with cv-factor 3



search.fit(X, y)  # start the grid search



# get the best model's parameters

print("Best parameter (CV score={:.2f}):{})".format(search.best_score_, search.best_params_))
# read the provided file that is the basis of a submission

df_test = pd.read_csv("../input/pk-hska/agegroups_test.csv").set_index('index')  # important - setting the index!

df_test.head()
# create predictions 

prediction = search.predict(df_test)

prediction
# set the prediction column to crowded (see sample-submission csv file)

df_test['age_group_number'] = prediction
df_test.head()
df_submission = df_test[['age_group_number']]  # important - only provide the target column

df_submission.to_csv('submission.csv', index=True, index_label='index')  # providing the index column is important!