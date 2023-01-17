print('''



Ask a home buyer to describe their dream house, 

and they probably won't begin with the height

of the basement ceiling or the proximity to

an east-west railroad. But this playground competition's dataset 

proves that much more influences price negotiations than 

the number of bedrooms or a white-picket fence.



With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.



It is your job to predict the sales price for each house. 

For each Id in the test set, you must predict the value of the SalePrice variable. 



Metric: 

Submissions are evaluated on Root-Mean-Squared-Error (RMSE)

between the logarithm of the predicted value and the logarithm

of the observed sales price. 



(Taking logs means that errors in predicting expensive houses 

and cheap houses will affect the result equally.)



''')
import pandas as pd

# import the training and holdout data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = train.append(test)
import numpy as np

logtransform = ['SalePrice']

for i in logtransform: 

    data[i] = np.log(np.array(data[i]))
# CATBOOST FEATURE PREP



data = data[data['SalePrice'].notna()]

data.fillna(-999, inplace = True)

X = data.drop(['SalePrice', 'Id'], axis =1)

y = data['SalePrice']

cat_features = np.where(X.dtypes == 'object')[0]



# TRAIN TEST SPLIT



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42)

from catboost import Pool, CatBoostRegressor, cv



train_pool = Pool(X_train, y_train, cat_features = cat_features)

test_pool = Pool(X_test, y_test, cat_features = cat_features)
# TRAIN TEST SPLIT



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=42)
from catboost import Pool, CatBoostRegressor

# more on MAE - https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

cat_model = CatBoostRegressor(

    loss_function='MAE',

    random_seed = 15,

    eval_metric = 'RMSE',

    verbose = 300, 

    max_ctr_complexity=15, 

    iterations = 1500

     )



cat_model.fit(

    train_pool,

    eval_set = test_pool, 

    use_best_model = True

    )
print('''

RMSE has been minimised to 0.1345 and R-Square of Test and Train datasets is above 0.90. 

what worked - > 

1. using Catboost to allow handling of multiple categorical features

2. allowing Catboost to explore more permutations of categorical features

3. using MAE instead of RMSE/R2 in Loss function. MAE helps model ignore influence of outliers. 

All in all, House Prices are easy to predict if we are given such rich data.

''')