import pandas as pd

import numpy as np

import scipy.stats as stats



import matplotlib.pyplot as plt

import seaborn as sns

import shap

import eli5

from collections import Counter



import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

%matplotlib inline



shap.initjs()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/usedcarscatalog/cars.csv')

df.shape
df = df.loc[df['model_name']=='Passat']

df.shape
df.price_usd.mean()
from sklearn.model_selection import train_test_split 



X = df.drop('price_usd', axis=1)

y = df['price_usd']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Number of cars in X_train dataset: ", X_train.shape) 

print("Number of cars in y_train dataset: ", y_train.shape) 

print("Number of cars in X_test dataset: ", X_test.shape) 

print("Number of cars in y_test dataset: ", y_test.shape)
%%time

# create train_pool object

from catboost import CatBoostRegressor

from catboost import Pool

from catboost import MetricVisualizer







cat_features=['manufacturer_name', 

              'model_name', 

              'transmission', 

              'color', 

              'engine_fuel',

              'engine_has_gas',

              'engine_type', 

              'body_type', 

              'has_warranty', 

              'state', 

              'drivetrain',

              'is_exchangeable', 

              'location_region',

              'feature_0',

              'feature_1',

              'feature_2',

              'feature_3',

              'feature_4',

              'feature_5',

              'feature_6',

              'feature_7',

              'feature_8',

              'feature_9',]



train_pool = Pool(

    data=X_train, 

    label=y_train,

    cat_features = cat_features

)



# create validation_pool object

validation_pool = Pool(

    data=X_test, 

    label=y_test,

    cat_features = cat_features

)
%%time



# pretty basic model, max_depth=10 give slightly better results

cbs = CatBoostRegressor(iterations=4000,

                         learning_rate=0.012,

                         loss_function='MAE',

                         max_depth=10, 

                         early_stopping_rounds=200,

                         cat_features = cat_features)



# we are passing categorical features as parameters here

cbs.fit(

    train_pool,

    eval_set=validation_pool,

    verbose=False,

    plot=True 

);
error = test_predictions - y_test

# print(type(error))



plt.figure(figsize=(10,10))

plt.scatter(y_test, 

            test_predictions, 

            c=error,

            s=2,

            cmap='hsv',

            )

plt.colorbar()

plt.xlabel('True Values [price_usd]')

plt.ylabel('Predictions [price_usd]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0, 20000])

plt.ylim([0, 20000])

plt.show()
plt.figure(figsize=(16,7))

plt.hist(error, bins = 40, rwidth=0.9)

plt.xlabel('Predictions Error [price_usd]')

_ = plt.ylabel('Count')

plt.xlim([-6000, 6000])

plt.show()
%%time



importance_types = ['PredictionValuesChange',

                    'LossFunctionChange'

                   ]





for importance_type in importance_types:

    print(importance_type)

    print(cbs.get_feature_importance(data=train_pool, 

                                     type=importance_type))

    print('\n\n\n\n')
%%time



import shap

shap.initjs()



shap_values = cbs.get_feature_importance(Pool(X_test, 

                                              label=y_test,

                                              cat_features=cat_features), 

                                         type="ShapValues")

print(type(shap_values))



expected_value = shap_values[0,-1]

print(expected_value)



shap_values = shap_values[:,:-1]
shap.summary_plot(shap_values, X_test, max_display=X_test.shape[1])
shap.dependence_plot(ind='year_produced', interaction_index='year_produced',

                     shap_values=shap_values, 

                     features=X_test,  

                     display_features=X_test)
shap.dependence_plot(ind='odometer_value', interaction_index='odometer_value',

                     shap_values=shap_values, 

                     features=X_test,  

                     display_features=X_test)
shap.dependence_plot(ind='engine_capacity', interaction_index='engine_capacity',

                     shap_values=shap_values, 

                     features=X_test,  

                     display_features=X_test)
shap.dependence_plot(ind='number_of_photos', interaction_index='number_of_photos',

                     shap_values=shap_values, 

                     features=X_test,  

                     display_features=X_test)


shap.force_plot(expected_value, shap_values[:1000,:], X_test.iloc[:1000,:])
for i in range(50,70):

    print('Sample', i, 'from the test set:')

    display(shap.force_plot(expected_value, shap_values[i,:], X_test.iloc[i,:]))

    print('Listed_price -------------------------------------->', y_test.iloc[i])

    print('parameters:\n', X_test.iloc[i,:])

    print('\n\n\n\n\n\n\n')