import pandas as pd
import numpy as np

from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from nose.tools import *
world_food_data=pd.read_csv("../input/openfoodfactsclean/world_food_scrubbed.csv")
assert_is_not_none(world_food_data)
assert_equal(world_food_data.shape,(71091,13))
world_food_data.head()
world_food_data.drop(columns=["product_name","packaging","additives_n","fp_lat","fp_lon"],inplace=True)
assert_equal(world_food_data.shape[1],8)
world_food_data.head()
world_food_data_for_modelling=pd.get_dummies(world_food_data)
world_food_data_features=world_food_data_for_modelling.drop(columns=["contains_additives"])
world_food_data_target=world_food_data.contains_additives
world_food_data_for_modelling.shape[1]
assert_equal(world_food_data_for_modelling.shape[1],1935)
assert_equal(world_food_data_features.shape[1],1934)
assert_equal(world_food_data_target.shape,(71091,))
scaler=MaxAbsScaler()
world_food_data_features_scaled = scaler.fit_transform(world_food_data_features)
assert_is_not_none(world_food_data_features_scaled)
print(world_food_data_features_scaled)
features_train, features_test, target_train, target_test = train_test_split(
    world_food_data_features_scaled, world_food_data_target, train_size = 0.7, test_size = 0.3, random_state = 42)
print("Training data shapes: Features:{}, Labels:{}".format(features_train.shape,target_train.shape))
print("Test data shapes: Features:{}, Labels:{}".format(features_test.shape,target_test.shape))
model=LogisticRegression()
model.fit(features_train,target_train)
assert_is_not_none(model)
score = model.score(features_test,target_test)
print("Additives prediction accuracy: {:.2f}".format(score*100))
assert_greater(score,0.5)
assert_less_equal(score,1)