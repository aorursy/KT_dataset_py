# setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.ml_explainability.ex3 import *

print("Setup is completed.")



# load data

import pandas as pd

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)



# remove data with extreme outlier coordinates or negative fares

data = data.query(

    'pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +

    'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +

    'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +

    'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +

    'fare_amount > 0'

)

y = data['fare_amount']



# select features

base_features = [

    'pickup_longitude',

    'pickup_latitude',

    'dropoff_longitude',

    'dropoff_latitude'

]

X = data[base_features]



# split train and test data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# define and fit model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)



print("Data sample:")

data.head()
data.describe()
data.query('fare_amount > 100 and fare_amount < 165').head()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



feat_name = 'pickup_longitude'

pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
%%javascript

IPython.OutputArea.auto_scroll_threshold = 9999;
for feat_name in base_features:

    pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
# check your answer (tun this code cell to receive credit!)

q_1.solution()
# add your code here

features_to_plot = ["pickup_longitude", "dropoff_longitude"]



# create data

pdp_inter = pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=features_to_plot)



# plot it

pdp.pdp_interact_plot(pdp_interact_out=pdp_inter, feature_names=features_to_plot, plot_type="contour")

plt.show()
# check your answer (Run this code cell to receive credit!)

q_2.solution()
savings_from_shorter_trip = 24 - 9



# check your answer

q_3.check()
# for a solution or hint, uncomment the appropriate line below

# q_3.hint()

# q_3.solution()
# this is the PDP for pickup_longitude without the absolute difference features, included to easy compare

feat_name = 'pickup_longitude'

pdp_dist_original = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist_original, feat_name)

plt.show()





# create new features

data['abs_lon_change'] = abs(data['pickup_longitude'] - data['dropoff_longitude'])

data['abs_lat_change'] = abs(data['pickup_latitude'] - data['dropoff_latitude'])



features_2  = [

    'pickup_longitude',

    'pickup_latitude',

    'dropoff_longitude',

    'dropoff_latitude',

    'abs_lat_change',

    'abs_lon_change']

X = data[features_2]



new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)



second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)



feat_name = 'pickup_longitude'

pdp_dist = pdp.pdp_isolate(model=second_model, dataset=new_val_X, model_features=features_2, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()



# check your answer

q_4.check()
# uncomment the lines below to see a hint or the solution (including an explanation of the important differences between the plots).

# q_4.hint()

q_4.solution()
# check your answer (run this code cell to receive credit!)

q_5.solution()
import numpy as np

from numpy.random import rand



n_samples = 20000



# create array holding predictive feature

X1 = 4 * rand(n_samples) - 2

X2 = 4 * rand(n_samples) - 2



# create y. you should have X1 and X2 in the expression for y

y = (-2 * X1 * (X1<-1)) + X1 + (-2 * X1 * (X1>1)) + X2



# create dataframe because pdp_isolate expects a dataFrame as an argument

my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

predictors_df = my_df.drop(['y'], axis=1)



my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df['y'])



pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')



# visualize your results

pdp.pdp_plot(pdp_dist, 'X1')

plt.show()



# check your answer

q_6.check()
# uncomment the lines below for a hint or solution

# q_6.hint()

# q_6.solution()
n_samples = 20000



# create array holding predictive feature

X1 = 4 * rand(n_samples) - 2

X2 = 4 * rand(n_samples) - 2

# create y. you should have X1 and X2 in the expression for y

y = X1 * X2



# create dataframe because pdp_isolate expects a dataFrame as an argument

my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

predictors_df = my_df.drop(['y'], axis=1)



my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df['y'])



pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')



# visualize

pdp.pdp_plot(pdp_dist, 'X1')

plt.show()



# calculate PIs

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model).fit(predictors_df, my_df.y)



# check your answer

q_7.check()



# show the weights for the calculated PIs

from eli5 import show_weights

show_weights(perm, feature_names = ['X1', 'X2'])
# uncomment the following lines for the hint or solution

# q_7.hint()

# q_7.solution()