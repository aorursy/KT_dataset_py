# Loading data, dividing, modeling and EDA below

import pandas as pd

import numpy as np

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score

#from sklearn.inspection import permutation_importance



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





data = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)



# Remove data with extreme outlier coordinates or negative fares

data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +

                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +

                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +

                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +

                  'fare_amount > 0'

                  )



y = data.fare_amount



base_features = ['pickup_longitude',

                 'pickup_latitude',

                 'dropoff_longitude',

                 'dropoff_latitude',

                 'passenger_count']



X = data[base_features]





train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)





# Using tuned parameters.

reg = LGBMRegressor(colsample_bytree = 0.8,max_depth = 3,min_child_weight=0.1,subsample=0.6, # Tuned hyperparameter

                    importance_type='gain', # Use importance type='gain' (Note default option is 'split')

                    random_state=42)

reg.fit(train_X, train_y)



print(r2_score(train_y,reg.predict(train_X))) # 0.40388748819243725

print(r2_score(val_y,reg.predict(val_X))) # 0.3464450279784468





data.head()
train_X.describe()
train_y.describe()
# ELI5 ist ein Python-Paket, mit dem Klassifizierer für maschinelles Lernen debuggt und ihre Vorhersagen erläutert werden können.

import eli5

from eli5.sklearn import PermutationImportance



# Make a small change to the code below to use in this problem. 

perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)



eli5.show_weights(perm, feature_names = val_X.columns.tolist())
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)

data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)



features_2  = ['pickup_longitude',

               'pickup_latitude',

               'dropoff_longitude',

               'dropoff_latitude',

               'abs_lat_change',

               'abs_lon_change']



X = data[features_2]

new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)

second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)



# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y

perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)



# show the weights for the permutation importance you just calculated

eli5.show_weights(perm2, feature_names = features_2)
#Code to plot the partial dependence plot for pickup_longitude

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



feat_name = 'pickup_longitude'

pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()

for feat_name in base_features:

    pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X,

                               model_features=base_features, feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
fnames = ['pickup_longitude', 'dropoff_longitude']

longitudes_partial_plot = pdp.pdp_interact(

    model=first_model, dataset=val_X,

    model_features=base_features, 

    features=fnames

)

pdp.pdp_interact_plot(

    pdp_interact_out=longitudes_partial_plot,

    feature_names=fnames, 

    plot_type='contour'

)

plt.show()
# This is the PDP for pickup_longitude without the absolute difference features. Included here to help compare it to the new PDP you create

feat_name = 'pickup_longitude'

pdp_dist_original = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist_original, feat_name)

plt.show()







# create new features

data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)

data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)



features_2  = ['pickup_longitude',

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
import numpy as np

from numpy.random import rand



n_samples = 20000



# Create array holding predictive feature

X1 = 4 * rand(n_samples) - 2

X2 = 4 * rand(n_samples) - 2

# Create y. you should have X1 and X2 in the expression for y

y = -2 * X1 * (X1<-1) + X1 - 2 * X1 * (X1>1) - X2



# create dataframe because pdp_isolate expects a dataFrame as an argument

my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

predictors_df = my_df.drop(['y'], axis=1)



my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)



pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')



# visualize your results

pdp.pdp_plot(pdp_dist, 'X1')

plt.show()
import eli5

from eli5.sklearn import PermutationImportance



n_samples = 20000



# Create array holding predictive feature

X1 = 4 * rand(n_samples) - 2

X2 = 4 * rand(n_samples) - 2

# Create y. you should have X1 and X2 in the expression for y

y = X1 * X2





# create dataframe because pdp_isolate expects a dataFrame as an argument

my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

predictors_df = my_df.drop(['y'], axis=1)



my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)





pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')

pdp.pdp_plot(pdp_dist, 'X1')

plt.show()



perm = PermutationImportance(my_model).fit(predictors_df, my_df.y)



# show the weights for the permutation importance you just calculated

eli5.show_weights(perm, feature_names = ['X1', 'X2'])
import shap



"""

Assuming we prepared the data from NY taxi fare data set and 

trained the model as 'reg' just as in the code of variable importance.

"""



print('Computing SHAP...')



# Using TreeSHAP, not KernelSHAP. The former runs quicker but only available for tree-based model.

# Example of KernelExplainer: "shap.KernelExplainer(reg.predict, X_test)" Notice you need to assign prediction function and data.

# I test-ran and only to run 100 samples out of 6258, it took 120 seconds!!

explainer = shap.TreeExplainer(reg)

shap_values = explainer.shap_values(val_X)



# Show how the SHAP values output looks like.

pd.DataFrame(shap_values,columns=val_X.columns)
shap.initjs() #SHAP visualization is nicer with JavaScript which can be done with this small line.



# visualize the first prediction's explanation, decomposition between average vs. row specific prediction.

shap.force_plot(explainer.expected_value, shap_values[0,:], val_X.iloc[0,:])
# Variable importance-like plot.

shap.summary_plot(shap_values, val_X, plot_type="bar")
# PDP-like plot.

shap.dependence_plot("dropoff_longitude", shap_values, val_X)
# Each plot represents one data row, with SHAP value for each variable,

# along with red-blue as the magnitude of the original data.

shap.summary_plot(shap_values, val_X)
# Pretty visualization of the SHAP values per data row.

shap.force_plot(explainer.expected_value, shap_values, val_X)