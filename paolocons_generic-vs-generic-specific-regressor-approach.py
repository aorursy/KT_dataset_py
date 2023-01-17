

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



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from itertools import combinations

import matplotlib.pyplot as plt

import seaborn as sns
# read the dataset

df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/bmw.csv')



# create a copy of the dataset (will be used later when original values are needed but the 'original dataframe has already been encoded)

df2=df.copy()
# this plot shows that the price for some models has less variance than other:

# this might suggest that a specific predict-model for car-model can lead to better overall performance

# (while the generic model can be used to manage those car-model that do not have many rows in dataset

# and so the specific model would lead to poor performance)

sns.catplot(x = 'model', y= 'price', data = df2, kind='point', aspect=4);
# these plots show how much data we can rely on, for any car model

df2.groupby('model').count()['year'].values

plt.figure(figsize=(15, 6))

plt.bar(df2.groupby('model').count()['year'].index,df2.groupby('model').count()['year'].values,color='#005500', alpha=0.7, label='Number or records')

plt.xticks(df2.groupby('model').count()['year'].index, (df2.groupby('model').count()['year'].index), rotation = 90);

plt.xlabel('Car Model')

plt.ylabel('Number of Records')

plt.ylim([0,2500])

plt.suptitle('Number of Records vs Car Model')

plt.legend();
# STEP 1 CREATE A GENERIC PREDICTION MODELS (BASED ON WHOLE DATASET)



# prepare the 'generic' model to predict those models who doens't have a specific model providing good results

# also prepare the map of testset for predictions with 'best_model' regressor approach



# create an array of dataset one for each car model (will be used later)

car_model_names_list_from_df = df.model.unique();



# encode

car_model_le = LabelEncoder()

df['model'] = car_model_le.fit_transform(df.model.values)

transm_le = LabelEncoder()

df['transmission'] = transm_le.fit_transform(df.transmission.values)

fuelType_le = LabelEncoder()

df['fuelType'] = fuelType_le.fit_transform(df.fuelType.values)



# prepare X and Y

X = df.drop(['price'], axis=1).values

Y = df.price.values



# split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)



#before predicting, initialize two maps with tests and results: keys are the car model (it will be used later) 



model_to_x_test_map = {}

model_to_y_test_map = {}



for car_model_name in car_model_names_list_from_df:

    model_to_x_test_map[car_model_name] = []

    model_to_y_test_map[car_model_name] = []



for test_index, test in enumerate(X_test[:,0]):

    car_model_to_load = car_model_le.inverse_transform(np.array([int(test)]))

    model_to_x_test_map[car_model_to_load[0]].append(X_test[test_index])

    model_to_y_test_map[car_model_to_load[0]].append(Y_test[test_index])



# sanity check on lengths

#for car_model_name in df.model.values:

#print(len(model_to_x_test_map[car_model_name]), len(model_to_y_test_map[car_model_name]))

  

#scale

scaler = MinMaxScaler()

scaler.fit_transform(X_train);

scaler.transform(X_test);



# prepare the generic model (whole dataset)

general_rfr = RandomForestRegressor()

# Fit the model

general_rfr.fit(X_train,Y_train)

# Get predictions

Y_pred_train = general_rfr.predict(X_train)

Y_pred_test = general_rfr.predict(X_test)

# Calculate MAE and r2

mae_general_train = mean_absolute_error(Y_train,Y_pred_train)

mae_general_test = mean_absolute_error(Y_test,Y_pred_test)

r2_general_score_train = r2_score(Y_train,Y_pred_train)

r2_general_score_test = r2_score(Y_test,Y_pred_test)

print("'Generic' Approach Mean Absolute Error - Train: %.4f , Test: %.4f" %(mae_general_train, mae_general_test))

print("'Generic' Approach r2 score            - Train: %.4f , Test: %.4f" %(r2_general_score_train, r2_general_score_test))
# STEP 2 CREATE 'BY CAR MODEL' PREDICTION MODELS (ONE FOR EACH CAR MODEL) and save them



df_list = []



for car_model in car_model_names_list_from_df:

    by_car_model_df = df[df2['model'] == car_model].reindex()

    by_car_model_df.drop(['model'], inplace=True, axis=1)

    df_list.append(by_car_model_df)
# for each dataset in the list, do predictions and rate them



results_map = {}



predict_model_map = {}



for df_index, df_car_model in enumerate(df_list):

    # prepare X and Y

    X = df_car_model.drop(['price'], axis=1).values

    Y = df_car_model.price.values

    # split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)

    #scale

    scaler = StandardScaler()

    scaler.fit_transform(X_train);

    scaler.transform(X_test);

    # Instanciate the model

    rfr = RandomForestRegressor()

    # Fit the model

    rfr.fit(X_train,Y_train)

    # Get predictions

    Y_pred_train = rfr.predict(X_train)

    Y_pred_test = rfr.predict(X_test)

    # Calculate MAE and r2

    mae_train = mean_absolute_error(Y_train,Y_pred_train)

    mae_test = mean_absolute_error(Y_test,Y_pred_test)

    r2_score_train = r2_score(Y_train,Y_pred_train)

    r2_score_test = r2_score(Y_test,Y_pred_test)

    # put model in the model map

    predict_model_map[car_model_names_list_from_df[df_index]] = rfr

    print('Results for \'%s\'' %(car_model_names_list_from_df[df_index]))

    print("Mean Absolute Error - Train: %.4f , Test: %.4f" %(mae_train, mae_test))

    print("r2 score - Train: %.4f , Test: %.4f" %(r2_score_train, r2_score_test))

    results_map[car_model_names_list_from_df[df_index]] = (mae_train,mae_test,r2_score_train,r2_score_test);
# collect train/test results in a map



car_model_names = list(results_map.keys())

mae_train_values = [] 

mae_test_values =  []

r2_train_values =  []

r2_test_values =  []



for car_model in results_map.keys():

    results = results_map[car_model]

    mae_train_values.append(results[0])

    mae_test_values.append(results[1])

    r2_train_values.append(results[2])

    r2_test_values.append(results[3]);
# create a dataframe from that result map

n = pd.DataFrame(results_map)

n = n.transpose()

n.columns = ['MAE train','MAE test','R2 train', 'R2 test'];
# plot MAE train vs test on 'by car model' models

plt.figure(figsize=(10, 10))

plt.subplot(211)

plt.bar(car_model_names, mae_train_values, color='#550055', alpha=0.7, label="Mae Train")

plt.xticks(car_model_names, car_model_names, rotation = 90);

plt.xlabel('Car Model')

plt.ylabel('Mae Train')

plt.bar(car_model_names, mae_test_values, color='#005500', alpha=0.7, label="Mae Test")

plt.xticks(car_model_names, car_model_names, rotation = 90)

plt.xlabel('Car Model')

plt.ylabel('Mae Test')

plt.suptitle('MAE Train vs Test')

plt.legend()

plt.show()



# plot R2 train vs test

plt.figure(figsize=(10, 10))

plt.subplot(211)

plt.bar(car_model_names, r2_train_values, color='#550055', alpha=0.7, label='R2 Train')

plt.xticks(car_model_names, car_model_names, rotation = 90);

plt.xlabel('Car Model')

plt.ylabel('R2 Train')

#plt.subplot(241)

plt.bar(car_model_names, r2_test_values, color='#005500', alpha=0.7, label='R2 Test')

plt.xticks(car_model_names, car_model_names, rotation = 90)

plt.xlabel('Car Model')

plt.ylabel('R2 Train')

plt.ylim([-1,1])

plt.suptitle('R2 Score Train vs Test')

plt.legend()

plt.show()
# save those car models whose prices are better predicted with the specific ('by car model') regressor

car_models_better_predicted_with_specific_model = n[n['R2 test'] > r2_score_test].index
# STEP3: create a method that predict by using the appropriate regressor depending on car model ('generic' or 'specific' model)



def predict_with_best_model(single_x_test, car_model_to_predict, predict_model_map):

    if car_model_to_predict in car_models_better_predicted_with_specific_model:

        regressor = predict_model_map[car_model_to_predict]

        reshaped = single_x_test[1:].reshape(1, -1)

        return regressor.predict(reshaped)

    else:

        return general_rfr.predict(single_x_test.reshape(1, -1))



# go over mapped test set and predict each value, measure results

Y_optimized_pred_test = []

Y_expected_value_test = []

for car_model_in_map in model_to_x_test_map.keys():

    all_tests = model_to_x_test_map[car_model_in_map]

    all_test_results = model_to_y_test_map[car_model_in_map]

    for test_index, test in enumerate(all_tests):

        Y_optimized_pred_test.append(predict_with_best_model(test, car_model_in_map, predict_model_map))

        Y_expected_value_test.append(model_to_y_test_map[car_model_in_map][test_index])

Y_optimized_pred_test = np.array(Y_optimized_pred_test)



# Calculate MAE and r2 on test set with "by_car_model" regressor 

mae_optimized_test = mean_absolute_error(Y_expected_value_test,Y_optimized_pred_test)

r2_optimized_score_test = r2_score(Y_expected_value_test,Y_optimized_pred_test)

print("'Generic+Specific' Approach -  Mean Absolute Error -  Test: %.4f" %(mae_optimized_test))

print("'Generic+Specific' Approach -  R2 score            -  Test: %.4f" %(r2_optimized_score_test))
#The comparison of the results on the same test set seems to show that

# a 'Generic+Speciific' regressor leads to better predictions thant the sole 'Generic' model



print("Mean Absolute Error on test set - 'Generic': %.4f\t, 'Generic+Specific': %.4f" %(mae_general_test,mae_optimized_test))

print("R2 Score            on test set - 'Generic': %.4f\t, 'Generic+Specific': %.4f" %(r2_general_score_test,r2_optimized_score_test))
