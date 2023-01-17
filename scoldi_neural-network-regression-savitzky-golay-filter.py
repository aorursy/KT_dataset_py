import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt



from sklearn.neural_network import MLPRegressor



from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



from scipy.signal import savgol_filter
data_raw = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')

data_copy = data_raw.copy(deep=True)
data_raw.head()
# The task is to predict the 4 target variables ('pm', 'stator_yoke', 'stator_winding', 'stator_tooth') for

# for entries with 'profile_id' equal to 65 and 72 using data with smaller values of 'profile_id'.

# For the purposes of this notebook target variables only for entries with 'profile_id' = 65 will be predicted



target_variables = ['pm', 'stator_yoke', 'stator_winding', 'stator_tooth']

drop_cols = ['profile_id']
### 'torque' column is dropped according to the task 



data_copy.drop(columns=['torque'], inplace=True)



### 'profile_id' less than 65 for the training set



profile_list = np.array([i for i in data_raw['profile_id'].unique() if i < 65])
profile_list
cols = [item for item in list(data_copy.columns) if item not in drop_cols + target_variables]
cols
# apply MinMaxScaler in order to improve performance of MLPRegressor



scaler = MinMaxScaler()

data_copy[cols] = scaler.fit_transform(data_copy[cols])
data_copy[cols]
data_train = data_copy[data_copy['profile_id'].isin(profile_list)]



data_test = data_copy[data_copy['profile_id'] == 65]
data_train
# Take a look at how predictor variables behave 

for profile_id in [4, 11, 30, 45, 52]:

    print('id: ', profile_id)

    plt.figure(figsize=(26, 3))

    temp_data = data_train[data_train['profile_id'] == profile_id]

    i=1

    for col in cols: 

        sub = plt.subplot(1,7,i)

        i+=1

        plt.plot(temp_data[col])

        sub.set(xlabel='index', ylabel=col)

    plt.show()
# Take a look at how target variables behave  

for profile_id in [4, 11, 30, 45, 52]:

    print('id: ', profile_id)

    plt.figure(figsize=(22, 3))

    temp_data = data_train[data_train['profile_id'] == profile_id]

    i=1

    for col in target_variables: 

        sub = plt.subplot(1,4,i)

        i+=1

        plt.plot(temp_data[col])

        sub.set(xlabel='index', ylabel=col)

    plt.show()
corr = data_train.corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
X_train = data_train.drop(columns = target_variables + drop_cols)

Y_train = data_train[target_variables]
X_train.head()
reg = MLPRegressor(hidden_layer_sizes=(49), max_iter=2000, activation='tanh', verbose=False, random_state=1)
%%time

reg.fit(X_train, Y_train)
X_test = data_test.drop(columns = target_variables + drop_cols)

Y_test_actual = data_test[target_variables]
Y_test_prediction = reg.predict(X_test)  
prediction_df = pd.DataFrame(Y_test_prediction)
prediction_df.head()
prediction_df = prediction_df.rename(columns={0: 'pm_pred', 1: 'stator_yoke_pred', 

                                              2: 'stator_winding_pred', 3: 'stator_tooth_pred'})
Y_test_actual
prediction_df = Y_test_actual.reset_index().merge(prediction_df, left_index = True, right_index=True).set_index('index')
plt.figure(figsize=(30, 7))

print('actual')



for idx,col in enumerate(target_variables): 

    plt.subplot(1, 4, idx+1)

    plt.plot(prediction_df[col + '_pred'])

    plt.plot(prediction_df[col], color='red')

    plt.legend(loc="upper right")

plt.show()
predicted_cols = [col + '_pred' for col in target_variables] 

smoothed_cols = [col + '_smoothed' for col in predicted_cols] 
for column in predicted_cols:

    prediction_df[column+'_smoothed'] = savgol_filter(prediction_df[column], 

                                                      501, 1)
prediction_df.head()
plt.figure(figsize=(30, 7))

print('actual')



for idx,col in enumerate(target_variables): 

    plt.subplot(1, 4, idx+1)

    plt.plot(prediction_df[col + '_pred_smoothed'], label=col+': Predicted value')

    plt.plot(prediction_df[col], color='red', lw=1 , label=col+': Actual value')

    plt.legend(loc="upper right")

plt.show()
for column in target_variables:

    print('column: ', column)

    print('no smooth: ', mean_squared_error(Y_test_actual[column], prediction_df[column+'_pred'].to_numpy()))

    print('smooth: ', mean_squared_error(Y_test_actual[column], prediction_df[column+'_pred_smoothed'].to_numpy()), '\r\n')

    