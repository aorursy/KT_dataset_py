# Libraries

import numpy as np
import pandas as pd
import time
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Data

print("Training data")
print(train.info(verbose=False))

print("\nTesting data")
print(test.info(verbose=False))
# Feature engineering

# Training data

train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

# Pythagoras theorem
train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)

# Means
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3  
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 

# Testing data

test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

# Pythagoras theorem
test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

# Means
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2
# Features

def wilderness_feature(df):
    df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']] = df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].sum(axis=1)
    return df

def soil_features(df):
    soil_types = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', \
                  'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', \
                  'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', \
                  'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', \
                  'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    df[soil_types] = df[soil_types].multiply([i for i in range(1, 41)], axis=1)
    df['soil_type'] = df[soil_types].sum(axis=1)
    return df
train = wilderness_feature(train)
train = soil_features(train)

test = wilderness_feature(test)
test = soil_features(test)
# Set style

sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,8.27)})
# Distance to hydrology (Horizontal versus Vertical) with Elevation

ax = plt.scatter(x=train['Horizontal_Distance_To_Hydrology'], y=train['Vertical_Distance_To_Hydrology'], c=train['Elevation'], cmap='jet')
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Distance')
plt.title("Distance to Hydrology with Elevation")
plt.show()
# Values and Labels

cols = train.columns.tolist()
print("Columns: ", cols)
    
columns = cols[1:11] + cols[56:]

values = train[columns]
labels = train['Cover_Type']

print("\nFeatures: ", columns)
# Model train and predict

start = time.time()

model_1 = ExtraTreesClassifier(n_estimators=375)  
model_1.fit(train[columns], train['Cover_Type'])
model_1_output = pd.DataFrame({"Id": test['Id'],"Cover_Type": model_1.predict(test[columns])})

print("Runtime ExtraTreesClassifier: ", time.time() - start)
# Predictions

model_1_output.head()
import lightgbm as lgb

params = {
    'learning_rate': 0.05, 
    'max_depth': 13, 
    'boosting': 'gbdt', 
    'objective': 'multiclass', 
    'num_class': 7,
    'metric': ['multi_logloss'], 
    'is_training_metric': True, 
    'seed': 19, 
    'num_leaves': 256, 
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8,
    'bagging_freq': 5, 
    'lambda_l1': 4, 
    'lambda_l2': 4, 
    'num_threads': 12
}
start = time.time()

model_2 = lgb.train(params, 
                    lgb.Dataset(values, label=labels-1),
                    1265,
                    verbose_eval=100, 
#                     early_stopping_rounds=100
                   )
model_2_predict = np.round(np.argmax(model_2.predict(test[columns].values), axis=1)).astype(int) + 1
model_2_output = pd.DataFrame({'Id': test['Id'], 'Cover_Type': model_2_predict})

print("Runtime LightGradientBoost: ", time.time() - start)
# Prediction

model_2_output.head()
# Output

final_predictions = model_1_output['Cover_Type'] * 0.5 + model_2_output['Cover_Type'] * 0.5
ceil_final_predictions = pd.DataFrame({'Id': test['Id'], 'Cover_Type': np.ceil(final_predictions).astype(np.int64)})
floor_final_predictions = pd.DataFrame({'Id': test['Id'], 'Cover_Type': np.floor(final_predictions).astype(np.int64)})

print(ceil_final_predictions.head())
print(floor_final_predictions.head())
# Files

ceil_final_predictions.to_csv('output_ceil.csv', index=False)
floor_final_predictions.to_csv('output_floor.csv', index=False)