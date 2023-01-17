import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_train = pd.read_csv('/kaggle/input/car-price-modelling/train_data.tsv', sep='\t').drop(columns=['id',])
df_test =  pd.read_csv('/kaggle/input/car-prices-test/test_data.tsv', sep='\t')
df_train.head()
for data in [df_train, df_test]:
    engine_data = data['engine_summary'].str.split('/', expand=True)

    data['engine_inner_size'] = engine_data[0].str.extract('((\d+\.\d+)|(\d+))', expand=True).astype('float')[0]
    data['engine_hp'] = engine_data[1].str.extract('(\d+)', expand=True).astype('float')
    data['engine_type'] = engine_data[2]
    
    data['modification'] = data['brand'] + ' / ' + data['model'] + ' / ' + data['generation'] + ' / '+ data['engine_summary']
categorical = ['city', 'drive_type', 'owner_type', 'color', 'body', 'gear_type', 'wheel_type', 
               'state', 'is_new', 'doors_count', 'brand', 'engine_summary', 
               'brand', 'model', 'generation', 'engine_type', 'modification']

ignore = ['id',]
means = {}

for column in df_train.columns:
    if column in ignore:
        continue
    if column in categorical:
        means[column] = df_train[column].value_counts().index.tolist()[0]
    else:
        means[column] = df_train[column].mean(skipna = True)
        
means
df_train.fillna(value=means, inplace=True)
df_test.fillna(value=means, inplace=True)
df_train['doors_count'] = df_train['doors_count'].astype('str')
df_test['doors_count'] = df_test['doors_count'].astype('str')
df_train.head()
from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                            df_train.drop(columns=['price',]), np.log(df_train['price']), test_size=0.3, random_state=42)
model = CatBoostRegressor(iterations=10000,
                            learning_rate=1,
                             loss_function='MAPE',
                             gpu_ram_part=True)

# train the model
model.fit(train_X, train_y, cat_features=categorical)
predictions = np.exp(model.predict(df_test.drop(columns=['id',])))

data = {
    'id': df_test['id'],
    'price': predictions
}

result_df = pd.DataFrame(data)
result_df.to_csv('prices_2.csv', index=False)