import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style('darkgrid')
import sklearn
import tensorflow as tf
from tensorflow import keras
import lightgbm as lgb
data_path = '../input/abalone-dataset/abalone.csv'

data = pd.read_csv(data_path)
data.head()
split_size = int(0.9 * data.shape[0])
df = data[: split_size]
test_df = data[split_size: ]

print(df.shape)
print(test_df.shape)
df.info()
df.head()
df = pd.concat([df, pd.get_dummies(df['Sex'])], axis=1)
df = df.drop('Sex', axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df['Sex'])], axis=1)
test_df = test_df.drop('Sex', axis=1)
df = df.sample(frac=1, random_state=2020).reset_index(drop=True)

train_size = int(0.8 * df.shape[0])

train_df = df[: train_size]
valid_df = df[train_size: ]

train_labels = train_df.pop('Rings').values
train_data = train_df.values

valid_labels = valid_df.pop('Rings').values
valid_data = valid_df.values

print(train_df.shape)
print(valid_df.shape)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
BATCH_SIZE = 32
def build_nn():
    model = keras.models.Sequential([
        keras.layers.Dense(32, 'relu', kernel_initializer=keras.initializers.HeUniform(), 
                           activity_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-6)),
        keras.layers.Dense(1),
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
model = build_nn()

history = model.fit(train_data, train_labels, epochs=500, batch_size=BATCH_SIZE,  
          validation_data=(valid_data, valid_labels), validation_batch_size=BATCH_SIZE, 
          callbacks=[es], verbose=0
         )

res = model.evaluate(valid_data, valid_labels, verbose=0)
print('Validation MAE', res[1])
epochs = len(history.history['loss'])

y1 = history.history['loss']
y2 = history.history['val_loss']
x = np.arange(1, epochs+1)

plt.plot(x, y1, y2)
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
y1 = history.history['mae']
y2 = history.history['val_mae']
x = np.arange(1, epochs+1)

plt.plot(x, y1, y2)
plt.legend(['mae', 'val_mae'])
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.tight_layout()
lgb_train = lgb.Dataset(train_data, train_labels)
lgb_eval = lgb.Dataset(valid_data, valid_labels, reference=lgb_train)


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mse', 'mae'},
    'num_leaves': 25,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=0)
preds = gbm.predict(valid_data, num_iteration=gbm.best_iteration)

mae = keras.losses.MeanAbsoluteError()
mae(valid_labels, preds).numpy()
test_labels = test_df.pop('Rings').values
test_data = test_df.values
preds = model.predict(test_data)
preds = preds.flatten()

res = mae(test_labels, preds).numpy()
res
preds = gbm.predict(test_data, num_iteration=gbm.best_iteration)

res = mae(test_labels, preds).numpy()
res
df['bins'] = pd.cut(df['Rings'], 3, labels=[0, 1, 2])
df.head()
from sklearn.model_selection import StratifiedKFold

new_df = temp = pd.DataFrame()

skf = StratifiedKFold(n_splits=5)
for fold, (train_index, test_index) in enumerate(skf.split(df, df['bins'])):
    print('Fold: ', fold)
    
    sample = df.drop('bins', axis=1)
    y_train, y_val = sample['Rings'].values[train_index], sample['Rings'].values[test_index]
    sample = sample.drop('Rings', axis=1)
    x_train, x_val = sample.values[train_index], sample.values[test_index]
   
    
    model = build_nn()
    history = model.fit(x_train, y_train, epochs=500, batch_size=BATCH_SIZE,  
              validation_data=(x_val, y_val), validation_batch_size=BATCH_SIZE, 
              callbacks=[es], verbose=0)
    
    preds = model.predict(x_val).flatten()
    
    temp = pd.DataFrame({'nn': preds, 'target': y_val})
    
    path = f'nn_{fold}' 
    model.save_weights(path)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
    
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=0)
    
    gb_preds = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    
    temp['gb'] = gb_preds
    new_df = new_df.append(temp)
    
    path = f'lgb_{fold}.txt'
    gbm.save_model(path, num_iteration=gbm.best_iteration) 
new_df.head()
new_labels = new_df.pop('target')
new_data = new_df

new_labels = new_labels.values
new_data = new_data.values
from sklearn.model_selection import train_test_split

new_train_data, new_valid_data, new_train_labels, new_valid_labels = train_test_split(new_data, new_labels, 
                                                                                      test_size=0.2, random_state=42)
def build_meta_learner():
    model = keras.models.Sequential([
        keras.layers.Dense(16, 'relu', kernel_initializer=keras.initializers.HeUniform(), 
                           activity_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-5)),
        keras.layers.Dense(1),
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
meta_learner = build_meta_learner()

history = meta_learner.fit(new_train_data, new_train_labels, epochs=500, batch_size=BATCH_SIZE,  
          validation_data=(new_valid_data, new_valid_labels), validation_batch_size=BATCH_SIZE, 
          callbacks=[es], verbose=0
         )
res = meta_learner.evaluate(new_valid_data, new_valid_labels, verbose=0)
print('Validation MAE', res[1])
nn_preds = np.array([0.] * test_data.shape[0])
gb_preds = np.array([0.] * test_data.shape[0])

for i in range(5):
    path = f'./nn_{i}'
    model = build_nn()
    model.load_weights(path)
    nn_preds += model.predict(test_data).flatten()
    
    path = f'lgb_{i}.txt'
    reg = lgb.Booster(model_file=path)
    gb_preds += reg.predict(test_data)
nn_preds = nn_preds / 5
gb_preds = gb_preds / 5

new_test = pd.DataFrame({'nn': nn_preds, 'gb': gb_preds})
new_test.head()
answers = np.rint(meta_learner.predict(new_test).flatten())
answers[: 5]
res = mae(test_labels, answers).numpy()
res
pd.DataFrame({'answers': answers, 'labels': test_labels}).head(10)