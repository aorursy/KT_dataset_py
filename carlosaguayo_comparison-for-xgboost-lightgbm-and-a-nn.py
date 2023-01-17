import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('../input/voice.csv', header=0).values
x = dataset[:, :-1]
y = dataset[:, -1]
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)
test_size = 0.33
seed = 7
x_training, x_test, y_training, y_test = train_test_split(x,
                                                          label_encoded_y,
                                                          test_size=test_size,
                                                          random_state=seed)
# XGBoost
model = XGBClassifier()
model.fit(x_training, y_training)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy XGBoost: %.2f%%" % (accuracy * 100.0))
# LightGBM
import lightgbm as lgb
lgb_train = lgb.Dataset(x_training, y_training)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbose': -1
}
gbm = lgb.train(params,
                lgb_train,
                verbose_eval=False,
                valid_sets=lgb_eval)
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy LightGBM: %.2f%%" % (accuracy * 100.0))
# Neural Network
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(20*2, input_dim=x_training.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_training, y_training, epochs=30, verbose=1, validation_data=(x_test, y_test))
print (history.history['val_acc'][-1])