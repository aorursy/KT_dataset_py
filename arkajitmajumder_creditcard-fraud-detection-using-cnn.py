#loading the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
#loading the data
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
data.info()
pd.set_option("display.float","{:.2f}".format)
data.describe()
data.isnull().sum().sum()
data.columns
Labels = ['Normal', 'Fraud']

count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind="bar", rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), Labels)
plt.xlabel("labels")
plt.ylabel("frequency")
data.Class.value_counts()
fraud = data[data['Class']==1]
normal = data[data['Class']==0]
print("shape of fraud transaction:: {}".format(fraud.shape))
print("shape of normal transaction:: {}".format(normal.shape))

pd.concat([fraud.Amount.describe(), normal.Amount.describe()], axis=1)

pd.concat([fraud.Time.describe(), normal.Time.describe()], axis=1)

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Time Distribution(seconds)")
sns.distplot(data['Time'], color="red")

plt.subplot(2,2,2)
plt.title("Distribution of Amount")
sns.distplot(data['Amount'], color="green")

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
data[data.Class==1].Time.hist(bins=35, color = 'purple', alpha=0.5, label = "Fraudulant Transaction")
plt.legend()

plt.subplot(2,2,2)
data[data.Class==0].Time.hist(bins=35, color = 'indigo', alpha=0.5, label = "Non Fraudulant Transaction")
plt.legend()
data.hist(figsize=(20,20))

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), cmap="seismic")
plt.show()
#preprocessing the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = data.drop('Class', axis=1)
y = data.Class

X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train,X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validate_scaled = scaler.transform(X_validate)
preprocessed_dict = {}
preprocessed_dict['X_train_scaled'] = X_train_scaled
preprocessed_dict['X_test_scaled'] = X_test_scaled
preprocessed_dict['X_validate_scaled'] = X_validate_scaled
preprocessed_dict['y_train'] = y_train
preprocessed_dict['y_test'] = y_test
preprocessed_dict['y_validate'] = y_validate
# from sklearn.externals import joblib
import pickle
pickle.dump(preprocessed_dict, open("preprocessed_data.pkl1", "wb"))
#loading the data
preprocessed_data = pickle.load(open("preprocessed_data.pkl1", "rb"))
X_train_scaled = preprocessed_data['X_train_scaled']
X_test_scaled = preprocessed_data['X_test_scaled']
X_validate_scaled = preprocessed_data['X_validate_scaled']
y_train = preprocessed_data['y_train']
y_test = preprocessed_data['y_test']
y_validate = preprocessed_data['y_validate']
import numpy as np
import matplotlib.pyplot as plt

X_train_d = np.expand_dims(X_train_scaled, -1)
X_test_d = np.expand_dims(X_test_scaled, -1)
X_validate_d = np.expand_dims(X_validate_scaled, -1)
print("TRAINING DATA: X_train_d: {}, y_train: {}".format(X_train_d.shape, y_train.shape))
print("-"*60)
print("VALIDATION DATA: X_validate_d: {}, y_validate: {}".format(X_validate_d.shape, y_validate.shape))
print("-"*60)
print("TESTING DATA: X_test_d: {}, y_test: {}".format(X_test_d.shape, y_test.shape))
from tensorflow import keras
model = keras.models.Sequential([
        keras.layers.Conv1D(32, 2, activation='relu', input_shape = X_train_d[0].shape),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
    
        keras.layers.Conv1D(64, 2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
    
        keras.layers.Conv1D(128, 2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
    
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(1, activation="sigmoid")
])
model.summary()
model.compile(optimizer=keras.optimizers.Adam(0.0001), loss = "binary_crossentropy", metrics=['accuracy'])
fit1 = model.fit(X_train_d, y_train, validation_data=(X_validate_d, y_validate), batch_size=500, epochs=20)
model.evaluate(X_validate_d, y_validate)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(fit1.history['loss'], label='Loss')
plt.plot(fit1.history['val_loss'], label='val_Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(fit1.history['accuracy'], label='auc_1')
plt.plot(fit1.history['val_accuracy'], label='val_acc_1')
plt.legend()
model.save('CNN_model1.h5')
model1 = keras.models.Sequential([
        keras.layers.Conv1D(32, 2, activation='relu', input_shape = X_train_d[0].shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool1D(2),
        keras.layers.Dropout(0.1),
    
        keras.layers.Conv1D(64, 2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool1D(2),
        keras.layers.Dropout(0.2),
    
        keras.layers.Conv1D(128, 2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool1D(2),
        keras.layers.Dropout(0.3),
    
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(1, activation="sigmoid")
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
r1 = model1.fit(X_train_d, y_train, 
              validation_data=(X_validate_d, y_validate),
              batch_size=50, 
              epochs=20, 
             )
model1.evaluate(X_validate_d, y_validate)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(r1.history['loss'], label='Loss')
plt.plot(r1.history['val_loss'], label='val_Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(r1.history['accuracy'], label='auc_1')
plt.plot(r1.history['val_accuracy'], label='val_acc_1')
plt.legend()
model1.save('CNN_model2.h5')
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
y_predict1 = model.predict_classes(X_test_d)
y_predict2 = model.predict_classes(X_test_d)
print("for model 1")
print("accuracy score:: {}".format(accuracy_score(y_test, y_predict1)))
print("-"*50)
print("confusion matrix")
print(confusion_matrix(y_test, y_predict1))
print("*"*60)
print("for model 2")
print("accuracy score:: {}".format(accuracy_score(y_test, y_predict2)))
print("-"*50)
print("confusion matrix")
print(confusion_matrix(y_test, y_predict2))



