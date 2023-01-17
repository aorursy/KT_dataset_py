import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
import seaborn as sns
from sklearn.compose import ColumnTransformer
from statistics import stdev
from keras import regularizers
df_raw = pd.read_csv("../input/crystal-system-properties-for-liion-batteries/lithium-ion batteries.csv")
df_raw.head()
data = df_raw # temporary save
data = data.drop(['Materials Id'], axis=1)
numerical_transformer = StandardScaler()
label_transformer = OrdinalEncoder()

n_cols = [c for c in data.columns if data[c].dtype in ['int64', 'float64', 'int32', 'float32']]
obj_cols = [c for c in data.columns if data[c].dtype in ['object', 'bool']]
print(n_cols, obj_cols)

ct = ColumnTransformer([('num', numerical_transformer, n_cols), ('non_num', label_transformer, obj_cols),])
processed = ct.fit_transform(data)
new_data = pd.DataFrame(columns=data.columns, data=processed)
new_data.head()
new_data.hist(figsize=(14,14), xrot=-45)
plt.show()
X = new_data.drop('Crystal System', axis=1)
y = new_data['Crystal System']
print(X.shape)
print(y.shape)
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, lw=0.5, cmap='coolwarm', annot=True)
corr_matrix = X.corr()
sns.pairplot(corr_matrix)
def train_model(n_runs, t_size=0.25):
    score = []
    for j in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=t_size, shuffle=True)
        y_encoded = to_categorical(y_train)
        model = Sequential()
        model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='softsign', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        global history 
        history = model.fit(X_train, y_encoded, epochs=100, verbose=False, validation_split=0.2)

        preds=model.predict_classes(X_test)

        score.append(accuracy_score(y_test, preds))
            
    print(f'Average score: '+ str(sum(score)/len(score)))
    print(f'Standard deviation: ' + str(stdev(score)))
    
    plt.title('Accuracy over ' + str(n_runs) + ' runs')
    plt.plot(score, label='Accuracy Score')
    plt.ylabel('Accuracy %')
    plt.xlabel('Runs')
    plt.legend()
    plt.show()
    
train_model(20, 0.3)
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
plt.title('Model Accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.plot(history.history['accuracy'], label='train accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
print(history.history.keys())