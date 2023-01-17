import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import shap



shap.initjs()
np.random.seed(27)

tf.random.set_seed(27)
df = pd.read_csv("../input/heart-disease-uci/heart.csv")

print(f"Dataset shape: {df.shape}")

print(df.head())

print(df.describe())
# quick and dirty one-hot encoding

df = pd.get_dummies(df,prefix=['ca','thal','cp'],columns=['ca','thal','cp'], drop_first=True)
df_train = df.dropna()

df_target = df_train.pop('target')



X_train, X_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.2, random_state=42)



normalized_X_train=(X_train-X_train.mean())/X_train.std()

normalized_X_test=(X_test-X_train.mean())/X_train.std()
def build_model():

    inputs = keras.Input(shape=(len(X_train.keys()),))

    x = layers.Dense(64, activation='tanh')(inputs) 

    x = layers.Dense(64, activation='tanh')(x)

    x = layers.Dense(32, activation='tanh')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)



    model = keras.Model(inputs, outputs, name="death_rate_model")



    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='binary_crossentropy',

                  optimizer=optimizer,

                  metrics=['accuracy'])

    return model

  

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)   



class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: print('')

        print('.', end='')
model = build_model()
model.fit(X_train, y_train, epochs=3000, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

print("")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
kernel_explainer = shap.KernelExplainer(model.predict, X_test)

shap_values = kernel_explainer.shap_values(X_test, nsamples=100, l1_reg="aic")[0]
for i in np.random.choice(range(len(X_test)),5):

    x = shap.force_plot(kernel_explainer.expected_value, shap_values[i], X_train.iloc[i])

    display(x)
shap.summary_plot(shap_values, X_test)
shap.force_plot(kernel_explainer.expected_value, shap_values, X_test)