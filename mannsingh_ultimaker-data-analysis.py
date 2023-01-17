import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
import tensorflow as tf
df = pd.read_csv("../input/3dprinter/data.csv")
df = df.rename(columns={"tension_strenght": "tensile_strength"})
df
df.info()
df.infill_pattern.value_counts()
df.material.value_counts()
for column in df.columns:
    print(str(column)+": "+str(df[column].unique()))
    print()
#In this data set, ABS and PLA assigned 0 and 1 values for materials (abs = 0, pla = 1)
df.material = [0 if each=='abs' else 1 for each in df.material]

#In this data set, grid and honeycomb assigned 0 and 1 values for infill_pattern (grid = 0, honeycomb = 1)
df.infill_pattern = [0 if each=='grid' else 1 for each in df.infill_pattern]

df.head()
df.layer_height = df.layer_height*100
df.elongation = df.elongation*100
corr = df.corr()
corr.style.background_gradient(cmap='Accent').set_precision(2)
target_cols = ['tensile_strength', 'roughness', 'elongation']
y = df[target_cols].values
x = df.drop(target_cols,axis=1).values
from IPython.display import YouTubeVideo
YouTubeVideo('3lRhZTdafE4', width=800, height=450)
sns.scatterplot(x=df.fan_speed,y=df.tensile_strength,hue=df.material)
sns.scatterplot(x=df.layer_height,y=df.roughness,hue=df.material)
sns.scatterplot(x=df.infill_pattern,y=df.elongation,hue=df.material)
sns.scatterplot(x=df.nozzle_temperature,y=df.roughness,hue=df.material)
sns.scatterplot(x=df.print_speed,y=df.roughness,hue=df.material)
model = tf.keras.Sequential([
tf.keras.layers.Input(9),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1024),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.38),    
tf.keras.layers.PReLU(),   
tf.keras.layers.Dense(256),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.35),
tf.keras.layers.PReLU(),    
tf.keras.layers.Dense(3, activation="sigmoid")
])

model.compile(tf.optimizers.Adam(lr = 1e-3),loss='mean_squared_error',metrics=['accuracy'])
for n, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True).split(x)):
    print(f'Fold {n+1}')

    history = model.fit(x[tr],y.astype(float)[tr],
                validation_data=(x[te], y.astype(float)[te]),
                epochs=45, batch_size=8, verbose=2)
    print('')
    

plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss') 
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)