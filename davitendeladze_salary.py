# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.preprocessing import StandardScaler



from tensorflow.python.keras.layers import Input, Dense, Activation,Dropout

from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras import metrics

from tensorflow.python.keras.callbacks import ModelCheckpoint



plt.style.use('ggplot')

df = pd.read_csv('/kaggle/input/random-salary-data-of-employes-age-wise/Salary_Data.csv')

df
len(df)
plt.hist(df['YearsExperience'])

plt.show()

plt.hist(df['Salary'])

plt.show()
g = sns.PairGrid(df, vars=['YearsExperience','Salary'],

 )

g.map(plt.scatter, alpha=0.8)
df.corr()
f = plt.figure(figsize=(19, 15))

plt.matshow(df.corr(), fignum=f.number)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);

plt.show()
scaler = StandardScaler()



df[['YearsExperience']] = scaler.fit_transform(df[['YearsExperience']])

df
feature = np.asarray(df['YearsExperience'])

target = np.asarray(df['Salary'])
train_feature = feature[:20]

train_target = target[:20]

val_target = target[20:25]

val_feature = feature[20:25]

test_target = target[25:]

test_feature = feature[25:]
model = Sequential([

    Dense(1024, input_shape=(1,)),

    Activation('relu'),

    Dropout(0.1),

    

    Dense(512),

    Activation('relu'),

    Dropout(0.1),



])

model.add(Dense(1))



model.summary()
model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mape"])
mcp_save = ModelCheckpoint('.salary_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(train_feature, train_target,

          validation_data=(val_feature, val_target),

          batch_size=5,

          epochs=2000,

          callbacks=[mcp_save]         

                   )

#Look at how loss is minimised with epoch

epochs = range(1, len(history.history["loss"])+1)

plt.figure(1, figsize=(8,4))

plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")

plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")

plt.xlabel("Gradient step"), plt.ylabel("RMS Loss");

plt.legend()

#Keep plot window alive!

plt.show()
model.load_weights(filepath=".salary_model.hdf5")

y = model.predict(test_feature)

error = model.evaluate(test_feature,test_target)

print("Test MS loss = " +str(error))

print("Model Accuracy =  " +str(100-error[1]))
print(np.transpose(y))

print(test_target)