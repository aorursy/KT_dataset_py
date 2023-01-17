import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from math import isnan

from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/olxmobilbekas/mobilbekas.csv')
import tensorflow as tf



from tensorflow import keras

from tensorflow import feature_column

from tensorflow.keras import layers



print(tf.__version__)
# Selecting features

df1 = df.loc[:, ('Harga', 'Merek', 'Model', 'Varian', 'Tahun', 'Jarak tempuh')]
# Take only the maximum value of 'Jarak tempuh' from the range

for index, row in df.iterrows():

    df1.at[index, 'Jarak tempuh'] = row['Jarak tempuh'].split('-')[-1].replace(' km', '').replace('<', '').replace('>', '').replace('.', '')



df1.head()
# Convert 'Jarak tempuh' values to integer

df1['Jarak tempuh'] = df1['Jarak tempuh'].astype(int)
# Drop rows that contain 'Lain-lain' in 'Model' column

df1 = df1[df1.Model != 'Lain-lain']



# It doesn't make sense to have a car with 'Lain-lain' as a model
#Plot the number of each 'Merek'

sns.countplot(x ='Merek', data = df1, order=df1['Merek'].value_counts().index)

plt.xticks(rotation=90) 
#Plot the number of each 'Model' 

sns.countplot(x ='Model', data = df1, order=df1['Model'].value_counts().index) 

plt.xticks(rotation=90)
# Fill NaN values in Varian column by modus of corresponding model

from scipy import stats

from math import isnan



model_nan_filler = {}

model_set = set()



for item in df1['Model']:

    model_set.add(item)



for model in model_set:

    model_df = df1.loc[df1['Model'] == model]

    filler = stats.mode(model_df['Varian'])[0][0]

    

    model_nan_filler[model] = filler



for index, row in df1.iterrows():

    if row['Varian'] is np.nan:

        df1.at[index, 'Varian'] = model_nan_filler[row['Model']]
# Drop 'Model' whose number less than 100

df2 = df1.groupby("Model").filter(lambda x: len(x) > 100)

print(len(df2))
#Plot the number of each 'Model'

sns.countplot(x ='Model', data = df2, order=df2['Model'].value_counts().index)

plt.xticks(rotation=90) 
#Plot the number of each 'Merek'

sns.countplot(x ='Merek', data = df2, order=df2['Merek'].value_counts().index)

plt.xticks(rotation=90) 
#Plot the number of 'Model' for each 'Merek'

merek_set = set()



for item in df2['Merek']:

    merek_set.add(item)



fig, axes =plt.subplots(4,4, figsize=(20,15), sharex=True)

axes = axes.flatten()

for ax, merek in zip(axes, merek_set): 

    a = pd.Series(df2['Model'][df2['Merek'] == merek].tolist()) 

    sns.countplot(x=df2['Merek'][df2['Merek'] == merek].tolist(), data=df2, ax=ax, 

                  hue=df2['Model'][df2['Merek'] == merek].tolist(),

                  hue_order=a.value_counts().index).set(title=merek)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True) 

    

plt.xticks([])

plt.tight_layout() 

plt.show()
#Plot the number of 'Varian' for each 'Model'

model_set = set()



for item in df2['Model']:

    model_set.add(item)



fig, axes =plt.subplots(31,1, figsize=(5,100), sharex=True)

axes = axes.flatten()

for ax, model in zip(axes, model_set):  

    a = pd.Series(df2['Varian'][df2['Model']==model].tolist()) 

    sns.countplot(x=df2['Model'][df2['Model']==model].tolist(), data=df2, ax=ax, 

                  hue=df2['Varian'][df2['Model']==model].tolist(),

                  hue_order=a.value_counts().index).set(title=model)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=4) 



plt.xticks([])

plt.tight_layout() 

plt.show()
#'Harga' Boxplot for each 'Model'

model_set = set()



for item in df2['Model']:

    model_set.add(item)



fig, axes =plt.subplots(7,5, figsize=(10,20), sharex=True)

axes = axes.flatten()

for ax, model in zip(axes, model_set):  

    sns.boxplot(y='Harga', data=df2[df2['Model']==model], showfliers=True, ax=ax).set(title=model)

    

plt.tight_layout() 

plt.show()
# Drop outliers on 'Harga' grouped by 'Model'



outlier_params = {}

model_set = set()

indexes = []



for item in df2['Model']:

    model_set.add(item)



for model in model_set:

    model_df = df2.loc[df2['Model'] == model]

    param = [model_df['Harga'].mean(),

             model_df['Harga'].std() + 1]

    

    outlier_params[model] = param



for index, row in df2.iterrows():

    # Calculate Z-score then group index whose value >=3

    if (np.abs((row['Harga'] - outlier_params[row['Model']][0]) / outlier_params[row['Model']][1]) >= 3):

        indexes.append(index)



print(len(indexes))

df2 = df2.drop(index=indexes) # Remove rows with Z-score >=3



print(df2.shape)
#'Harga' Boxplot for each 'Model' after outliers removed

model_set = set()



for item in df2['Model']:

    model_set.add(item)



fig, axes =plt.subplots(7,5, figsize=(10,20), sharex=True)

axes = axes.flatten()

for ax, model in zip(axes, model_set):  

    sns.boxplot(y='Harga', data=df2[df2['Model']==model], showfliers=True, ax=ax).set(title=model)

    

plt.tight_layout() 

plt.show()
# Scale the labels

scale_factor = 1000000



df2["Harga"] /= scale_factor 



# Shuffle the examples

df2 = df2.reindex(np.random.permutation(df2.index))
# Scaling column 'Jarak Tempuh'

df2['Jarak tempuh'] = minmax_scale(df2['Jarak tempuh'])
# Set 1986 as minimum value in column 'Tahun'

df2['Tahun'].replace('<1986', 1986, inplace=True)



# Scaling column 'Tahun'

df2['Tahun'] = minmax_scale(df2['Tahun'])
# Rename dataframe columns

df2 = df2.rename(columns={"Merek": "merek", "Model": "model", "Tahun": "tahun", 

                        "Jarak tempuh": "jarak_tempuh", "Harga": "harga", "Varian":"varian"})
# Drop column 'Varian'

df3 = df2.drop(['varian'], axis=1) 
# Split the dataframe into train and test

train, test = train_test_split(df3, test_size=0.2)

print(len(train), 'train examples')

print(len(test), 'test examples')
# Create an empty list that will eventually hold all feature columns.

feature_columns = []



# Create a numerical feature column to represent tahun.

tahun = tf.feature_column.numeric_column("tahun")

feature_columns.append(tahun)



# Create a numerical feature column to represent jarak tempuh.

jarak_tempuh = tf.feature_column.numeric_column("jarak_tempuh")

feature_columns.append(jarak_tempuh)
# Create a categorical feature column to represent merek.

merek = feature_column.categorical_column_with_vocabulary_list(

      'merek', df3['merek'].unique())

merek_one_hot = feature_column.indicator_column(merek)

feature_columns.append(merek_one_hot)



# Create a categorical feature column to represent model.

model = feature_column.categorical_column_with_vocabulary_list(

        'model', df3['model'].unique())

model_one_hot = feature_column.indicator_column(model)

feature_columns.append(model_one_hot)
# Split the dataset into features and label.

features = {name:np.array(value) for name, value in train.items()}

label = np.array(features.pop('harga'))



test_features = {name:np.array(value) for name, value in test.items()}

test_label = np.array(test_features.pop('harga')) 
# The following variables are the hyperparameters.

learning_rate = 0.1

epochs = 300

batch_size = 30
model = tf.keras.models.Sequential([

    # Convert the list of feature columns into a layer that will later be fed into the model.

    tf.keras.layers.DenseFeatures(feature_columns),

    # Define the output layer.

    tf.keras.layers.Dense(1)

])
model.compile(optimizer=tf.optimizers.Adam(lr=learning_rate),

              loss=tf.keras.losses.MeanAbsoluteError(),

              metrics=[tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(features,

                    label,

                    batch_size=batch_size,

                    epochs=epochs,

                    shuffle=True,

                    verbose=1)



print("\n Evaluate the new model against the test set:")

model.evaluate(x = test_features, y = test_label, batch_size=batch_size) 
# The list of epochs is stored separately from the rest of history.

epochs = history.epoch



# To track the progression of training, gather a snapshot

# of the model's mean squared error at each epoch. 

hist = pd.DataFrame(history.history)

mae = hist["mean_absolute_error"]



plt.figure()

plt.xlabel("Epoch")

plt.ylabel("mean_absolute_error")



plt.plot(epochs, mae, label="Loss")

plt.legend()

plt.ylim([mae.min()*0.95, mae.max() * 1.03])

plt.show()  
# The following variables are the hyperparameters.

learning_rate = 0.001

epochs = 200

batch_size = 30
model = tf.keras.models.Sequential([

    # Convert the list of feature columns into a layer that will later be fed into the model.

    tf.keras.layers.DenseFeatures(feature_columns),

    # Define the first hidden layer with 256 nodes. 

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)),

    # Define the second hidden layer with 256 nodes. 

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)),

    # Define the third hidden layer with 256 nodes. 

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)),

    # Define the output layer.

    tf.keras.layers.Dense(1)  

])
model.compile(optimizer=tf.optimizers.Adam(lr=learning_rate),

              loss=tf.keras.losses.MeanAbsoluteError(),

              metrics=[tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(features,

                    label,

                    batch_size=batch_size,

                    epochs=epochs,

                    shuffle=True,

                    verbose=1)



print("\n Evaluate the new model against the test set:")

model.evaluate(x = test_features, y = test_label, batch_size=batch_size) 
# The list of epochs is stored separately from the rest of history.

epochs = history.epoch



# To track the progression of training, gather a snapshot

# of the model's mean squared error at each epoch. 

hist = pd.DataFrame(history.history)

mae = hist["mean_absolute_error"]



plt.figure()

plt.xlabel("Epoch")

plt.ylabel("mean_absolute_error")



plt.plot(epochs, mae, label="Loss")

plt.legend()

plt.ylim([mae.min()*0.95, mae.max() * 1.03])

plt.show()  
# Split the dataframe into train and test

train, test = train_test_split(df2, test_size=0.2)

print(len(train), 'train examples')

print(len(test), 'test examples')
# Create an empty list that will eventually hold all feature columns.

feature_columns = []



# Create a numerical feature column to represent tahun.

tahun = tf.feature_column.numeric_column("tahun")

feature_columns.append(tahun)



# Create a numerical feature column to represent jarak tempuh.

jarak_tempuh = tf.feature_column.numeric_column("jarak_tempuh")

feature_columns.append(jarak_tempuh)
# Create a categorical feature column to represent merek.

merek = feature_column.categorical_column_with_vocabulary_list(

      'merek', df2['merek'].unique())

merek_one_hot = feature_column.indicator_column(merek)

feature_columns.append(merek_one_hot)



# Create a categorical feature column to represent model.

model = feature_column.categorical_column_with_vocabulary_list(

        'model', df2['model'].unique())

model_one_hot = feature_column.indicator_column(model)

feature_columns.append(model_one_hot)



# Create a categorical feature column to represent model.

model = feature_column.categorical_column_with_vocabulary_list(

        'varian', df2['varian'].unique())

model_one_hot = feature_column.indicator_column(model)

feature_columns.append(model_one_hot)
# Split the dataset into features and label.

features = {name:np.array(value) for name, value in train.items()}

label = np.array(features.pop('harga'))



test_features = {name:np.array(value) for name, value in test.items()}

test_label = np.array(test_features.pop('harga')) 
# The following variables are the hyperparameters.

learning_rate = 0.001

epochs = 200

batch_size = 30
model = tf.keras.models.Sequential([

    # Convert the list of feature columns into a layer that will later be fed into the model.

    tf.keras.layers.DenseFeatures(feature_columns),

    # Define the first hidden layer with 256 nodes. 

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)),

    # Define the second hidden layer with 256 nodes. 

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)),

    # Define the third hidden layer with 256 nodes. 

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)),

    # Define the output layer.

    tf.keras.layers.Dense(1)  

])
model.compile(optimizer=tf.optimizers.Adam(lr=learning_rate),

              loss=tf.keras.losses.MeanAbsoluteError(),

              metrics=[tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(features,

                    label,

                    batch_size=batch_size,

                    epochs=epochs,

                    shuffle=True,

                    verbose=1)



print("\n Evaluate the new model against the test set:")

model.evaluate(x = test_features, y = test_label, batch_size=batch_size) 
# The list of epochs is stored separately from the rest of history.

epochs = history.epoch



# To track the progression of training, gather a snapshot

# of the model's mean squared error at each epoch. 

hist = pd.DataFrame(history.history)

mae = hist["mean_absolute_error"]



plt.figure()

plt.xlabel("Epoch")

plt.ylabel("mean_absolute_error")



plt.plot(epochs, mae, label="Loss")

plt.legend()

plt.ylim([mae.min()*0.95, mae.max() * 1.03])

plt.show()  