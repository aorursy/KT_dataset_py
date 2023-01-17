import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('dark')

import sklearn

import tensorflow as tf

from tensorflow import keras
data_path = '../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv'



df = pd.read_csv(data_path)

df.head()
df.shape
df.info()
df = df.dropna()
df = df[df['sex'] != '.']
df.shape
df.info()
df.describe()
sns.countplot(df['species'])

plt.tight_layout()
sns.countplot(df['island'])

plt.tight_layout()
sns.countplot(df['sex'])

plt.tight_layout()
sns.distplot(df['culmen_length_mm'], bins=40, kde=True)

plt.tight_layout()
sns.distplot(df['culmen_depth_mm'], bins=40, kde=True)

plt.tight_layout()
sns.distplot(df['flipper_length_mm'], bins=40, kde=True)

plt.tight_layout()
sns.distplot(df['body_mass_g'], bins=40, kde=True)

plt.tight_layout()
pd.crosstab(df['island'],df['species']).plot.bar()

plt.tight_layout()
pd.crosstab(df['sex'],df['species']).plot.bar()

plt.tight_layout()
df = df.drop('sex', axis=1)

df.head()
sns.scatterplot(data=df, x="culmen_length_mm", y='culmen_depth_mm', hue="species")

plt.tight_layout()
sns.scatterplot(data=df, x="culmen_length_mm", y='flipper_length_mm', hue="species")

plt.tight_layout()
sns.scatterplot(data=df, x="culmen_length_mm", y='body_mass_g', hue="species")

plt.tight_layout()
sns.scatterplot(data=df, x="culmen_depth_mm", y='flipper_length_mm', hue="species")

plt.tight_layout()
sns.scatterplot(data=df, x="culmen_depth_mm", y='body_mass_g', hue="species")

plt.tight_layout()
sns.scatterplot(data=df, x="body_mass_g", y='flipper_length_mm', hue="species")

plt.tight_layout()
corr = df.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)

plt.tight_layout()
df['culmen_area_mm'] = df['culmen_length_mm'] * df['culmen_depth_mm']

df.head()
pd.get_dummies(df['island'])
df = pd.concat([df, pd.get_dummies(df['island'])], axis=1)

df.head()
df = df.drop('island', axis=1)

df.head()
my_species = df['species'].value_counts()

species_dict = {species: idx for idx, species in enumerate(list(my_species.index))}

species_dict
df['species'] = df['species'].map(species_dict)

df.head()
corr = df.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)

plt.tight_layout()
df.shape
df = df.sample(frac=1).reset_index(drop=True)

df.head()
all_labels = df.pop('species')

all_labels[: 10]
df = (df - df.min())/(df.max() - df.min())

df.head()
all_data = df.values

all_data[0]
train_size = round(0.9 * all_data.shape[0])



train_data = all_data[: train_size]

test_data = all_data[train_size: ]



train_labels = all_labels[: train_size]

test_labels = all_labels[train_size: ]



assert(len(train_data) == len(train_labels))
def build_model():

    model = keras.models.Sequential([

        keras.layers.Dense(64, 'selu'),

        keras.layers.Dense(64, 'selu'),

        keras.layers.Dropout(0.25),

        keras.layers.Dense(3, 'softmax'),

    ])

    

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

    

    return model
my_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, 

        restore_best_weights=True)
from sklearn.model_selection import StratifiedKFold



all_val_loss_histories = []

all_val_scores = []



skf = StratifiedKFold(n_splits=3)



for train_index, test_index in skf.split(train_data, train_labels):

    print('Fold...')

    x_train, x_val = train_data[train_index], train_data[test_index]

    y_train, y_val = train_labels[train_index], train_labels[test_index]

    

    model = build_model()



    history = model.fit(x_train, y_train, epochs=500, 

                  validation_data=(x_val, y_val),

                  callbacks=[my_cb], verbose=1)

    

    all_val_loss_histories.append(history.history['val_loss'])

    

    all_val_scores.append(model.evaluate(x_val, y_val, verbose=0))
avg_loss = np.mean([x[0] for x in all_val_scores])

print(f'Average loss is {avg_loss}')
avg_val_loss_history = [np.mean([x[i] for x in all_val_loss_histories]) for i in range(26)]



len(avg_val_loss_history)
plt.plot(range(1, len(avg_val_loss_history) + 1), avg_val_loss_history)

plt.xlabel('Epochs')

plt.ylabel('Validation Loss')

plt.tight_layout()
model = build_model()



history = model.fit(train_data, train_labels, epochs=26, 

              callbacks=[my_cb], verbose=1)
epochs = len(history.history['loss'])



y1 = history.history['loss']

x = np.arange(1, epochs+1)



plt.plot(x, y1)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.tight_layout()
y1 = history.history['acc']

x = np.arange(1, epochs+1)



plt.plot(x, y1)

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.tight_layout()
model.evaluate(test_data, test_labels)