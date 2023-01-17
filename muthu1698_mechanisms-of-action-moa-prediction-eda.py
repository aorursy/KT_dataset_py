import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict, cross_validate

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder, RobustScaler, Normalizer

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from sklearn.pipeline import Pipeline

from xgboost import XGBRFClassifier

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import multilabel_confusion_matrix, label_ranking_loss, log_loss, roc_auc_score

from sklearn.linear_model import LogisticRegression, Perceptron

from sklearn.neural_network import MLPClassifier

from sklearn.multioutput import MultiOutputClassifier

from sklearn.decomposition import PCA 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test_targets = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
#quick look at our data types & null counts 

train_features.info()
# To better understand the numeric data, we want to use the .describe() method. 

# This gives us an understanding of the central tendencies of the data. 



train_features.describe()
# Reading the train dataset

train_features.head(3)
# Reading the test dataset 

test_features.head(3)
# To calculate the rows and columns in the dataset

print('The Training dataset has {} rows and {} columns.'.format(len(train_features), len(train_features.columns)))
print('The Test dataset has {} rows and {} columns.'.format(len(test_features), len(test_features.columns)))
fig, ax = plt.subplots(1, 3, figsize=(20, 5))





sns.countplot(train_features['cp_type'],palette=("Blues"), ax=ax[0])

ax[0].set_title('cp_type distribution')



sns.countplot(train_features['cp_time'],palette=("Blues"), ax=ax[1])

ax[1].set_title('cp_time distribution')



sns.countplot(train_features['cp_dose'],palette=("Blues"), ax=ax[2])

ax[2].set_title('cp_dose distribution')



fig.suptitle('Distribution in Train dataset of Type, Time and Dose')
fig, ax = plt.subplots(1, 3, figsize=(20, 5))





sns.countplot(test_features['cp_type'],palette=("BuGn_r"), ax=ax[0])

ax[0].set_title('cp_type distribution')



sns.countplot(test_features['cp_time'],palette=("BuGn_r"), ax=ax[1])

ax[1].set_title('cp_time distribution')



sns.countplot(test_features['cp_dose'],palette=("BuGn_r"), ax=ax[2])

ax[2].set_title('cp_dose distribution')



fig.suptitle('Distribution in Test dataset of Type, Time and Dose')
train_features['cp_type'] = train_features['cp_type'].astype('category')

train_features['cp_type'].cat.categories = [0, 1]

train_features['cp_type'] = train_features['cp_type'].astype("int")
train_features['cp_dose'] = train_features['cp_dose'].astype('category')

train_features['cp_dose'].cat.categories = [0, 1]

train_features['cp_dose'] = train_features['cp_dose'].astype("int")
train_features['cp_time'] = train_features['cp_time'].astype('category')

train_features['cp_time'].cat.categories = [0, 1, 2]

train_features['cp_time'] = train_features['cp_time'].astype("int")
train_features.head()
print('Number of "g-" features are: ', len([i for i in train_features.columns if i.startswith('g-')]))

print('Number of "c-" features are: ', len([i for i in train_features.columns if i.startswith('c-')]))
fig, ax = plt.subplots(3, 3, figsize=(20, 5))



sns.kdeplot(test_features['g-0'], shade = True, color = 'coral', ax=ax[0, 0])

sns.kdeplot(test_features['g-20'], shade = True, color = 'coral', ax=ax[0, 1])

sns.kdeplot(test_features['g-555'], shade = True, color = 'coral', ax=ax[0, 2])

sns.kdeplot(test_features['g-105'], shade = True, color = 'coral', ax=ax[1, 0])

sns.kdeplot(test_features['g-725'], shade = True, color = 'coral', ax=ax[1, 1])

sns.kdeplot(test_features['g-598'], shade = True, color = 'coral', ax=ax[1, 2])

sns.kdeplot(test_features['g-366'], shade = True, color = 'coral', ax=ax[2, 0])

sns.kdeplot(test_features['g-450'], shade = True, color = 'coral', ax=ax[2, 1])

sns.kdeplot(test_features['g-600'], shade = True, color = 'coral', ax=ax[2, 2])
fig, ax = plt.subplots(3, 3, figsize=(20, 5))



sns.kdeplot(test_features['c-0'], shade = True, color = 'blue', ax=ax[0, 0])

sns.kdeplot(test_features['c-20'], shade = True, color = 'blue', ax=ax[0, 1])

sns.kdeplot(test_features['c-99'], shade = True, color = 'blue', ax=ax[0, 2])

sns.kdeplot(test_features['g-66'], shade = True, color = 'blue', ax=ax[1, 0])

sns.kdeplot(test_features['g-88'], shade = True, color = 'blue', ax=ax[1, 1])

sns.kdeplot(test_features['g-73'], shade = True, color = 'blue', ax=ax[1, 2])

sns.kdeplot(test_features['g-45'], shade = True, color = 'blue', ax=ax[2, 0])

sns.kdeplot(test_features['g-59'], shade = True, color = 'blue', ax=ax[2, 1])

sns.kdeplot(test_features['g-37'], shade = True, color = 'blue', ax=ax[2, 2])
g_cols = [f'g-{i}' for i in range(772)]

fig, ax = plt.subplots(1, 2, figsize=(20, 4))



sns.distplot(train_features[g_cols].mean(), kde=False,color = 'green', bins = 75, ax = ax[0])

sns.distplot(train_features[g_cols].std(), kde=False,color = 'green',  bins = 75, ax = ax[1])
c_cols = [f'c-{i}' for i in range(100)]

fig, ax = plt.subplots(1, 2, figsize=(20, 4))



sns.distplot(train_features[c_cols].mean(), kde=False,color = 'purple', bins = 15, ax = ax[0])

sns.distplot(train_features[c_cols].std(), kde=False,color = 'purple',  bins = 15, ax = ax[1])
# Compute the correlation matrix

corr = train_features[g_cols[:40]].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Pairwise correlations of gene features for first 40')

plt.show()
f, ax = plt.subplots(figsize=(15, 4))

sns.distplot(train_features[g_cols].corr(),color='r')

plt.title("Distribution for gene feature")

plt.show()
plt.figure(figsize=(10,7))

sns.heatmap(train_features[c_cols].corr(), cmap='coolwarm');
f, ax = plt.subplots(figsize=(15, 4))

sns.distplot(train_features[c_cols].corr(),color='lightsalmon')

plt.title("Distribution for Gene Feature")

plt.show()
train_features_cor = train_features.corr()
train_features_cor.head()
vifs = pd.DataFrame(np.linalg.inv(train_features_cor.values).diagonal(), index = train_features_cor.index, columns=['VIF'])
vifs.tail(6)
# we take a value where vif is greater than 15.



greater_vifs = vifs.where(vifs>15)

greater_vifs = greater_vifs.dropna()
cols_remove = greater_vifs.index
cols_remove 

# This is a feature that have highly correlated with any number of the other variables.
new_train_features_data = train_features.drop(columns=cols_remove) # we drop these columns highly correlated
new_train_features_data.head()
test_features['cp_type'] = test_features['cp_type'].astype('category')

test_features['cp_type'].cat.categories = [0, 1]

test_features['cp_type'] = test_features['cp_type'].astype("int")
test_features['cp_dose'] = test_features['cp_dose'].astype('category')

test_features['cp_dose'].cat.categories = [0, 1]

test_features['cp_dose'] = test_features['cp_dose'].astype("int")
test_features['cp_time'] = test_features['cp_time'].astype('category')

test_features['cp_time'].cat.categories = [0, 1, 2]

test_features['cp_time'] = test_features['cp_time'].astype("int")
test_features.head()
targets = train_targets_scored
targets.head()
print('The Tergets dataset has {} rows and {} columns.'.format(len(targets), len(targets.columns)))
target_cols = targets.columns[1:]
targets_fre = (targets[target_cols].mean() * 100).sort_values()[-20:].index
plt.figure(figsize=(15,4))

(targets[targets_fre].mean() * 100).sort_values().plot.bar()

plt.title('Most frequent targets')

plt.ylabel('% of true labels')

plt.show()
plt.figure(figsize=(15,4))

vc = targets[target_cols].sum(axis=1).value_counts()

plt.title('True labels per row distribution')

plt.ylabel('# of rows')

plt.xlabel('# of true targets per row')

plt.bar(vc.index, vc.values)

plt.show()
train_targets_scored.drop(['sig_id'], axis=1, inplace=True)
train_targets_scored.head()
test_ids = test_features['sig_id']
for d in [new_train_features_data, test_features]:

    d.drop(['sig_id','cp_type', 'cp_dose', 'cp_time'], axis=1, inplace=True)

    

train_features.head()
x_train, x_cv, y_train, y_cv = train_test_split(new_train_features_data, train_targets_scored, test_size=0.2) 
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(512, input_dim=x_train.shape[1], activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')

])



model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)



model.summary()
factor= 0.8250037987063858

patience=2

min_lr= 5.101088055532695e-05

lr=6.353131263848553e-05

batch_size=353

epochs=359



def callbacks(file_path):

    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',

                                         factor=factor,

                                         patience=patience,

                                         cooldown=1,

                                         min_lr=min_lr,

                                         verbose=1)

    checkpoint = ModelCheckpoint(filepath = file_path,monitor='val_loss',

                                 mode='min',save_best_only=True,verbose=1)



    early = EarlyStopping(monitor="val_loss", mode="min", patience= patience)



    return [reduce_learning_rate,checkpoint,early]



file_path = model.name+'best_weights.hd5'

callbacks_list = callbacks(file_path = file_path)



optimizer = tf.keras.optimizers.Adam(lr=lr, amsgrad=True)

#compile the model

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer)



history=model.fit(x_train,y_train,epochs= epochs, batch_size=batch_size, callbacks = callbacks_list)
(test_targets[pd.read_csv('../input/lish-moa/sample_submission.csv').columns]).to_csv('submission.csv',index=False)