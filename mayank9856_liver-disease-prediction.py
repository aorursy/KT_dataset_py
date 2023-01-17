import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



import tensorflow as tf
dataset = pd.read_csv('../input/indian_liver_patient.csv')

dataset.shape
dataset.head()
dataset.info()
dataset.isnull().sum()
dataset.describe().transpose()
dataset['Dataset'].value_counts()
# Categorical Fix

dataset['Gender'] = dataset['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
# Fill NA

dataset['Albumin_and_Globulin_Ratio'] = dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].mean())
# Normalize

dataset =(dataset - dataset.mean())/dataset.std()
plt.figure(figsize=(20,10)) 

sns.heatmap(dataset.corr(), annot=True)
dataset_features = ['Age', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Alkaline_Phosphotase', 'Aspartate_Aminotransferase', 'Gender', 'Total_Bilirubin', 'Dataset']

dataset = dataset[dataset_features]

dataset.shape
train_dataset = dataset.sample(frac=0.8,random_state=0)

test_dataset = dataset.drop(train_dataset.index)



train_labels = train_dataset.pop('Dataset')

test_labels = test_dataset.pop('Dataset')
train_dataset.shape, train_labels.shape, test_dataset.shape, test_labels.shape
model = tf.keras.Sequential([

    tf.keras.layers.Dense(7, activation='relu', input_shape=[len(train_dataset.keys())]),

    tf.keras.layers.Dense(14, kernel_initializer = 'uniform', activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(128, kernel_initializer = 'uniform', activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(2, activation='softmax')

#     tf.keras.layers.Dense(1, activation='sigmoid')

  ])



model.compile(

    optimizer='adam',

    loss=tf.keras.losses.CategoricalCrossentropy(),

#     optimizer='rmsprop',

#     loss='binary_crossentropy',

#     metrics=['accuracy', 'mean_absolute_error', 'mean_squared_error'],

    metrics=['accuracy'],

)



model.summary()
EPOCHS = 50



history = model.fit(train_dataset, 

                    train_labels, 

                    epochs=EPOCHS,

#                     validation_split=0.1,

                    verbose=1 )
# acc, loss, mae, mse

model.evaluate(test_dataset, test_labels, verbose=1)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch



plt.figure()

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.plot(hist['epoch'], hist['acc'],

       label='Train Error')

# plt.plot(hist['epoch'], hist['val_acc'],

#        label = 'Val Error')

plt.ylim([0,5])

plt.legend()



# plt.figure()

# plt.xlabel('Epoch')

# plt.ylabel('Mean Abs Error [MPG]')

# plt.plot(hist['epoch'], hist['mean_absolute_error'],

#        label='Train Error')

# plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

#        label = 'Val Error')

# plt.ylim([0,5])

# plt.legend()



# plt.figure()

# plt.xlabel('Epoch')

# plt.ylabel('Mean Square Error [$MPG^2$]')

# plt.plot(hist['epoch'], hist['mean_squared_error'],

#        label='Train Error')

# plt.plot(hist['epoch'], hist['val_mean_squared_error'],

#        label = 'Val Error')

# plt.ylim([0,20])

# plt.legend()

# plt.show()



model.predict(test_dataset)