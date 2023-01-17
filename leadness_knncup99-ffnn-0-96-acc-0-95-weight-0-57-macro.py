import os
import requests
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
tf.__version__


# Get features types dict and features list
content = open('../input/kdd-cup-1999-data/kddcup.names', 'r').readlines()
content
buf, *features = content
attack_types = buf.split(',')
attack_types[-1] = attack_types[-1][:-1]
features_types_dict = {f.split(':')[0]: f.split(':')[1][1:-2] for f in features}
features = list(features_types_dict.keys())
features_types_dict
# Get attack types dict
content = open('../input/kdd-cup-1999-data/training_attack_types', 'r').readlines()
content
buf = content[:-1]
target_classes = {
    'normal': 0,
    'u2r': 1,
    'r2l': 2,
    'probe': 3,
    'dos': 4
}
attack_types_dict = {line.split()[0]: line.split()[1] for line in buf}
attack_types_dict['normal'] = 'normal'
attack_types_dict
# Load data into df
data_file = '../input/kdd-cup-1999-data/kddcup.data.gz'
data = pd.read_csv(data_file, 
                      header=None, 
                      names=features + ['label'])
data['label'] = [i[:-1] for i in data['label'].values]
data
# Unnormalized numerical features
num_attrs = []
for i in [0, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 22, 31, 32]:
    num_attrs.append(features[i])
num_attrs


# Categorical features
cat_attrs = []
for i in [1, 2, 3]:
    cat_attrs.append(features[i])
cat_attrs


# Classification of data
label_counts_dict = data['label'].value_counts()
print('+', '-' * 7, '+', '-' * 16, '+', '-' * 8, '+', sep='')
print('|%-6s |%-15s |%-7s |' % ('Class', "Attack type", 'Count'))
print('+', '-' * 7, '+', '-' * 16, '+', '-' * 8, '+', sep='')
for (count, attack_type) in zip(label_counts_dict, label_counts_dict.keys()):
    print('|%-6s |%-15s |%-7d |' % (attack_types_dict[attack_type], attack_type, count))
    print('+', '-' * 7, '+', '-' * 16, '+', '-' * 8, '+', sep='')
# Detect small and numerous classes
numerous_classes = ['smurf', 'neptune', 'normal']
small_classes = [cl for cl in attack_types if cl not in numerous_classes]
small_classes, numerous_classes
# Form training, test and validation dataframes
train_df, test_df, val_df = data[1:2], data[2:3], data[3:4]

TRAIN_NUM = 15000
TEST_NUM = 1500

for cl in numerous_classes:
    train_df = train_df.merge(data[data['label']==cl][:TRAIN_NUM], how='outer')
    test_df = test_df.merge(data[data['label']==cl][TRAIN_NUM:TRAIN_NUM+TEST_NUM], how='outer')
    val_df = val_df.merge(data[data['label']==cl][TRAIN_NUM+TEST_NUM:TRAIN_NUM+TEST_NUM+TEST_NUM], how='outer')

for cl in small_classes:
    TRAIN_NUM = round(len(data[data['label']==cl]) * 0.8)
    TEST_NUM = round(len(data[data['label']==cl]) * 0.1)
    train_df = train_df.merge(data[data['label']==cl][:TRAIN_NUM], how='outer')
    test_df = test_df.merge(data[data['label']==cl][TRAIN_NUM:TRAIN_NUM+TEST_NUM], how='outer')
    val_df = val_df.merge(data[data['label']==cl][TRAIN_NUM+TEST_NUM:TRAIN_NUM+TEST_NUM+TEST_NUM], how='outer')
    
print('train_df: ', len(train_df))
print('test_df:  ', len(test_df))
print('val_df:   ', len(val_df))
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

# Define a data pipeline for a RandomForestClassifier       
def pipeline(data):
    df = data.copy()
    cat_encoder = LabelBinarizer()
    scaler = MinMaxScaler()
    for attr in cat_attrs:
        df[attr] = cat_encoder.fit_transform(df[attr].values.reshape(-1, 1))
    for attr in num_attrs:
        df[attr] = scaler.fit_transform(df[attr].values.reshape(-1, 1))
    return df
# Fit RandomForestClassifier to know feature importances
from sklearn.ensemble import RandomForestClassifier

train_data = pipeline(train_df)
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(train_data.drop('label', axis=1), train_data['label'])
# Determine the accuracy of a trained model
from sklearn.metrics import accuracy_score

test_data = pipeline(test_df)
target = test_data.pop('label')
test_predicted = rnd_clf.predict(test_data)
accuracy_score(target, test_predicted)
# Based on RandomForestClassifier, determine feature importances
print('+', '-' * 28, '+', '-' * 16, '+', sep='')
print('|%-27s |%-s|' % ('Feature', 'Importance(in %)'))
print('+', '-' * 28, '+', '-' * 16, '+', sep='')
for (feature, importance) in zip(features_types_dict, rnd_clf.feature_importances_):
    print('|%-28s|%-16f|' % (feature, importance * 100))
    print('+', '-' * 28, '+', '-' * 16, '+', sep='')
# Determine number of important features depending on their percentage of contribution
d = {i: imp for i, imp in enumerate(rnd_clf.feature_importances_)}
for bias in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
    print('>', bias * 100, '%: ', len(list(filter(lambda x: d[x] > bias, d))))
# Define data pipeline for NN model 
def df_to_dataset(input_df, main_df=data, bias = 0.05, shuffle=True, batch_size=32):
    """
    Performs data preprocessing.
    
    param data: pandas.Dataframe
    """
    df = input_df.copy()
    cat_encoder = LabelBinarizer()
    scaler = MinMaxScaler()
    for attr in cat_attrs:
        cat_encoder.fit(main_df[attr].values.reshape(-1, 1))
        df[attr] = cat_encoder.transform(df[attr].values.reshape(-1, 1))
    for attr in num_attrs:
        scaler.fit(main_df[attr].values.reshape(-1, 1))
        df[attr] = scaler.fit_transform(df[attr].values.reshape(-1, 1))
            
    d = dict(zip(features, rnd_clf.feature_importances_)) # dict(feature: feature_importance)
    df = df.drop(list(filter(lambda x: d[x] < bias, d)), axis=1) # drop unimportance features

    df['label'] = df['label'].apply(lambda x: target_classes[attack_types_dict[x]]) # 10 classes > 4 main classes
    df['label'], _ = df['label'].factorize()
        
    labels = df.pop('label')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    print(f'Features count: {len(df.columns)}')
    print('Сlass distribution: ', dict(labels.value_counts()))
    return dataset
# Form training, test and validation datasets
BATCH_SIZE = 32
BIAS = 0.03

train_dataset = df_to_dataset(train_df, bias=BIAS, batch_size=BATCH_SIZE)
val_dataset = df_to_dataset(val_df, bias=BIAS, batch_size=BATCH_SIZE)
test_dataset = df_to_dataset(test_df,bias=BIAS, batch_size=BATCH_SIZE)
# Define model
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model
# Compile and fit model
model = get_compiled_model()
epochs = 30
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    use_multiprocessing=True, 
                    epochs=epochs,
                   )
# Check accuracy and loss curves
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# Check accuracy on test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print('Accuracy on test dataset:', test_accuracy)
def labels_to_list(data):
    df = data.copy()
    df['label'] = df['label'].apply(lambda x: target_classes[attack_types_dict[x]]) # 10 classes > 4 main classes
    df['label'], _ = df['label'].factorize()
    return df['label'].values.tolist()
# Check confusion matrix on test dataset and get classification report
from sklearn.metrics import classification_report, confusion_matrix

predict_labels = model.predict(df_to_dataset(test_df, bias=BIAS, shuffle=False, batch_size=BATCH_SIZE))
predicted_labels = [np.argmax(predict) for predict in predict_labels]
true_labels = labels_to_list(test_df)
print('\nConfusion matrix:\n', confusion_matrix(true_labels, predicted_labels))
print(classification_report(true_labels, predicted_labels))