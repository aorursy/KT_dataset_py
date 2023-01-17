import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
DATA_PATH = '../input/mushroom-classification/'
FILE_NAME = 'mushrooms.csv'
def load_data(data_path=DATA_PATH, file_name=FILE_NAME):
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)

dataset = load_data()
dataset.head()
dataset.info()
edible, poisonous = dataset['class'].value_counts()

print("Edible:\t  ", edible,"\nPoisonous:", poisonous)
# categorical to numerical
labels = {'e': 0, 'p': 1}
dataset['class'].replace(labels, inplace=True)

edible, poisonous = dataset['class'].value_counts()
print("0 - Edible:   ", edible,"\n1 - Poisonous:", poisonous)
X, y =  dataset.drop('class', axis=1), dataset['class'].copy()

print("X:",X.shape,"\ny:",y.shape)
from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("85% - X_train size:", X_train_full.shape[0], " y_train size:", y_train_full.shape[0])
print("15% - X_test size: ", X_test.shape[0], " y_test size: ", y_test.shape[0])
X_valid, X_train = X_train_full[:500], X_train_full[500:]
y_valid, y_train = y_train_full[:500], y_train_full[500:]

print("X_train:", X_train.shape[0], "y_train", y_train.shape[0])
print("X_valid: ", X_valid.shape[0], "y_valid ", y_valid.shape[0])
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

cat_attr_pipeline = Pipeline([
                        ('encoder', OrdinalEncoder())
                    ])

cols = list(X)
pipeline = ColumnTransformer([
                ('cat_attr_pipeline', cat_attr_pipeline, cols)
            ])


X_train = pipeline.fit_transform(X_train)
X_valid = pipeline.fit_transform(X_valid)
X_test  = pipeline.fit_transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
tf.random.set_seed(42)
model = Sequential([
    InputLayer(input_shape=(22,)),    # input  layer
    Dense(45, activation='relu'),     # hidden layer
    Dense(1,   activation='sigmoid')  # output layer
])
model.summary()
model.compile(loss='binary_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_cb = ModelCheckpoint('best_model.h5',
                                save_best_only=True)

early_stopping_cb = EarlyStopping(patience=3,
                                  restore_best_weights=True)
train_model = model.fit(X_train, y_train,
                          epochs=100,
                          validation_data=(X_valid, y_valid),
                          callbacks=[checkpoint_cb,
                                     early_stopping_cb])
pd.DataFrame(train_model.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
model.evaluate(X_test, y_test)
import seaborn as sns

#Parameters
title = 'Confusion Matrix'
custom_color = '#ffa600'   

#Function for drawing confusion matrix
def draw_confusion_matrix(cm, title = title, color = custom_color):
    palette = sns.light_palette(color, as_cmap=True)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d', cmap=palette)
    # Title
    ax.set_title('\n' + title + '\n',
                 fontweight='bold',
                 fontstyle='normal', 
                )
    # x y labels 
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold');
    # Classes names
    x_names = ['Poisonous', 'Edible']
    y_names = ['Poisonous', 'Edible']
    ax.xaxis.set_ticklabels(x_names, ha = 'center')
    ax.yaxis.set_ticklabels(y_names, va = 'center')
from sklearn.metrics import confusion_matrix

y_test_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_test_pred)

draw_confusion_matrix(cm)
#Function for plotting the ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, custom_color, label='Area: %0.3f' %roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate - Recall')
    plt.legend(loc='lower right')
    plt.show()
from sklearn.metrics import roc_curve, auc

y_test_prob = model.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plot_roc_curve(fpr, tpr, roc_auc)
X_new = X_test[:5]
y_prob = model.predict(X_new)
print(y_prob.round(3))
y_pred = model.predict_classes(X_new)
print(y_pred)
