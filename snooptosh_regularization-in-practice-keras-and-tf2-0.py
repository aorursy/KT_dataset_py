import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

iris = load_iris()



# Load data into a DataFrame

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(f"Number of samples --> {df.shape}")

# Convert datatype to float

df = df.astype(float)

# append "target" and name it "label"

df['label'] = iris.target

# Use string label instead

df['label'] = df.label.replace(dict(enumerate(iris.target_names)))

print(f"\n## Target class distribution --> \n{df['label'].value_counts()}")

df.head()

# label -> one-hot encoding

label = pd.get_dummies(df['label'], prefix='label')

df = pd.concat([df, label], axis=1)

# drop old label

df.drop(['label'], axis=1, inplace=True)



df.head()
# Creating X and y

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Convert DataFrame into np array

X = np.asarray(X)

y = df[['label_setosa', 'label_versicolor', 'label_virginica']]

# Convert DataFrame into np array

y = np.asarray(y)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense



def create_model(): 

    model = Sequential([

        Dense(64, activation='relu', input_shape=(4,)),

        Dense(128, activation='relu'),

        Dense(128, activation='relu'),

        Dense(128, activation='relu'),

        Dense(64, activation='relu'),

        Dense(64, activation='relu'),

        Dense(64, activation='relu'),

        Dense(3, activation='softmax')

    ])

    return model



model = create_model()

model.summary()
# In order to train a model, we have ti configure our model using compile()

# categorical_crossentropy) for our multiple-class classification problem



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# call model.fit() to fit our model to the training data.

history = model.fit(X_train, y_train, epochs=200, validation_split=0.25, batch_size=40, verbose=2)



%matplotlib inline

%config InlineBackend.figure_format = 'svg'



# helper function to plot metrics

def plot_metric(history, metric):

    train_metrics = history.history[metric]

    val_metrics = history.history['val_'+metric]

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics)

    plt.plot(epochs, val_metrics)

    plt.title('Training and validation '+ metric)

    plt.xlabel("Epochs")

    plt.ylabel(metric)

    plt.legend(["train_"+metric, 'val_'+metric])

    plt.show()



plot_metric(history, 'accuracy') # plots accuracy for train and validation

plot_metric(history, 'loss') # plots loss for train and validation
from tensorflow.keras.layers import Dropout

from tensorflow.keras.regularizers import l2



def create_regularized_model(factor, rate):

    model = Sequential([

        Dense(64, kernel_regularizer=l2(factor), activation="relu", input_shape=(4,)),

        Dropout(rate),

        Dense(128, kernel_regularizer=l2(factor), activation="relu"),

        Dropout(rate),

        Dense(128, kernel_regularizer=l2(factor), activation="relu"),

        Dropout(rate),

        Dense(128, kernel_regularizer=l2(factor), activation="relu"),

        Dropout(rate),

        Dense(64, kernel_regularizer=l2(factor), activation="relu"),

        Dropout(rate),

        Dense(64, kernel_regularizer=l2(factor), activation="relu"),

        Dropout(rate),

        Dense(64, kernel_regularizer=l2(factor), activation="relu"),

        Dropout(rate),

        Dense(3, activation='softmax')

    ])

    return model



model = create_regularized_model(1e-5, 0.3)

model.summary()

# First configure model using model.compile()

model.compile(

    optimizer='adam', 

    loss='categorical_crossentropy', 

    metrics=['accuracy']

)

# Then, train the model with fit()

history = model.fit(

    X_train, 

    y_train, 

    epochs=200, 

    validation_split=0.25, 

    batch_size=40, 

    verbose=2

)
plot_metric(history, 'accuracy') # plots accuracy for train and validation

plot_metric(history, 'loss') # plots loss for train and validation