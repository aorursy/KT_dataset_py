#import matplotlib.pyplot as plt

import numpy as np, pandas as pd

import tensorflow as tf



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
df = px.data.iris()

df
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

fig.show()
fig = px.scatter_3d(df, x="sepal_width", y="sepal_length", z="petal_width", color="species")

fig.show()
X = np.array(df[['sepal_width', 'sepal_length', 'petal_width', 'petal_length']])

X
y = np.array(df['species_id'].values) - 1

y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model_input = tf.keras.Input(shape=(4), name='data_in')

x = tf.keras.layers.Dense(20, activation='relu')(model_input)

x = tf.keras.layers.Dense(3, activation='sigmoid')(x)



model_output = x

model = tf.keras.Model(model_input, model_output, name='aNetwork')



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'acc')

model.summary()
history = model.fit(X_train, y_train, 

                    validation_data = (X_test, y_test),

                    epochs=100)
df_history = pd.DataFrame(history.history)

df_history
df_history = df_history.reset_index().rename(columns={'index':'Epoch'})



fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['val_loss'], name = 'val_loss'), row=1, col=1)

fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['loss'], name = 'loss'),     row=1, col=1)



fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['val_acc'], name = 'val_acc'),row=1, col=2)

fig.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['acc'], name = 'acc'),row=1, col=2)



fig.show()
probs_test = model.predict(X_test)

probs_test
pred_test = np.argmax(probs_test, axis=1)

pred_test
target_names = df["species"].unique()

print(classification_report(y_test, pred_test, target_names=target_names))
import matplotlib.pyplot as plt

import itertools

from sklearn.metrics import confusion_matrix



def pretty_print_conf_matrix(y_true, y_pred, 

                             classes,

                             normalize=False,

                             title='Confusion matrix',

                             cmap=plt.cm.Blues):

    """

    Mostly stolen from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py



    Normalization changed, classification_report stats added below plot

    """



    cm = confusion_matrix(y_true, y_pred)



    # Configure Confusion Matrix Plot Aesthetics (no text yet) 

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=14)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    plt.ylabel('True label', fontsize=12)

    plt.xlabel('Predicted label', fontsize=12)



    # Calculate normalized values (so all cells sum to 1) if desired

    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]



    # Place Numbers as Text on Confusion Matrix Plot

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black",

                 fontsize=12)





    # Plot

    plt.tight_layout()
pretty_print_conf_matrix(y_test, pred_test, target_names)
# 20 / 100