import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from mpl_toolkits.axes_grid1 import Grid



import tensorflow



from tensorflow import keras, nn

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout



from tensorflow.keras.datasets import fashion_mnist
np.random.seed(42)
((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()
X_train.shape
labels_description = {

    'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

    'description': ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

}



labelsdf = pd.DataFrame(data=labels_description)

labelsdf.set_index('label', inplace=True)



labelsdf.T
fig = plt.figure(figsize=(12,6))



grid = Grid(fig, rect=111, nrows_ncols=(2,5), axes_pad=0.25, label_mode='L')

randoms = np.random.choice(X_train.shape[0], 10)



for ax in grid:

    sample, randoms = randoms[-1], randoms[:-1]

    ax.set_title(labelsdf.loc[[y_train[sample]]]['description'].values[0])

    ax.imshow(X_train[sample], cmap='gray')



plt.tight_layout()
plt.imshow(X_train[0], cmap='gray')

plt.colorbar()
X_train = X_train/float(255)

X_test = X_test/float(255)
model = Sequential([

    Flatten(input_shape=(28, 28)),

    Dense(256, activation=nn.relu),

    Dropout(0.2),

    Dense(10, activation=nn.softmax)

])



model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)
history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)
model.evaluate(X_test, y_test)
historydf = pd.DataFrame(data=history.history)

historydf = historydf.rename({

    'loss': 'trainning loss',

    'acc': 'trainning accuracy',

    'val_loss': 'validation loss',

    'val_acc': 'validation accuracy'

}, axis=1)
sns.set(style="whitegrid")



fig, ax = plt.subplots(1,2, figsize=(15, 5))



trainning_plot = sns.lineplot(data=historydf[['trainning loss', 'validation loss']],

             palette='Set2', ax=ax[0])

trainning_plot.set_title('Loss')

trainning_plot.legend(loc=3)



evaluation_plot = sns.lineplot(data=historydf[['trainning accuracy', 'validation accuracy']],

             palette='Set2_r', ax=ax[1])

evaluation_plot.set_title('Accuracy')

evaluation_plot.legend(loc=3)



fig.show()
model.save('model.h5')