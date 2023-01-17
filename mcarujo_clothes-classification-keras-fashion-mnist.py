# Importing libs

import tensorflow

from tensorflow import keras



import pandas as pd

import plotly.express as px

import matplotlib.pyplot as plt



# Importing database to a variable

dataset = keras.datasets.fashion_mnist

((x_train,y_train ),(x_test,y_test)) = dataset.load_data()



# Creating a label for each type of clothes

labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Size of our train dataset

print("Train dataset",x_train.shape)

# Size of our test dataset

print("Test dataset",x_test.shape)
# Show some examples

plt.figure(figsize=(25,15))

for img in range(40):

    plt.subplot(4,10,img+1)

    plt.imshow(x_train[img])

    plt.title(labels[y_train[img]])  
x_train = x_train/255

x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0],*(28,28,1))

x_test = x_test.reshape(x_test.shape[0],*(28,28,1))
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard





model = Sequential()

model.add(

    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1))

)

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.1))



model.add(Flatten())

model.add(Dense(units=1024, activation="relu"))

model.add(Dense(units=1024, activation="relu"))

model.add(Dense(units=10, activation="softmax"))



model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['mse','accuracy'])

# Trainig and returning back the results.

history = model.fit(x_train,y_train, epochs = 50)
def plot_train_statics(history, tags = None):

    dfs = []

    tags =  tags if tags != None else list(history.history) 

    

    for tag in tags:

        df = pd.DataFrame(history.history[tag], columns = ['value'])

        df['epoch'] = df.index

        df['color'] = tag

        dfs.append(df)



    df = pd.concat(dfs)

    df.value = df.value.round(4)

    fig = px.line(df, x="epoch", y="value", color='color', title='Training Curve')

    fig.show()
plot_train_statics(history,['loss', 'accuracy'])
test = model.evaluate(x_test,y_test)

loss, mse, acc = test

print("The results of the training are -> loss: {}, mse: {}, and an accuracy: {}".format(round(loss,3), round(mse,3), round(acc,3)))
import seaborn as sns

from sklearn.metrics import confusion_matrix



prediction = model.predict_classes(x_test)

cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, cmap='Reds',fmt="d").set(xlabel="Predict", ylabel="Real")
from sklearn.metrics import classification_report

import plotly.figure_factory as ff



num_classes = 10

ff.create_table(pd.DataFrame(classification_report(y_test,prediction, target_names=labels,output_dict=True)).round(4).T, index=True)