!pip install tensorflow==2.0.0
import tensorflow as tf

print(tf.__version__)
#load the Mnist data

mnist = tf.keras.datasets.mnist



#Split into training and testing

(x_train,y_train) , (x_test,y_test) = mnist.load_data()

#Normalizing

x_train,x_test = x_train/255.0 , x_test/255.0
x_test.shape
#Build the model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #Shapeof our data

model.add(tf.keras.layers.Dense(130,activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))

#we are using softmax layer because we have to do multiclass classification

model.add(tf.keras.layers.Dense(10,activation='softmax')) 
#Compile the model

model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
r = model.fit(x_train,y_train,validation_data=(x_test, y_test), epochs=10)
import matplotlib.pyplot as plt

#To adjust size of Figure

plt.rcParams['figure.figsize'] = [10,5]



plt.plot(r.history['loss'],label='loss')

plt.plot(r.history['val_loss'],label='val_loss')

plt.legend()


plt.plot(r.history['accuracy'],label='accuracy')

plt.plot(r.history['val_accuracy'],label='val_accuracy')

plt.legend()
# Evaluate the model

print(model.evaluate(x_test, y_test))
#Plot confusion matrix

# Plot confusion matrix

from sklearn.metrics import confusion_matrix

import numpy as np

import itertools

plt.rcParams['figure.figsize'] = [10,7]



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

  """

  This function prints and plots the confusion matrix.

  Normalization can be applied by setting `normalize=True`.

  """

  if normalize:

      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

      print("Normalized confusion matrix")

  else:

      print('Confusion matrix, without normalization')



  print(cm)



  plt.imshow(cm, interpolation='nearest', cmap=cmap)

  plt.title(title)

  plt.colorbar()

  tick_marks = np.arange(len(classes))

  plt.xticks(tick_marks, classes, rotation=45)

  plt.yticks(tick_marks, classes)



  fmt = '.2f' if normalize else 'd'

  thresh = cm.max() / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

      plt.text(j, i, format(cm[i, j], fmt),

               horizontalalignment="center",

               color="white" if cm[i, j] > thresh else "black")



  plt.tight_layout()

  plt.ylabel('True label')

  plt.xlabel('Predicted label')

  plt.show()





p_test = model.predict(x_test).argmax(axis=1)

cm = confusion_matrix(y_test, p_test)

plot_confusion_matrix(cm, list(range(10)))



# Do these results make sense?

# It's easy to confuse 9 <--> 4, 9 <--> 7, 2 <--> 7, etc. 
#Find some accurate prediction

mp = np.where(p_test==y_test)[0]

i = np.random.choice(mp)

plt.imshow(x_test[i],cmap='gray')

plt.title("True label: %s Predicted: %s" % (y_test[i], p_test[i]));
#Find some misclassified prediction

mp = np.where(p_test!=y_test)[0]

i = np.random.choice(mp)

plt.imshow(x_test[i],cmap='gray')

plt.title("True label: %s Predicted: %s" % (y_test[i], p_test[i]));