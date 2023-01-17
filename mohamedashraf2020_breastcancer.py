import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
y = pd.read_csv("/kaggle/input/tumorxray/CSV2.csv")
y = y.iloc[ : , 1]
len(y)
X  = pd.read_csv("/kaggle/input/tumorxray/CSV.csv")
X = X.iloc[: , 1:]
X = np.array(X)
y = np.array(y)
def CreateImage(FlattenZero):
    Reshaped = FlattenZero.reshape((150 , 150 , 3))
    return Reshaped
X = [CreateImage(X[row]) for row in range(len(X))]
X = np.array(X)
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train , X_test , y_tarin , y_test = train_test_split(X , y)
code = {"begnin":0 , "malignant":1 , "normal":2}
def GetCode(n):
    for x , y in code.items():
        if n==y:
            return  x
plt.figure(figsize = (20 , 20))
for n , i in enumerate(list(np.random.randint(0 , len(X_train) , 42) )):
    plt.subplot(6 , 7 , n+1)
    plt.imshow(X_train[i])
    plt.axis("off")
    plt.title(GetCode(y_tarin[i]))
import keras
VGG16 = keras.applications.vgg16.VGG16(input_shape = (150 , 150 , 3) ,include_top=False ) 
VGG16.trainable  = False 
Model = keras.Sequential([
    VGG16 ,
    keras.layers.Flatten(),
    keras.layers.Dense(units = 256 , activation = "relu") ,
    keras.layers.Dense(units = 128 , activation = "relu") ,
    keras.layers.Dense(units = 3 )
])
Model.compile(optimizer = "adam" , loss =  keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,  metrics = ["accuracy"])
model = Model.fit(X_train ,y_tarin , epochs = 20   , verbose = 2 , batch_size = 60 , validation_data=(X_test , y_test) , shuffle = True)
import matplotlib.pyplot as plt
plt.plot(model.history['accuracy'], label='accuracy')
plt.plot(model.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
from sklearn.metrics import confusion_matrix
Predict = np.argmax(Model.predict(X_test) , axis = 1)
CM = confusion_matrix(y_test , Predict)
sns.heatmap(CM  , annot = True)
Model.save("Cancer2.h5")