import numpy as np

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
%matplotlib inline



from scipy.spatial import distance
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from tensorflow.keras.utils  import plot_model, model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
DSPATH="../input/olivetti-faces/"
X = np.load(DSPATH+"olivetti_faces.npy")
y = np.load(DSPATH+"olivetti_faces_target.npy")

 
ThiefImage={}
ThiefImage["False"]= image.imread("../input/thief-images/False.jpg")
ThiefImage["True"]=image.imread("../input/thief-images/True.jpg")
 



 
ClASSES=np.unique(y)
# N_CLASSES=len(np.unique(labels))
# Split into train/val

StratifiedSplit = StratifiedShuffleSplit( test_size=0.4, random_state=0)
StratifiedSplit.get_n_splits(X, y)
for train_index, test_index in StratifiedSplit.split(X, y):
    X_train, X_test, y_train, y_test= X[train_index], X[test_index], y[train_index], y[test_index]
    
# X_train, X_test, y_train, y_test = train_test_split(    
#     X, y, test_size=.40, random_state=42)
 
import seaborn as sns
g = sns.countplot(y_train)


# calculate class mean 
class_mean=[]
for i in range(len(X_train)):
    class_mean.append(X_train[y_train==i].mean(axis = 0))
#Show Data

def ShowTrainingData(showNclasses=5):
    if showNclasses>=40:
        showNclasses=ClASSES
    rows,cols=2,4
    
    for i in range(showNclasses+1):
        fig,ax =  plt.subplots(rows,cols )
        j=0
        for face in X_train[y_train==i]:
            j+=1
            if j==cols:
                j=5
            ax=plt.subplot(rows,cols,j)
            ax.imshow(face ,'gray' )

        ax = plt.subplot(1,cols,cols)
       
        ax.imshow( class_mean[i], 'gray' )
        plt.xlabel("Class "+str(i)+" mean " )
        fig.tight_layout(pad=1.0)
        plt.show()
    
#Show Data

def ShowTrainingData2(showNclasses):
    if showNclasses>=40:
        showNclasses=ClASSES
    rows,cols=2,4
    
    for i in range(showNclasses+1):
        fig = plt.figure(figsize=(8, 4))
    
        j=0
        for face in X_train[y_train==i]:
            j+=1
            if j==cols:
                j=5
            fig.add_subplot(rows, cols, j)
            plt.imshow(face, cmap = plt.get_cmap('gray'))
 
            plt.axis('off')

        
        fig.add_subplot(1,cols,cols)
        plt.imshow(class_mean[i], cmap = plt.get_cmap('gray'))
        plt.title("class_mean {}".format(i), fontsize=16)
        plt.axis('off')
 #         fig.tight_layout(pad=1.0)


        plt.suptitle("There are 6 image for class {}".format(i), fontsize=15)
        plt.show()

    
def ShowPredictions(predic_Model,ShowNPredictions=5):
    
    if ShowNPredictions>=len(y_predictions):
        ShowNPredictions=len(y_predictions)
    rows,cols=1,3
 
    for index, row in y_predictions.iterrows():
        if (index>ShowNPredictions):
            break
            
        x=int(row["x"])
        actually=int(row["actually"])
        y_predic=int(row[predic_Model+"_predic"])
        IsTrue=str(row[predic_Model+"_True"])

        fig,ax =  plt.subplots(rows,cols )
        j=1
        ax=plt.subplot(rows,cols,j)
        ax.imshow(X_test[x] ,'gray' )
        plt.xlabel("Test Number :"+str(x)  )

        j=2
        ax=plt.subplot(rows,cols,j)
        ax.imshow(class_mean[y_predic] ,'gray' )
        plt.xlabel("Class "+str(y_predic)+" mean " )

        j=3
        ax=plt.subplot(rows,cols,j)
        ax.imshow(ThiefImage[IsTrue] ,'gray' )
        plt.xlabel("Class "+str(actually)+" mean " )



        fig.tight_layout(pad=2.0)
        plt.show()   
   
ShowTrainingData2(5)
distanceTable=np.array([(i,y_test[i],c,distance.euclidean(X_test[i].flatten() , class_mean[c].flatten() )) for c in ClASSES  for i in range(len(X_test))])
distanceTable
distanceTable=distanceTable.T
# distanceTable.shape=(4,6400)
 
d = {'x': distanceTable[0], 'actually':distanceTable[1],'KNN_predic':distanceTable[2],'distance':distanceTable[3]}
df= pd.DataFrame(data=d)
df.head()

df[df.x==0]
y_predictions=pd.merge(df ,df.groupby(["x","actually"]).distance.min(), how = 'inner',  on=["x","actually","distance"])
y_predictions["KNN_True"]=y_predictions["KNN_predic"]==y_predictions["actually"]
correct_predictions = np.nonzero(y_predictions["KNN_True"].values==1)[0]
incorrect_predictions = np.nonzero(y_predictions["KNN_True"].values==0)[0]
print(len(correct_predictions)," classified correctly")
print(len(incorrect_predictions)," classified incorrectly")
print("KNN_predic")
print("=============")

ShowPredictions("KNN",5)
X_train = X_train.reshape(-1,64,64,1)
X_test = X_test.reshape(-1,64,64,1)
 

print("X_train shape: ",X_train.shape,"y_train shape: ",y_train.shape)
print("x_test shape: ", X_test.shape,"y_test shape: ",y_test.shape)
model = Sequential()

model.add(Conv2D(filters = 20, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,1)))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 50, kernel_size = (6,6),padding = 'Same', 
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 150, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,1)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(40, activation = "softmax"))


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.7, 
                                            min_lr=0.00000000001)
early_stopping_monitor = EarlyStopping(patience=2)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])
epoch = 37
batch_size = 20

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.05, # Randomly zoom image 
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

history = model.fit_generator(
                              datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epoch, 
                              validation_data = (X_test,y_test),
                              verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction]
                             )
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("CNN_predic")
print("=============")

score = model.evaluate(X_test,y_test,batch_size=32)
print(score)
CNN_predic=model.predict_classes(X_test)
X_test = X_test.reshape(-1,64,64)
del y_predictions

distanceTable=np.array([(i,y_test[i],CNN_predic[i] )  for i in range(len(X_test))])
distanceTable=distanceTable.T
d = {'x': distanceTable[0], 'actually':distanceTable[1],'CNN_predic':distanceTable[2] }
y_predictions= pd.DataFrame(data=d)
y_predictions.head()
y_predictions["CNN_True"]=y_predictions["CNN_predic"]==y_predictions["actually"]


y_predictions.head(100)
correct_predictions = np.nonzero(y_predictions["CNN_True"].values==1)[0]
incorrect_predictions = np.nonzero(y_predictions["CNN_True"].values==0)[0]
print(len(correct_predictions)," classified correctly")
print(len(incorrect_predictions)," classified incorrectly")



ShowPredictions("CNN")



