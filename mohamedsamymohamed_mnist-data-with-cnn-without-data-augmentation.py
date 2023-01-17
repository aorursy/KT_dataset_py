# import important library 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
%matplotlib inline
# load data 
train = pd.read_csv(r"../input/digit-recognizer/train.csv")
test = pd.read_csv(r"../input/digit-recognizer/test.csv")
X_train = train.drop("label" , axis = 1)
train_label =train["label"]
# check null value in train 
X_train.isnull().any().describe()
# check null value in test 
test.isnull().any().describe()
# visualize data 
sns.countplot(train_label)
pd.Series(train_label).value_counts()
X_train = X_train.values.reshape(-1 , 28 , 28 , 1)
X_train.shape
# visualize mnist data 
labels=["zero" , "one" , "two" , "three" , "four" , "five" , "six" , "seven" , "eight" , "nine"]
indcis = np.random.randint(0 , X_train.shape[0] , 14)
plt.figure(figsize=(20 , 20))
for i ,  index in enumerate(indcis):
    plt.subplot(7 , 7 , i+1)
    plt.imshow(X_train[index][: , : , 0] , cmap=plt.cm.binary)
    plt.title(str(labels[train_label[index]]))
plt.show()
    
# check shape of image 
shapes =[]
for i in range(X_train.shape[0]):
    shape = X_train[i].shape
    shapes.append(shape)
pd.Series(shapes).value_counts()
       
# data normalization 
X_train = X_train /255.0
test = test.values /255.0


# important callbacks function
class metricsloss(tf.keras.callbacks.Callback):
    def on_train_begin(self , epoch):
        print("Training : strat training")
    def on_epoch_end(self , epoch , logs = None):
        if (logs["accuracy"]>0.995):
            print("\ntraining is cancalled because accuracy reached at 99.5%")
            self.model.stop_training =True
# make early stopping 
early_stopping = tf.keras.callbacks.EarlyStopping(
patience=3 , 
    monitor="val_accuracy" , 
    mode="max" , 
    min_delta=0.0001
)
# saving models 
check_point_path ="check_point_path/best_model_in_epoch{epoch}"
check_point = tf.keras.callbacks.ModelCheckpoint(
  filepath= check_point_path , 
    save_weights_only=True , 
    save_best_only=True , 
    
    monitor="val_accuracy" , 
    verbose=1,
    period=2,
    mode ="max"
    
)
# make learning rate decay
learning_rate_decay =tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.1, 
            patience=3, 
            verbose=0, 
            mode='max', 
            min_delta=0.0001, 
            cooldown=0, 
            min_lr=0.00001)
# build model 
def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=120 ,input_shape = input_shape , kernel_size=(3 , 3)  , strides=(1 , 1) , activation="relu" ))
    model.add(tf.keras.layers.MaxPool2D((2 , 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(130 , (3 , 3) , activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D((2 , 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(140 , activation=tf.nn.relu , kernel_initializer="he_uniform"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(140 , activation=tf.nn.relu , kernel_initializer="random_uniform"))
    model.add(tf.keras.layers.Dense(50 , activation=tf.nn.relu ,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean = 0.0001 , stddev=0.005)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10 , activation=tf.nn.softmax))
    return model
# compile model 
def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer , loss ="sparse_categorical_crossentropy" , metrics=["accuracy"])
# split train data into train and validation and test 
from sklearn.model_selection import train_test_split
X_train , X_val , train_label , val_label = train_test_split(X_train , train_label , 
                                                            shuffle = True , random_state =33 , 
                                                            test_size = 0.2)

X_train.shape
X_val.shape
# split validation into val nd test 
X_val , X_test , val_label , test_label = train_test_split(X_val , val_label , random_state=33 , 
                                                          shuffle = True , test_size =0.5)
# fit model 
model = build_model(input_shape=(28 , 28 , 1))
model.summary()
#combile model 
compile_model(model)
history =model.fit(X_train , train_label , epochs=15 , validation_data=(X_val , val_label) , 
         batch_size=50 , callbacks=[ metricsloss(),early_stopping , check_point , learning_rate_decay])
pd.DataFrame(history.history)
# visualize model training for training
plt.plot(history.epoch , history.history["accuracy"]  , "r" , label="accuracy")
plt.plot(history.epoch , history.history["loss"] , "g" , label ="loss")
plt.xlabel("epochs")
plt.title("loss and accuracy in training")
plt.legend()
plt.show()

# visualize model training for validation
plt.plot(history.epoch , history.history["val_accuracy"]  , "r" , label="val_accuracy")
plt.plot(history.epoch , history.history["val_loss"] , "g" , label ="val_loss")
plt.xlabel("epochs")
plt.title("loss and accuracy in validation")
plt.legend()
plt.show()
# make prediction 
y_pred = model.predict(X_test)
# predict test data 
plt.figure(figsize=(30 , 30))
indecis = np.random.randint(0 , X_test.shape[0] , 25)
for j , i in enumerate(indecis):
    plt.subplot(5 , 5 , j+1)
    plt.imshow(X_test[i].reshape(28 , 28 ) , cmap=plt.cm.binary)
    plt.title(labels[y_pred[i].argmax()])
plt.show()

y_predicted =[y_pred[i].argmax() for i in range(y_pred.shape[0])]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_label , y_predicted)
sns.heatmap(cm)
pd.DataFrame(data=cm , index=[f"actual {i}"  for i in labels ], columns =[f"prdicted {j}" for j in labels])
# mislead value 
mislead =test_label-y_predicted
index_of_mislead=[]
x = 0
for i in mislead:
    if i !=0:
        index_of_mislead.append(x)
    x+=1
# number of mislead value 
len(index_of_mislead)
# visualize mislead value 
plt.figure(figsize=(50 , 50))
for i , j in enumerate(index_of_mislead[0:25]):
    plt.subplot(5 , 5 , i+1 )
    plt.imshow(X_test[j].reshape(28 , 28) , cmap=plt.cm.binary)
    plt.title(f"actual {labels[list(test_label)[j]]} , but bredicted {labels[y_predicted[j]]}",fontdict={"fontsize":31} )
plt.show()
