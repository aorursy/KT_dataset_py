import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#Load training data
df= pd.read_csv("../input/train.csv")
Y= df["label"]
X= df.drop("label",axis=1)
# One Hot Encoding for Y
from keras.utils.np_utils import to_categorical
Y_cat =to_categorical(Y)
Y_cat.shape
# No. of digits
sns.countplot(Y)
print(Y.value_counts())
#>> Check for null value
X.isnull().any().any()

#>> Normaliztion
# X values are ranging from 0 to 255    
X.describe()
X =X /255.0
#>> Reshaping Data
# keras require a shape 4D tensor
#[Batch Size, Height of Image, Width of Image, No. of Color Channels]

X=X.values.reshape(-1,28,28,1)
X.shape
# Change Value of num to plot Different Images
num =4
plt.imshow(X[num][:,:,0],cmap="gray")
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y_cat, test_size = 0.1)
# Importing model config
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop
model= Sequential()

model.add(Conv2D(32,(4,4),padding="Same",activation="relu", input_shape=(28,28,1)  ))
model.add(Conv2D(32, (4,4), padding="Same", activation="relu"))
model.add(MaxPool2D())

model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding="Same",activation="relu" ))
model.add(Conv2D(32, (3,3), padding="Same", activation="relu"))
model.add(MaxPool2D(strides=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))

model.compile(RMSprop(lr=0.001),"categorical_crossentropy",metrics=["accuracy"]   )
model.summary()
# Set a learning rate annealer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
lr_reduction= ReduceLROnPlateau(monitor='val_acc',factor=0.4, min_lr=0.00001,patience=1,verbose=1)
# Training model with lr reduction and Early stopping
history =model.fit(X_train,Y_train,batch_size=64, epochs=15, validation_split=0.1,verbose=1,
         callbacks=[lr_reduction])
historydf=pd.DataFrame(history.history, index=history.epoch)
historydf.plot()
# Training model with 5 more epochs reduction and Early stopping
history =model.fit(X_train,Y_train,batch_size=64, epochs=5, validation_split=0.1,verbose=1,
         callbacks=[lr_reduction])
## Saving model
from keras.models import load_model
model.save("mnist_5.h5")
#model= load_model("mnist_5.h5")
# Predict from model on Validation data
Y_pred = model.predict(X_val)

# argmax for predicted value 
y_predicted= np.argmax(Y_pred,axis = 1) 

# argmax for true values
y_true = np.argmax(Y_val,axis = 1)
# Validation data Score
from sklearn.metrics import accuracy_score
accuracy_score(y_true,y_predicted)
# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, rangee):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(rangee, rangee, rotation=40)
    plt.yticks(rangee, rangee)

   
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")
                

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# confusion matrix
conf_matrix = confusion_matrix(y_true, y_predicted) 
#print(conf_matrix)

# plot the confusion matrix
plot_confusion_matrix(conf_matrix, rangee = range(10)) 
# Finding wrong predicted images
error = y_true - y_predicted !=0

Y_true_err = y_true[error]
Y_pred_err = y_predicted[error]
X_val_err =  X_val[error]
print("Wrong preicted Y true values",Y_true_err)
# 0 for first wrong predicted image
num = 0

plt.imshow(X_val_err[num][:,:,0],cmap="gray")
plt.show()
print("True Value: ",Y_true_err[num] )
print("Predicted Value: ",Y_pred_err[num])
# Retraining model on Validation data 

history2 =model.fit(X_val,Y_val,batch_size=32, epochs=2, validation_split=0.1,verbose=1,
         callbacks=[lr_reduction,EarlyStopping(monitor='loss', patience=2)])
# Again, Checking Accuracy score on Validation data

y_pred2=model.predict(X_train)
y_predd2=np.argmax(y_pred2,axis=1)
y_truee2 = np.argmax(Y_train,axis=1)

accuracy_score(y_truee2,y_predd2)
# Importing test.csv
df_tst = pd.read_csv("../input/test.csv")

# Reshaping test data
df_tst=df_tst.values.reshape(-1,28,28,1)

# Predicting with model
df_tst_result=model.predict(df_tst)
y_result=np.argmax(df_tst_result,axis=1)
range_n=np.arange(1,len(y_result) + 1 )

final_results=pd.concat([pd.DataFrame(range_n) , pd.DataFrame(y_result)],axis=1)
final_results.columns=(["ImageId","Label"])

final_results.to_csv("result.csv")
# Upload "result.csv"