import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/data.csv')
data.shape
data.head()
data = data.drop(["id","Unnamed: 32"],axis=1)
data.head()
data.describe()
data.corr()
sns.heatmap(data.corr(), annot=True)
ax = sns.countplot(y,label="Count")       # M = 212, B = 357
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
plt.show()
#just checking if there is any null value..
pd.isnull(data).sum()
#mapping function to map different string objects to integer
#def mapping(data,feature):
#    featureMap=dict()
 #   count=0
  #  for i in sorted(data[feature].unique(),reverse=True):
   #     featureMap[i]=count
    #    count=count+1
  #  data[feature]=data[feature].map(featureMap)
   # return data
   # Malignant is mapped to 0, Benign is mapped to 1
   # data=mapping(data,feature="diagnosis")
# Mapping the values
#map_diagnosis = { 'M' : 0, 'B' : 1}

# Setting the map to the data_frame
#data['diagnosis'] = data['diagnosis'].map(map_diagnosis)

# Let's see what we have done
#data.head()
#mapping function to map different string objects to integer
def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data
data=mapping(data,feature="diagnosis")
data.sample(5)
X =data[[ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']] 
X.head()
Y = data[['diagnosis']]
Y.head()
#another approach :)

# y includes our labels and x includes our features
 #  y = data.diagnosis                          
# M or B 
  # list = ['Unnamed: 32','id','diagnosis']
     # x = data.drop(list,axis = 1 )
  # x.head()

data.shape
X.shape
Y.shape
# another approcah
#divide dataset into x(input) and y(output)
   # X=data.drop(["diagnosis"],axis=1)
   # y=data["diagnosis"]
#divide dataset into training set, cross validation set, and test set
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
trainX.shape
testX.shape
trainY.shape
testY.shape
valX.shape
valY.shape
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.models import Sequential
visible = Input(shape=(30,))
hidden1 = Dense(30, activation='relu')(visible)
hidden2 = Dense(50, activation='relu')(hidden1)
hidden3 = Dense(100, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
model.fit(trainX,trainY,epochs=40,callbacks=[plot_losses])
scores=model.evaluate(valX,valY)
scores
print("Loss:",scores[0])
print("Accuracy",scores[1]*100)
predY=model.predict(np.array(testX))
