#importing the required modules

from keras.models import Sequential

from keras.layers import Dense

from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use("fivethirtyeight")

import numpy as np

import seaborn as sns

import pydot
#loading the dataset

filename = "/kaggle/input/diabetes/diabetes.csv"

dataset = pd.read_csv(filename)

dataset.head()
dataset.describe().transpose()
sns.heatmap(dataset.corr())
dataset.corr()
#separating the data

x_data=dataset.iloc[:,0:8].values

y_data=dataset.iloc[:,8:].values

#splitting the data into training and testing

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,random_state=10)
#creating model

def model_creation(opt='adam',init='uniform'):

    model=Sequential()

    model.add(Dense(12,input_dim=8,kernel_initializer=init,activation='relu'))

    model.add(Dense(8,kernel_initializer=init,activation='relu'))

    model.add(Dense(8,kernel_initializer=init,activation='relu'))

    model.add(Dense(1,kernel_initializer=init,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    return model



model=model_creation()

#we created 3 hidden layers and each input and output layer
model.summary()
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#fitting our model 

values=model.fit(x_train,y_train,epochs=150,batch_size=10,verbose=1)
scores=model.evaluate(x_data,y_data)

print(model.metrics_names[1],scores[1]*100)
#predicying our model on test data

y_predict=model.predict(x_test,batch_size=10)

y_predicted_labes=[]
#changing outputs in the last layer in network

for i in range(len(y_predict)):

    if y_predict[i]>=0.5:

        y_predicted_labes.append(1)

    else:

        y_predicted_labes.append(0)
# analysing the results

print(confusion_matrix(y_test,y_predicted_labes))

print(classification_report(y_test,y_predicted_labes))
#keys in model

print(values.history.keys())
#plotting accuracy and loss 

plt.plot(values.history["loss"],color="blue",linewidth=2)

plt.plot(values.history["accuracy"],color="red",linewidth=2)

plt.xlabel("no of epochs")

plt.ylabel("loss and accuracy rate")

plt.title("model analysing")

plt.legend(["loss","accuracy"],loc="upper left")

plt.show()
#Evaluating our model with StratifiedKfold and Cross val score

#importing required modules

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold,cross_val_score
model2=KerasClassifier(build_fn=model_creation,epochs=150,batch_size=10,verbose=1)

kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=5)

results=cross_val_score(model2,x_data,y_data,cv=kfold,n_jobs=3)

print(f"accuracy ==>{results} , mean accuracy is{results.mean()}")