import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from mpl_toolkits.mplot3d import Axes3D

#Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#plotting missing data
import missingno as msno

#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

#Neural network building libraries
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History 
from keras.utils import plot_model
from keras.optimizers import SGD
voice=pd.read_csv("../input/voicegender/voice.csv")
voice.head(5)
print("\n",voice.info())
voice.describe()
#visualizing no missing value.
msno.matrix(voice)
#creating a copy
data=voice.copy()
# Distribution of target varibles
colors = ['pink','Lightblue']
df = data[data.columns[-1]]
plt.pie(df.value_counts(),colors=colors,labels=['female','male'])
plt.axis('equal')
print (data['label'].value_counts())
#Radviz circle 
#Good to compare every feature
pd.plotting.radviz(data,"label")
# Pairplotting
sns.pairplot(data[['meanfreq', 'Q25', 'Q75',
                'skew', 'centroid', 'label']], 
                 hue='label', size=2)
data.drop('label' ,axis=1).hist(bins=30, figsize=(12,12))
pl.suptitle("Histogram for each numeric input variable")
plt.show()
#corelation matrix.
cor_mat= data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(15,15)
sns.heatmap(data=cor_mat,square=True,annot=True,cbar=True,cmap='Spectral')
# Convert string label to float : male = 1, female = 0
dict = {'label':{'male':1,'female':0}}      # label = column name
data.replace(dict,inplace = True)           # replace = str to numerical
x = data.loc[:, data.columns != 'label']
y = data.loc[:,'label']
array = data.values
X = array[:,0:20]
Y = array[:,20]
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Appending different Models to a list

models = []

models.append(( 'LR ', LogisticRegression()))
models.append(( 'SVC', SVC(kernel='linear', C=1.0, random_state=0)))
models.append(( 'LDA', LinearDiscriminantAnalysis()))
models.append(( 'KNN', KNeighborsClassifier(n_neighbors=20, p=2, metric='minkowski')))
models.append(( 'CLF', DecisionTreeClassifier(criterion="entropy",max_depth=3)))
models.append(( 'RFC', RandomForestClassifier(max_depth=2, random_state=0)))
models.append(( 'MLP', MLPClassifier(hidden_layer_sizes=(3,3),
                                     max_iter=3000, activation = 'relu',
                                     solver='adam',random_state=1)))
models.append(( 'GNB', GaussianNB()))
#Finding Mean Accuracy for Models

results = []
names = []
meanscore=[]
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: (%f)" % (name, cv_results.mean()*100)
    meanscore.append(cv_results.mean()*100)
    print("Mean Accuracy score", msg)

print("\nHighest Mean Accuracy is for the classifer LDA", max(meanscore))
plt.plot(names,meanscore,marker='o')
plt.xlabel('Models')
plt.ylabel('Model Accuracy')
plt.title('ML classifiers and Accuracy score',size=25)

classifier=Sequential()
history = History()

#number of input variables = 20 so input_dim is only for the first layer
classifier.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=20)) #first layer
classifier.add(Dense(output_dim=16,init='uniform',activation='relu'))   #first Hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))    #Second Hidden layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid')) #output layer

#Running the artificial neural network
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()
trained=classifier.fit(X_train,Y_train,batch_size=5,epochs=20,validation_split=0.2,callbacks=[history],shuffle=2)
y_pred=classifier.predict(X_train)
y_pred = np.round(y_pred)

print('Accuracy by the Neural Network on train dataset is',metrics.accuracy_score(y_pred,Y_train)*100,'%')

y_pred=classifier.predict(X_test)
y_pred = np.round(y_pred)

print('Accuracy by the Neural Network on test dataset is ',metrics.accuracy_score(y_pred,Y_test)*100,'%')
plt.plot(history.history['loss'], color = 'red',label='Variaton Loss over the epochs',)
plt.plot(history.history['accuracy'],color='green',label='Variation in Accuracy over the epochs')

plt.xlabel('Epochs')
plt.title('Loss/Accuracy VS Epoch on test Dataset using our model')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='best')
plt.show()
plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()
plt.plot(Y_test[-30:],linestyle='--',label='Actual value',linewidth=3,marker='o' ,markerfacecolor='green',markersize=15,color='green')
plt.plot(y_pred[-30:],linestyle='-.',label='Predicted value',linewidth=3,marker='o' ,markerfacecolor='red',markersize=10,color='red')
plt.title('Validating the Model for 30 voices',size=15)
plt.xlabel("Voice notes")
plt.ylabel("Male(1)/ female(0)")
plt.legend(loc='center left')