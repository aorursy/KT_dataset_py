#importing libraries
import pandas as pd 
import numpy as np
import keras
import tensorflow
import seaborn as sns
import matplotlib.pyplot as plt
#loading our dataset
df=pd.read_csv("../input/creditcard.csv")
df.tail()
#Heatmap to see the correlations between the variables
plt.figure(figsize=(15,10))
sns.heatmap(df.corr())
print("there is no correlation between the variables")
from sklearn.preprocessing import StandardScaler

#We are going to standarize the column Amount due the range of values it has.
df['normAmount']= StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df= df.drop(['Amount'],axis=1)
df= df.drop(['Time'],axis=1)
#Splitting the dataset into X and Y
X= df.iloc[:,df.columns != 'Class']
Y= df.iloc[:,df.columns == 'Class']
#Splitting the dataset into the train set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=2019)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test =np.array(y_test)
X_train.shape
#importing the libraries of the DNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Defining the neural network
model = Sequential([
    Dense(units=16,input_dim=29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(units=20,activation='relu'),
    Dense(units=24,activation='relu'),
    Dense(units=1,activation='sigmoid'),
])
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)
score= model.evaluate(X_test,y_test)
print(score)
y_pred= model.predict(X_test)
y_test=pd.DataFrame(y_test)
#Defining the confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#Confusion matrix of our Test set
c_mat=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(c_mat,classes=[0,1])
#Confusion matrix of the dataset
y_pred=model.predict(X)
y_expected=pd.DataFrame(Y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
#creating an index for the fraudulent and normal transactions
fraud_index=np.array(df[df.Class==1].index)
normal_index=df[df.Class == 0].index
count_fraud=len(fraud_index)

rand_normal_index = np.random.choice(normal_index,count_fraud,replace=False)
rand_normal_index = np.array(rand_normal_index)

undersample_index=np.concatenate([fraud_index,rand_normal_index])
print(len(undersample_index))
#undersampling the dataset
under_sample_data = df.iloc[undersample_index,:]
X_undersample = under_sample_data.iloc[:,under_sample_data.columns !='Class']
Y_undersample = under_sample_data.iloc[:,under_sample_data.columns =='Class']

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_undersample,Y_undersample,test_size=0.3,random_state=2019)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test =np.array(y_test)
#training the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)
#confusion matrix of the test (for the undersampling)
y_pred =model.predict(X_test)
y_expected=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
#confusion matrix for the dataset (for the undersampling)
y_pred=model.predict(X)
y_expected=pd.DataFrame(Y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
#creating the oversample
from imblearn.over_sampling import SMOTE
X_resample, y_resample = SMOTE().fit_sample(X,Y.values.ravel())
X_resample = pd.DataFrame(X_resample)
y_resample = pd.DataFrame(y_resample)
#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3,random_state=1492)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test =np.array(y_test)
#training the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)
#confusion matrix of the test (oversampling)
y_pred =model.predict(X_test)
y_expected=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
#confusion matrix of the dataset (oversampling)
y_pred =model.predict(X)
y_expected=pd.DataFrame(Y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
