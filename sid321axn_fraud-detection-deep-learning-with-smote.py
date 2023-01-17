import pandas as pd
import numpy as np
import keras
np.random.seed(2)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
# Function to plot Confusion Matrix (to be used later).
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
data=pd.read_csv('../input/creditcard.csv')
data.head()
data.tail()
# To check the count of fraudulent and normal transactions
sns.countplot(data['Class'],facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3), label = "Count")
# Now Checking actual number of fraudulent transactions
fraud_indices=np.array(data[data.Class==1].index)
no_records_fraud=len(fraud_indices)
normal_indices=np.array(data[data.Class==0].index)
no_records_normal=len(normal_indices)

print("No. of Fraudulent Transaction is {} and No. of Normal Transaction is {}".format(no_records_fraud, no_records_normal))
# To see the actual distribution of data 
sns.pairplot(data, hue = 'Class', vars = ['V1', 'V2', 'V3', 'V15', 'V18','Amount'] )
sns.kdeplot(data['Amount'],shade=True)
# To see the the actual distribution of Amount

fig=sns.FacetGrid(data,hue='Class',aspect=4)
fig.map(sns.kdeplot,'Amount',shade=True)
oldest=data['Amount'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
sns.scatterplot(x = 'Amount', y = 'V1',hue='Class',  data = data)

dataset2 = data.drop(columns = ['Class'])


dataset2.corrwith(data.Class).plot.bar(
        figsize = (20, 10), title = "Correlation with Class", fontsize = 20,
        rot = 45, grid = True)

data['normalized_amount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
# Dropping the actual Amount column from the dataset.
data=data.drop(['Amount'],axis=1)
# To check the dataset for changed column
data.head()
# I think Time is the irrelevant column so we are dropping the Time column from dataset.
data=data.drop(['Time'],axis=1)
data.head()
# Assigning X and Y 
X=data.iloc[:,data.columns!='Class']
y=data.iloc[:,data.columns=='Class']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
X_train.shape
X_test.shape
# As we have to supply the X test,X_Train,ytest,y_train into deep learning models so we have to convert it into numpy arrays.
X_train = np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
model = Sequential([
     #First Layer
     Dense(units=16, input_dim=29, activation='relu'),
      #Second Layer
     Dense(units=24,activation='relu'),
     Dropout(0.5),
      #Third Layer
     Dense(20,activation='relu'),
     #Fourth Layer
     Dense(24,activation='relu'),
     #Fifth Layer
     Dense(1,activation='sigmoid')  
    
    
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=15, epochs=5)
score=model.evaluate(X_test,y_test)
print(score)
y_pred=model.predict(X_test)
y_test=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=model.predict(X)
y_test=pd.DataFrame(y)
cnf_matrix=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
X=data.iloc[:,data.columns!='Class']
y=data.iloc[:,data.columns=='Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train.values.ravel())
y_pred=random_forest.predict(X_test)
cnf_matrix=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=random_forest.predict(X)

cnf_matrix=confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
X=data.iloc[:,data.columns!='Class']
y=data.iloc[:,data.columns=='Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
decc=DecisionTreeClassifier()
decc.fit(X_train,y_train.values.ravel())
y_pred=decc.predict(X_test)
decc.score(X_test,y_test)
cnf_matrix=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=decc.predict(X)

cnf_matrix=confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
fraud_indices=np.array(data[data.Class==1].index)
no_records_fraud=len(fraud_indices)
print(no_records_fraud)
normal_indices=data[data.Class==0].index
random_normal_indices=np.random.choice(normal_indices,no_records_fraud,replace=False)
random_normal_indices=np.array(random_normal_indices)
print(len(random_normal_indices))
under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))
under_sample_data=data.iloc[under_sample_indices,:]
under_sample_data.head()
X_undersample=under_sample_data.iloc[:,under_sample_data.columns!='Class']
y_undersample=under_sample_data.iloc[:,under_sample_data.columns=='Class']

X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size = 0.3, random_state=0)
X_train = np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
model = Sequential([
     Dense(units=16, input_dim=29, activation='relu'),
     Dense(units=24,activation='relu'),
     Dropout(0.5),
     Dense(20,activation='relu'),
     Dense(24,activation='relu'),
     Dense(1,activation='sigmoid')  
    
    
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=15, epochs=5)
y_pred=model.predict(X_test)
y_expected=pd.DataFrame(y_test)

cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=model.predict(X)

cnf_matrix=confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()

X_resample,y_resample=SMOTE().fit_sample(X,y.values.ravel())
y_resample=pd.DataFrame(y_resample)
X_resample=pd.DataFrame(X_resample)


X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size = 0.3, random_state=0)
X_train = np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=15, epochs=5)
y_pred=model.predict(X_test)
y_expected=pd.DataFrame(y_test)

cnf_matrix=confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=model.predict(X)

cnf_matrix=confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
