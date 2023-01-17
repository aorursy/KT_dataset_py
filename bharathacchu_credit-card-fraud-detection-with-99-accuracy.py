import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
%matplotlib inline
df = pd.read_csv('creditcard.csv')
df.head()
df.info()
df.shape
# Printing unique values present 
df.nunique()

df.describe().T
# check for null values
df.isnull().sum()
# Ploting the graph for transaction class distribution

count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

Labels = ['Legitimate','Fraud']

plt.xticks(range(2), Labels)

plt.xlabel("Class")

plt.ylabel("Frequency")
frauds = df.loc[df['Class'] == 1]
legitimate = df.loc[df['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(legitimate), "regular data points")
# amount of money used in different transaction classes
frauds.Amount.describe()
legitimate.Amount.describe()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(legitimate.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();
## Correlation

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df['Class'].value_counts()
x = df.drop('Class',axis=1)
y = df['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 0 ,stratify = y)
x_train.shape
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
x_test.shape
x_train.shape,x_test.shape
activation_function = 'relu'
loss = 'binary_crossentropy'
hidden_units_layer_1 = 34
hidden_units_layer_2 = 36
output_units = 1
batch_size = 5
epochs = 15
learning_rate = 0.001
model = Sequential([
    Dense(hidden_units_layer_1, input_dim=30, activation=activation_function),
    Dense(hidden_units_layer_2, activation=activation_function),
    Dense(output_units, activation='sigmoid')
])
model.summary()
optimizer = Adam(lr = 0.001)
model.compile(optimizer, loss= loss, metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
legend = ['Train','Validation']
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title("Training v/s Validation accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(legend);
loss = history.history['loss']
val_loss = history.history['val_loss']
legend = ['Training_Loss','Validation_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.title("Training Loss v/s Validation loss")
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(legend);
lss,acc =  model.evaluate(x_test, y_test, verbose=0)
print("Loss: {} and Accuracy : {}".format(lss,acc))

y_predicted = model.predict(x_test)

def plot_confusion_matrix(cm,classes,normalize=False,
                          title='confusion_matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=30)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('confusion  matrix without normalization')
    print(cm)
    
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('true label')

    plt.xlabel('predicted label')
predictions = model.predict_classes(x_test,batch_size=100,verbose=0)

cm = confusion_matrix(y_test, predictions)

import itertools
cm_plot_labels = ['Legitimate', 'Fraudlent']
_ = plot_confusion_matrix(cm,cm_plot_labels,title='Confusion_matrix')

# Do not commit
#model.save('creditcard_fraud_detection.h5')