# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import itertools

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import plot_model

from sklearn.metrics import classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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



    #print(cm)



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



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    

def printcfm(y_test,y_pred,title='confusion matrix'):

    cnf_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=['White','Red'],

                      title=title)
# Read in white wine data 

white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')



# Read in red wine data 

red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')



# Add `type` column to `red` with value 1

red['type'] = 1



# Add `type` column to `white` with value 0

white['type'] = 0



# Append `white` to `red`

wines = red.append(white, ignore_index=True)
# Print info on white wine

print(white.info())

print()

# Print info on red wine

print(red.info())
# First rows of `red` 

red.head()
# Last rows of `white`

white.tail()
# Take a sample of 5 rows of `red`

red.sample(5)
# Describe `white`

white.describe()
# Double check for null values in `red`

pd.isnull(red)
fig, ax = plt.subplots(1, 2)



ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")

ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")



fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)

ax[0].set_ylim([0, 1000])

ax[0].set_xlabel("Alcohol in % Vol")

ax[0].set_ylabel("Frequency")

ax[1].set_xlabel("Alcohol in % Vol")

ax[1].set_ylabel("Frequency")

#ax[0].legend(loc='best')

#ax[1].legend(loc='best')

fig.suptitle("Distribution of Alcohol in % Vol")



plt.show()
import matplotlib.pyplot as plt



fig, ax = plt.subplots(1, 2, figsize=(8, 4))



ax[0].scatter(red['quality'], red["sulphates"], color="red")

ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)



ax[0].set_title("Red Wine")

ax[1].set_title("White Wine")

ax[0].set_xlabel("Quality")

ax[1].set_xlabel("Quality")

ax[0].set_ylabel("Sulphates")

ax[1].set_ylabel("Sulphates")

ax[0].set_xlim([0,10])

ax[1].set_xlim([0,10])

ax[0].set_ylim([0,2.5])

ax[1].set_ylim([0,2.5])

fig.subplots_adjust(wspace=0.5)

fig.suptitle("Wine Quality by Amount of Sulphates")



plt.show()
import matplotlib.pyplot as plt

import numpy as np



np.random.seed(570)



redlabels = np.unique(red['quality'])

whitelabels = np.unique(white['quality'])



import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

redcolors = np.random.rand(6,4)

whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)



for i in range(len(redcolors)):

    redy = red['alcohol'][red.quality == redlabels[i]]

    redx = red['volatile acidity'][red.quality == redlabels[i]]

    ax[0].scatter(redx, redy, c=redcolors[i])

for i in range(len(whitecolors)):

    whitey = white['alcohol'][white.quality == whitelabels[i]]

    whitex = white['volatile acidity'][white.quality == whitelabels[i]]

    ax[1].scatter(whitex, whitey, c=whitecolors[i])

    

ax[0].set_title("Red Wine")

ax[1].set_title("White Wine")

ax[0].set_xlim([0,1.7])

ax[1].set_xlim([0,1.7])

ax[0].set_ylim([5,15.5])

ax[1].set_ylim([5,15.5])

ax[0].set_xlabel("Volatile Acidity")

ax[0].set_ylabel("Alcohol")

ax[1].set_xlabel("Volatile Acidity")

ax[1].set_ylabel("Alcohol") 

#ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))

ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))

#fig.suptitle("Alcohol - Volatile Acidity")

fig.subplots_adjust(top=0.85, wspace=0.7)



plt.show()
# # Add `type` column to `red` with value 1

# red['type'] = 1



# # Add `type` column to `white` with value 0

# white['type'] = 0



# # Append `white` to `red`

# wines = red.append(white, ignore_index=True)
import seaborn as sns

fig = plt.figure(figsize=(20, 10))                         

#sns.heatmap(pca.inverse_transform(np.eye(n_comp)), cbar=True, annot=True, cmap="hot")

sns.heatmap(wines.corr(), 

            cbar=True, annot=True, linewidths=.3, xticklabels=wines.columns, cmap="hot")

plt.ylabel('Principal component', fontsize=20);

#plt.xlabel('original feature index', fontsize=20);

plt.tick_params(axis='both', which='major', labelsize=18);

plt.tick_params(axis='both', which='minor', labelsize=12);

plt.show()
wines.type.value_counts().plot(kind='bar', title='Count Wine Type');
# Import `train_test_split` from `sklearn.model_selection`

from sklearn.model_selection import train_test_split



# Specify the data 

X=wines.iloc[:,0:11]



# Specify the target labels and flatten the array

y= np.ravel(wines.type)



# Split the data up in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Import `StandardScaler` from `sklearn.preprocessing`

from sklearn.preprocessing import StandardScaler



# Define the scaler 

scaler = StandardScaler().fit(X_train)



# Scale the train set

X_train = scaler.transform(X_train)



# Scale the test set

X_test = scaler.transform(X_test)
def classificationModel(input_shape):

    

    # Import `Sequential` from `keras.models`

    from keras.models import Sequential



    # Import `Dense` from `keras.layers`

    from keras.layers import Dense



    # Initialize the constructor

    model = Sequential()



    # Add an input layer 

    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))



    # Add one hidden layer 

    model.add(Dense(8, activation='relu'))



    # Add an output layer 

    model.add(Dense(1, activation='sigmoid'))

    

    return model
model = classificationModel(X.shape[1])
# Model output shape

model.output_shape
# Model summary

model.summary()
# Model config

model.get_config()
plot_model(model)
# List the number of weight tensors 

len(model.get_weights())
for weight in model.get_weights():

    print(weight.shape)
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
import time

start_time = time.time()

myEpochs=20

myBatch_size = 1

print("We are going to feed %d matrices in each epoch with %d samples each" %(X_train.shape[0]/myBatch_size,myBatch_size))

print()

model.fit(X_train, y_train,epochs=myEpochs, batch_size=myBatch_size, verbose=1)

# We have 4352 samples, with batch_size=1

print("--- %s seconds ---" % (time.time() - start_time))
import time

start_time = time.time()

myEpochs=20

myBatch_size = int(X_train.shape[0]/10)

print("We are going to feed %d matrices in each epoch with %d samples each" %(X_train.shape[0]/myBatch_size,myBatch_size))

print()

model1=model.fit(X_train, y_train,epochs=myEpochs, batch_size=myBatch_size, verbose=1)

# We have 4352 samples, with batch_size=1

print("--- %s seconds ---" % (time.time() - start_time))
from IPython.display import HTML

HTML('<center><iframe width="800" height="450" src="https://www.youtube.com/embed/kkWRbIb42Ms" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
y_pred = model.predict_classes(X_test)
y_pred[:10].T
y_test[:10]
score = model.evaluate(X_test, y_test,verbose=1)



print(score)
# Import the modules from `sklearn.metrics`

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score



# Confusion matrix

printcfm(y_test,y_pred,title='confusion matrix')
# Precision 

precision_score(y_test, y_pred)
# Recall

recall_score(y_test, y_pred)
# F1 score

f1_score(y_test,y_pred)
# Cohen's kappa

cohen_kappa_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
# Isolate target labels

y = wines.quality



# Isolate data

X = wines.drop('quality', axis=1) 
# Scale the data with `StandardScaler`

X = StandardScaler().fit_transform(X)
def regressionModel(input_shape):

    

    # Import `Sequential` from `keras.models`

    from keras.models import Sequential



    # Import `Dense` from `keras.layers`

    from keras.layers import Dense



    # Initialize the constructor

    model = Sequential()



    # Add an input layer 

    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))



    # Add one hidden layer 

    model.add(Dense(8, activation='relu'))



    # Add an output layer 

    model.add(Dense(1))

    

    return model
from sklearn.model_selection import StratifiedKFold



seed = 7

np.random.seed(seed)



myRegression = regressionModel(X.shape[1])

myRegression.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for train, test in kfold.split(X, y):

    myRegression.fit(X[train], y[train], epochs=10, verbose=1)
myRegression.fit(X[train], y[train], epochs=10, verbose=1)
mse_value, mae_value = myRegression.evaluate(X[test], y[test], verbose=0)



print(mse_value)
print(mae_value)
from sklearn.metrics import r2_score



r2_score(y_test, y_pred)