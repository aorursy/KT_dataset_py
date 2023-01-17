"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Purpose:    Data Exploration

@author:    Kevin Trinh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pieChart(sdss_df):
    '''Plot a pie chart for label count.'''
    label_counts = sdss_df['class'].value_counts()
    colors = ['skyblue', 'red', 'gold']
    fig1, ax1 = plt.subplots()
    ax1.pie(label_counts, labels=['Galaxy', 'Stars', 'Quasars'],
            autopct='%1.2f%%', startangle=45, colors=colors)
    ax1.axis('equal')
    plt.title('SDSS Object Classes')
    plt.show()

def distribution(sdss_df, axes, feature, row):
    '''Plot the distribution of a space object w.r.t. a given feature.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    for i in range(len(labels)):
        label = labels[i]
        ax = sns.distplot(sdss_df.loc[sdss_df['class']==label, feature], 
                          kde=False, bins=30, ax=axes[row, i], color=colors[i])
        ax.set_title(label)
        if (i == 0):
            ax.set(ylabel='Count')
            
def equitorial(sdss_df, row):
    '''Plot equitorial coordinates of observations.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    label = labels[row]
    sns.lmplot(x='ra', y='dec', data=sdss_df.loc[sdss_df['class']==label],
               hue='class', palette=[colors[row]], scatter_kws={'s': 2}, 
               fit_reg=False, height=4, aspect=2)
    plt.ylabel('dec')
    plt.title('Equitorial coordinates')
    

def main():

    # read in SDSS data
    filepath = '../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv'
    sdss_df = pd.read_csv(filepath, encoding='utf-8')

    # define lists of relevant features
    geo = ['ra', 'dec']
    nonugriv = ['redshift', 'plate', 'mjd', 'fiberid']
    ugriv = ['u', 'g', 'r', 'i', 'z']

    # plot pie chart of label count
    pieChart(sdss_df)

    # plot equitorial coordinates of observations
    for row in range(3):
        equitorial(sdss_df, row)
        plt.show()
    
    # plot the distribution of non-geo and non-ugriv features
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 14))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(nonugriv)):
        feat = nonugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()
        
    # plot the distribution of ugriv features
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(ugriv)):
        feat = ugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()

main()
"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Phase:      Feature engineering and ML classification

Algorithm: Deep neural network (DNN)

Steps:      1) Import libraries
            2) Read, shuffle, and partition data
            3) Restructure data as inputs for DNN
            4) Feature Engineering
            5) Create and train DNN
            6) Make predictions on validation sets
            7) Fine-tune models for highest performance on validation set
            8) Make predictions on test set
            9) Evaluate model with confusion matrix

@author:    Kevin Trinh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.utils import np_utils
import seaborn as sns


# read in and shuffle SDSS data
filename = '../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv'
sdss_df = pd.read_csv(filename, encoding='utf-8')
sdss_df = sdss_df.sample(frac=1)

# drop physically insignificant columns
sdss_df = sdss_df.drop(['objid', 'specobjid', 'run', 'rerun', 'camcol',
                        'field'], axis=1)


# partition SDSS data (60% train, 20% validation, 20% test)
train_count = 60000
val_count = 20000
test_count = 20000

train_df = sdss_df.iloc[:train_count]
validation_df = sdss_df.iloc[train_count:train_count+val_count]
test_df = sdss_df.iloc[-test_count:]


# obtain feature dataframes
X_train = train_df.drop(['class'], axis=1)
X_validation = validation_df.drop(['class'], axis=1)
X_test = test_df.drop(['class'], axis=1)

# one-hot encode labels for DNN
le = LabelEncoder()
le.fit(sdss_df['class'])
encoded_Y = le.transform(sdss_df['class'])
onehot_labels = np_utils.to_categorical(encoded_Y)

y_train = onehot_labels[:train_count]
y_validation = onehot_labels[train_count:train_count+val_count]
y_test = onehot_labels[-test_count:]

# scale features
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler to training data only
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_validation.columns)

# apply principal component analysis to wavelength intensities
pca = PCA(n_components=3)
dfs = [X_train, X_validation, X_test]
for i in range(len(dfs)):
    df = dfs[i]
    ugriz = pca.fit_transform(df[['u', 'g', 'r', 'i', 'z']])
    df = pd.concat((df, pd.DataFrame(ugriz)), axis=1)
    df.rename({0: 'PCA1', 1: 'PCA2', 2: 'PCA3'}, axis=1, inplace=True)
    df.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)
    dfs[i] = df
X_train, X_validation, X_test = dfs

# create a deep neural network model
num_features = X_train.shape[1]
dnn = Sequential()
dnn.add(Dense(9, input_dim=num_features, activation='relu'))
dnn.add(Dropout(0.1))
dnn.add(Dense(9, activation='relu'))
dnn.add(Dropout(0.1))
dnn.add(Dense(9, activation='relu'))
dnn.add(Dropout(0.05))
dnn.add(Dense(6, activation='relu'))
dnn.add(Dropout(0.05))
dnn.add(Dense(6, activation='relu'))
dnn.add(Dense(6, activation='relu'))
dnn.add(Dense(3, activation='softmax', name='output'))

dnn.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['categorical_accuracy'])

# train DNN
my_epochs = 50
history = dnn.fit(X_train, y_train, epochs=my_epochs, batch_size=50,
                    validation_data=(X_validation, y_validation))

# plot model loss while training
epochs_arr = np.arange(1, my_epochs + 1, 1)
my_history = history.history
line1 = plt.plot(epochs_arr, my_history['loss'], 'r-', label='training loss')
line2 = plt.plot(epochs_arr, my_history['val_loss'], 'b-', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model loss')
plt.legend()
plt.show()

# plot model accuracy while training
line1 = plt.plot(epochs_arr, my_history['categorical_accuracy'], 'r-', label='training accuracy')
line2 = plt.plot(epochs_arr, my_history['val_categorical_accuracy'], 'b-', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model accuracy')
plt.legend()
plt.show()

preds = pd.DataFrame(dnn.predict(X_validation))
preds = preds.idxmax(axis=1)
y_validation = y_validation.dot([0,1,2])
model_acc = (preds == y_validation).sum().astype(float) / len(preds) * 100

print('Deep Neural Network')
print('Validation Accuracy: %3.5f' % (model_acc))
# make predictions on test set (DO ONLY ONCE)
preds = pd.DataFrame(dnn.predict(X_test))
preds = preds.idxmax(axis=1)
y_test = y_test.dot([0,1,2])
model_acc = (preds == y_test).sum().astype(float) / len(preds) * 100

print('Deep Neural Network')
print('Test Accuracy: %3.5f' % (model_acc))

# plot confusion matrix
labels = np.unique(sdss_df['class'])

ax = plt.subplot(1, 1, 1)
ax.set_aspect(1)
plt.subplots_adjust(wspace = 0.3)
sns.heatmap(confusion_matrix(y_test, preds), annot=True,
                  fmt='d', xticklabels = labels, yticklabels = labels,
                  cbar_kws={'orientation': 'horizontal'})
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Deep Neural Network')
plt.show()