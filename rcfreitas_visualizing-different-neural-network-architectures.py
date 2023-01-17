import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
import numpy as np

%matplotlib inline
def history_plot(hist_dict):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    
    palette = itertools.cycle(sns.color_palette())

    sns.set(style="darkgrid")
    sns.set_context("notebook", font_scale=1.75)

    plt.figure(figsize=(8,8))
    
    for label, hist in hist_dict.items():
        plt.plot(hist.history['acc'], color=next(palette), 
                 linestyle='dashed', lw=2, label='{0} - train'.format(label))
        plt.plot(hist.history['val_acc'], color=next(palette),
                 lw=2, label='{0} - val'.format(label))

    #plt.ylim([0.0, 1.05])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy rate')
    plt.title('Accuracy')
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8,8))
    
    for label, hist in hist_dict.items():
        plt.plot(hist.history['loss'], color=next(palette), 
                 linestyle='dashed', lw=2, label='{0} - train'.format(label))
        plt.plot(hist.history['val_loss'], color=next(palette),
                 lw=2, label='{0} - val'.format(label))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    #plt.legend(loc="upper right")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    plt.show()
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.drop('customerID', axis='columns', inplace=True)
data.info()
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
print(data.shape)
data.dropna(inplace=True)
print(data.shape)

columns = list(data.columns)
columns.remove('MonthlyCharges')
columns.remove('tenure')
columns.remove('TotalCharges')
print(columns)
from sklearn.preprocessing import LabelEncoder, StandardScaler
X = data[columns[:-1]]

le = LabelEncoder()
y = le.fit_transform(data['Churn'])

X = pd.get_dummies(X)

X['MonthlyCharges'] = data['MonthlyCharges']
X['tenure'] = data['tenure']
X['TotalCharges'] = data['TotalCharges']
X[['MonthlyCharges', 'tenure', 'TotalCharges']] = StandardScaler().fit_transform(data[['MonthlyCharges', 'tenure', 'TotalCharges']])

X.shape
from keras import utils
X = X.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
train_class_weights = dict(enumerate(class_weights))
print (train_class_weights, "\n")
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
y_train_rs = utils.to_categorical(y_train, num_classes=2)
y_val_rs = utils.to_categorical(y_val, num_classes=2)
y_test_rs = utils.to_categorical(y_test, num_classes=2)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout

history = {}
i=1
n_list = [2**i for i in range(5,12)]

for neurons in n_list:
    print('\rModel {0} of {1}'.format(i, len(n_list)))
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(45,)),
        Dense(8, activation='relu'),
        Dropout(0.75),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd', metrics=['accuracy'])
    history['{0}'.format(neurons)] = model.fit(X_train, y_train_rs,
                                              batch_size=32, epochs=200, verbose=0,
                                              class_weight = train_class_weights, 
                                              validation_data=(X_val, y_val_rs))
    i+=1
history_plot(history)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout

history = {}
i=1
n_list = [i for i in range(6)]

for n_layers in n_list:
    print('\rModel {0} of {1}'.format(i, len(n_list)))
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(45,)))
    
    for layer in range(n_layers):
        model.add(Dense(32, activation='relu'))
        
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd', metrics=['accuracy'])
    history['{0}'.format(n_layers+2)] = model.fit(X_train, y_train_rs,
                                              batch_size=32, epochs=200, verbose=0,
                                              class_weight = train_class_weights, 
                                              validation_data=(X_val, y_val_rs))
    i+=1
history_plot(history)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras import regularizers

history = {}
i=1
r_params = [10**(-i) for i in range(0,4)]

for r in r_params:
    print('\rModel {0} of {1}'.format(i, len(r_params)))
    l2 = regularizers.l2(r)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(45,)),
        Dense(32, activation='relu', kernel_regularizer=l2),
        Dense(8, activation='relu'),
        Dropout(0.75),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd', metrics=['accuracy'])
    history['{0}'.format(r)] = model.fit(X_train, y_train_rs,
                                              batch_size=32, epochs=200, verbose=0,
                                              class_weight = train_class_weights, 
                                              validation_data=(X_val, y_val_rs))
    i+=1
history_plot(history)