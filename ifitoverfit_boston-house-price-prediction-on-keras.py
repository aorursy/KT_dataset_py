from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# add test data into train data, I will use k fold
x_train = np.concatenate((x_train,x_test))
y_train = np.concatenate((y_train,y_test))

# mean normalize features, so I prevented overshooting minimum loss
x_train = StandardScaler().fit_transform(x_train)

feature_size = x_train[0].__len__()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers

from sklearn.model_selection import KFold

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib

# Instantiate the cross validator
skf = KFold(n_splits=5, shuffle=True)

csv_scores = []
i = 1
for train, test in skf.split(x_train, y_train):
    print("Train on %d. validation split\n" % i)
    i += 1
    
    # Clear model, and create it
    model = None
    model = Sequential([
        Dense(32, input_shape=(feature_size,), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        #Dropout(0.1),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(1)
    ])
    
    # compile model
    model.compile(optimizer='Adam',
        loss='mean_absolute_error',
        metrics=['mae'])
    
    # train model
    history = model.fit(x_train[train], y_train[train], epochs=20, batch_size=16, validation_data=(x_train[test], y_train[test]))

    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    
    accuracy_history = history.history['mean_absolute_error']
    val_accuracy_history = history.history['val_mean_absolute_error']
    
    csv_scores.append(val_accuracy_history[-1])

    # plot losses
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # plot metrics
    plt.plot(accuracy_history)
    plt.plot(val_accuracy_history)
    plt.title('Metrics')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    
print("Average mean absolute error across kfold splits: %.4f (+/- %.2f%%)" % (np.mean(csv_scores), np.std(csv_scores)))