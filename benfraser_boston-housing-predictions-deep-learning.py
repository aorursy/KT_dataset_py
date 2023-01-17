import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
PATH =  "/kaggle/input/boston-house-prices/"
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 

                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']



housing_df = pd.read_csv(PATH + 'housing.csv', header=None, delimiter=r"\s+", names=column_names)



print("Shape of housing dataset: {0}".format(housing_df.shape))



housing_df.head(5)
train_data = housing_df.iloc[:404, :].copy()

test_data = housing_df.iloc[404:, :].copy()



X_train = train_data.iloc[:, :-1].copy()

y_train = train_data.iloc[:, -1:].copy()



X_test = test_data.iloc[:, :-1].copy()

y_test = test_data.iloc[:, -1:].copy()
X_train.describe()
def feature_normalisation(train_data, test_data):

    """ Normalize our dataframe features with zero mean and unit

        standard deviation """

    

    std_data = train_data.copy()

    

    mean = train_data.mean(axis=0)

    std_dev = train_data.std(axis=0)

    

    # centre data around zero mean and give unit std dev

    std_data -= mean

    std_data /= std_dev

    

    # if test data passed to func, convert test data using train mean / std dev

    test_data -= mean

    test_data /= std_dev

        

    return std_data, test_data
X_train, X_test = feature_normalisation(X_train, X_test)
ranf = RandomForestRegressor(random_state=1)

ranf.fit(X_train, y_train.values[:, 0])
columns = list(X_train.columns)



importances = ranf.feature_importances_

indices = np.argsort(importances)[::-1]

cols_ordered = []



for feat in range(X_train.shape[1]):

    print("{0:<5} {1:<25} {2:.5f}".format(feat + 1, columns[indices[feat]], importances[indices[feat]]))

    cols_ordered.append(columns[indices[feat]])

    

plt.figure(figsize=(6,4))

plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.xticks(range(X_train.shape[1]), cols_ordered, rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

plt.title("Random Forrest Feature Importances")

plt.show()
from keras import models

from keras import layers
def nn_model(dropout=False):

    """ Create a basic Deep NN for regression """

    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

    if dropout:

        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(64, activation='relu'))

    if dropout:

        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model
k = 4



num_val_samples = len(X_train) // k



epochs = 100



scores = []



# prepare validation and training partitions

for i in range(k):

    print('Cross-validation fold number {0}'.format(i))

    val_samples_x = X_train[i * num_val_samples: (i + 1) * num_val_samples]

    val_samples_y = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    

    print("X Val: {0}, y Val: {1}".format(val_samples_x.shape, val_samples_y.shape))

    

    train_samples_x = np.concatenate([X_train[:i * num_val_samples],

                                      X_train[(i + 1) * num_val_samples:]], axis=0)

    

    train_samples_y = np.concatenate([y_train[:i * num_val_samples], 

                                      y_train[(i + 1) * num_val_samples:]], axis=0)

    

    print("X Train: {0}, y Train: {1}".format(train_samples_x.shape, train_samples_y.shape))

    

    # instantiate model and fit training samples, then evaluate on val partition

    model = nn_model()

    model.fit(train_samples_x, train_samples_y, epochs=epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_samples_x, val_samples_y, verbose=0)

    scores.append(val_mae)
print(np.mean(scores))
k = 4



num_val_samples = len(X_train) // k



epochs = 100



mae_histories = []



# prepare validation and training partitions

for i in range(k):

    print('Cross-validation fold number {0}'.format(i))

    val_samples_x = X_train[i * num_val_samples: (i + 1) * num_val_samples]

    val_samples_y = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    

    print("X Val: {0}, y Val: {1}".format(val_samples_x.shape, val_samples_y.shape))

    

    train_samples_x = np.concatenate([X_train[:i * num_val_samples],

                                      X_train[(i + 1) * num_val_samples:]], axis=0)

    

    train_samples_y = np.concatenate([y_train[:i * num_val_samples], 

                                      y_train[(i + 1) * num_val_samples:]], axis=0)

    

    print("X Train: {0}, y Train: {1}".format(train_samples_x.shape, train_samples_y.shape))

    

    # instantiate model and fit training samples, then evaluate on val partition

    model = nn_model()

    history = model.fit(train_samples_x, train_samples_y, 

                        epochs=epochs, batch_size=1, 

                        verbose=0, validation_data=(val_samples_x, val_samples_y))

    

    val_mae_hist = history.history['val_mae']

    

    mae_histories.append(val_mae_hist)
average_mae_hist = [np.mean([x[i] for x in mae_histories]) for i in range(epochs)]
plt.figure(figsize=(10,6))

plt.plot(range(1, len(average_mae_hist) + 1), average_mae_hist)

plt.xlabel("Epochs")

plt.ylabel("Validation MAE")

plt.show()
plt.figure(figsize=(10,6))

plt.plot(range(1, len(average_mae_hist) + 1), average_mae_hist)

plt.xlabel("Epochs")

plt.ylabel("Validation MAE")

plt.xlim(0.0, 100.0)

plt.show()
k = 4



num_val_samples = len(X_train) // k



epochs = 100



reg_mae_histories = []



# prepare validation and training partitions

for i in range(k):

    print('Cross-validation fold number {0}'.format(i))

    val_samples_x = X_train[i * num_val_samples: (i + 1) * num_val_samples]

    val_samples_y = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    

    print("X Val: {0}, y Val: {1}".format(val_samples_x.shape, val_samples_y.shape))

    

    train_samples_x = np.concatenate([X_train[:i * num_val_samples],

                                      X_train[(i + 1) * num_val_samples:]], axis=0)

    

    train_samples_y = np.concatenate([y_train[:i * num_val_samples], 

                                      y_train[(i + 1) * num_val_samples:]], axis=0)

    

    print("X Train: {0}, y Train: {1}".format(train_samples_x.shape, train_samples_y.shape))

    

    # instantiate dropout regularised model and fit training samples with val data for eval

    model = nn_model(dropout=True)

    history = model.fit(train_samples_x, train_samples_y, 

                        epochs=epochs, batch_size=1, 

                        verbose=0, validation_data=(val_samples_x, val_samples_y))

    

    val_mae_hist = history.history['val_mae']

    

    reg_mae_histories.append(val_mae_hist)



average_reg_mae_hist = [np.mean([x[i] for x in reg_mae_histories]) for i in range(epochs)]
plt.figure(figsize=(10,6))

plt.plot(range(1, len(average_mae_hist) + 1), average_mae_hist, label='Original Model')

plt.plot(range(1, len(average_reg_mae_hist) + 1), average_reg_mae_hist, label='Regularised Model')

plt.xlabel("Epochs")

plt.ylabel("Validation MAE")

plt.xlim(1.0, 100.0)

plt.legend(loc='best')

plt.show()
# produce our deep NN model using dropout regularisation, trained on all training data

final_model = nn_model(dropout=True)

history = final_model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)
hist_dict = history.history



trg_loss = history.history['loss']

trg_acc = history.history['mae']



epochs = range(1, len(trg_acc) + 1)



fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(epochs, trg_loss, label='Training Loss')

ax[1].plot(epochs, trg_acc, label='Training MAE')

ax[0].set_ylabel('Training Loss')

ax[1].set_ylabel('Training MAE')



plt.show()
test_preds = final_model.predict(X_test)
test_mse, test_mae = final_model.evaluate(X_test, y_test, verbose=0)
print("Test set performance: \n- Test MSE: {0} \n- Test MAE: {1}".format(test_mse, test_mae))