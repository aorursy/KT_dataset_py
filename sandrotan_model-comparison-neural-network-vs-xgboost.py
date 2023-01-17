import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, BatchNormalization

from keras.optimizers import Adam
# Load csv file into numpy arrays

data = np.genfromtxt('../input/winequalityred/winequality-red.csv', delimiter=',', unpack=True, skip_header=1)



# Slice input and output (assume quality >= 7 is good and labelled as 1, otherwise 0)

X = data[:-1]

y = np.array([data[-1] >= 7]).astype(int).reshape(len(data[-1]), 1)



# Normalize the input

X = StandardScaler().fit_transform(X)



# Split the train and test sets

X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=1)
# Construct neutral network

model = Sequential([

    Dense(16, input_dim=11, activation='relu', kernel_initializer='he_uniform'),

    Dense(32, activation='relu'),

    Dense(32, activation='relu'),

    BatchNormalization(),

    Dense(1, activation='sigmoid'),

])
# Set Adam optimizer parameters

opt_adam = Adam(learning_rate=0.012, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)



# Compile the model

model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])



# Run the model

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500,

                    verbose=0,

                    batch_size=32)
# Evaluate prediction accuracy

_, train_acc = model.evaluate(X_train, y_train, verbose=0)

_, test_acc = model.evaluate(X_test, y_test, verbose=0)

print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))
# Plot loss during training

plt.subplot(2, 1, 1)

plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()



# Plot accuracy during training

plt.subplot(2, 1, 2)

plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
for lr in np.arange(0, 1, 0.1):

    model = XGBClassifier(learning_rate=lr, n_estimators=100)



    eval_set = [(X_train, y_train), (X_test, y_test)]

    eval_metric = ["auc", "error"]

    result = model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=False)



    # print(result.evals_result()['validation_1']['error'])



    # make predictions for test data

    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]



    # evaluate predictions

    accuracy = accuracy_score(predictions, y_test)

    print("Accuracy: %.2f%%" % (accuracy * 100.0), 'learning rate: %.4f' % lr)
plt.plot(result.evals_result()['validation_1']['error'])

plt.show()