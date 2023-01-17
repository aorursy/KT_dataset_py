import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/diabetes.csv')
print(df.info())
# Plotting the counts for the 'Outcome' column (class labels)
sb.countplot(x='Outcome', data=df)
# We replace the 0s in each column by the columnar mean.
for column in set(df.columns).difference({'Pregnancies', 'Outcome'}):
    df[column] = df[column].replace(0, df[column].mean())
# Displaying the heatmap.
sb.heatmap(df.corr())
print(df.head())
# Converting the dataframe into a numpy matrix.
df_values = df.values

# Shuffling rows of the matrix.
np.random.shuffle(df_values)
# Splitting the first N-1 columns as X.
x = df_values[:,:-1]

# Splitting the last column as Y.
y = df_values[:, -1].reshape(x.shape[0], 1)

print(x.shape)
print(y.shape)

from sklearn.utils import class_weight

# Computing the class weights.
# Note: This returns an ndarray.
weights = class_weight.compute_class_weight('balanced', np.unique(y), y.ravel()).tolist()

# Converting the ndarray to a dict.
weights_dict = {
    i: weights[i] for i in range(len(weights))
}

print("Class weights: ", weights_dict)
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Input
import keras.regularizers
# Instantiate the model.
model = Sequential()

# Add the input layer and the output layer.
# The '1' indicates the number of output units.
# The 'input_shape' is where we specify the dimensionality of our input instances.
# The 'kernel_regularizer' specifies the strength of the L2 regularization.
model.add(Dense(1, input_shape=(x.shape[1], ), kernel_regularizer=keras.regularizers.l2(0.017)))

# Adding the BatchNorm layer.
model.add(BatchNormalization())

# Adding the final activation, i.e., sigmoid.
model.add(Activation('sigmoid'))

# Printing the model summary.
print(model.summary())
# Mean, columnar axis.
x_mean = np.mean(x, axis=0, keepdims=True)

# Std. Deviation, columnar axis.
x_std = np.std(x, axis=0, keepdims=True)

# Normalizing.
x = (x - x_mean)/x_std

print(x[:5, :])
from sklearn.model_selection import train_test_split

# Split the model into a 0.9-0.1 train-test split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)
model.compile(loss='binary_crossentropy', optimizer='adam')
print('Model compiled!')
from keras.callbacks import EarlyStopping

# Initialize the Early Stopper.
stopper = EarlyStopping(monitor='val_loss', mode='min', patience=3)

# Fit the data to the model and get the per-batch metric history.
history = model.fit(x_train, y_train, validation_split=0.1, 
                    batch_size=128, epochs=700, 
                    callbacks=[stopper], class_weight=weights_dict, verbose=1)
# Plot the training loss.
plt.plot(history.history['loss'], 'r-')

# Plot the validation loss.
plt.plot(history.history['val_loss'], 'b-')

# X-axis label.
plt.xlabel('Epochs')

# Y-axis label.
plt.ylabel('Cost')

# Graph legend.
plt.legend(["Training loss", "Validation loss"])

# Graph title.
plt.title('Loss Graph')

plt.show()
# Initialize variables.
tp = 0
fp = 0
fn = 0
tn = 0

# Get the predictions for the test inputs.
# One critical thing to note here is that, unlike scikit-learn,
# Keras will return the non-rounded prediction confidence
# probabilities.Therefore, rounding-off is critical.
predictions = model.predict(x_test)

# The hyperparameter that controls the tradeoff between how
# 'precise' the model is v/s how 'safe' the model is.
pr_hyperparameter = 0.5

# Rounding-off the predictions.
predictions[predictions > pr_hyperparameter] = 1
predictions[predictions <= pr_hyperparameter] = 0

# Computing the precision and recall.
for i in range(predictions.shape[0]):
    if y_test[i][0] == 1 and predictions[i][0] == 1:
        tp += 1
    elif y_test[i][0] == 1 and predictions[i][0] == 0:
        fn += 1
    elif y_test[i][0] == 0 and predictions[i][0] == 1:
        fp += 1
    else:
        tn += 1

pr_positive = tp/(tp + fp + 1e-8)
re_postive = tp/(tp + fn + 1e-8)
pr_negative = tn/(tn + fn + 1e-8)
re_negative = tn/(tn + fp + 1e-8)

# Computing the F1 scores.
f1 = (2*pr_positive*re_postive)/(pr_positive + re_postive + 1e-8)
f1_neg = (2*pr_negative*re_negative)/(pr_negative + re_negative + 1e-8)

print("F1 score (y=1): {}".format(f1))
print("F1 score (y=0): {}".format(f1_neg))
from sklearn import metrics

# Print the detailed classification report.
print(metrics.classification_report(y_true=y_test, y_pred=predictions))

# Compute the confusion matrix.
conf_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=predictions)

# Print the confusion matrix.
print(conf_matrix)
# Display the heatmap for the confusion matrix.
sb.heatmap(conf_matrix)
