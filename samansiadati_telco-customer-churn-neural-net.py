import pandas as pd
telco = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telco
telco.sample
telco.isnull().sum()
telco['gender'].value_counts()
churn = telco.Churn
churn.value_counts()
tenure = telco.tenure
tenure.idxmax()
import seaborn as sns
sns.distplot(tenure)
sns.distplot(telco.MonthlyCharges)
sns.pairplot(telco)
telco['Churn']
telco['Churn'] = telco['Churn'].map({'Yes': 1, 'No': 0})
telco['Churn']
telco_1 = telco.drop(['customerID', 'TotalCharges'], axis=1)
telco_1
telco_0 = pd.get_dummies(telco_1)
telco_0.columns
X = telco_0.drop(columns=['Churn'])
y = telco_0 ['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
from keras.models import Sequential
model_krs = Sequential()
from keras import layers
from keras.layers.core import Dropout
Input_Shape = X_train.shape[1]
Input_Shape
model_krs.add(layers.Dense(1024, input_shape=(Input_Shape,), activation='relu'))
##Dropout for not memorize or overfitting the train data
model_krs.add(Dropout(0.2)) 
model_krs.add(layers.Dense(1024, activation='relu'))
model_krs.add(Dropout(0.2)) 
model_krs.add(layers.Dense(1, activation='sigmoid'))
model_krs.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_krs.summary()
fit_keras = model_krs.fit(X_train, y_train,
          epochs=100,
          verbose=True,
          validation_data=(X_test, y_test),
          batch_size=30)
accuracy = model_krs.evaluate(X_train, y_train, verbose=False)
print("Training Score: {:.4f}".format(accuracy[0]))
print("Training Accuracy: {:.4f}".format(accuracy[1]))
accuracy = model_krs.evaluate(X_test, y_test, verbose=False)
print("Testing Score: {:.4f}".format(accuracy[0]))
print("Testing Accuracy: {:.4f}".format(accuracy[1]))
def plot_history(fit_keras):
    acc = fit_keras.history['accuracy']
    val_acc = fit_keras.history['val_accuracy']
    loss = fit_keras.history['loss']
    val_loss = fit_keras.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Testing acc')
    plt.title('Training and Testing accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.legend()
import matplotlib.pyplot as plt
plot_history(fit_keras)