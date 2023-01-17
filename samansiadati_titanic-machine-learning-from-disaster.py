import pandas as pd
df_train=pd.read_csv('../input/titanic/train.csv')
df_test=pd.read_csv('../input/titanic/test.csv')
df_train.head(20)
df_test
X_train_0 = pd.get_dummies(df_train[['Parch','Pclass','Survived','Sex','Age']])
X_train_0
corr=X_train_0.corr()
corr
import seaborn as sns
sns.heatmap(corr, linewidths = 0.5, annot=True, center=0, cmap="YlGnBu")
X_train = X_train_0.drop(columns=['Survived'])
X_train
X_train.isnull().values.ravel().sum()
X_train.isna().sum()
X_train.isna().any()
X_train.fillna(X_train.mean(), inplace=True)
X_train.isna().any()
y_train=df_train['Survived']
y_train
X_test= pd.get_dummies(df_test[['Parch','Pclass','Sex','Age']])
X_test.isnull().values.ravel().sum()
X_test.isna().sum()
X_test.isna().any()
X_test.fillna(X_train.mean(), inplace=True)
X_test.isna().any()
from sklearn.linear_model import LinearRegression
model_lreg = LinearRegression()
model_lreg.fit(X_train,y_train)
y_test = model_lreg.predict(X_test)
y_test
import numpy as np
np.shape(y_test)
Prediction = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction
Prediction.to_csv('databizx_submission_lreg.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model_rf.fit(X_train,y_train)
y_test = model_rf.predict(X_test)
y_test
import numpy as np
np.shape(y_test)
Prediction1 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction1
Prediction1.to_csv('databizx_submission_rf.csv', index=False)
from xgboost import XGBClassifier
model_xgb= XGBClassifier()
model_xgb.fit(X_train,y_train)
y_test = model_xgb.predict(X_test)
y_test
Prediction2 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction2
Prediction2.to_csv('databizx_submission_xgb.csv', index=False)
from sklearn.linear_model import LogisticRegression
model_logr = LogisticRegression(random_state=0)
model_logr.fit(X_train,y_train)
y_test = model_logr.predict(X_test)
y_test
Prediction3 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction3
Prediction3.to_csv('databizx_submission_logr.csv', index=False)
from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(X_train,y_train)
y_test = model_gnb.predict(X_test)
y_test
Prediction4 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction4
Prediction4.to_csv('databizx_submission_gnb.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_test = model_knn.predict(X_test)
y_test
Prediction5 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction5
Prediction5.to_csv('databizx_submission_knn.csv', index=False)
from sklearn import svm
model_svm= svm.SVC(kernel='linear')
model_svm.fit(X_train,y_train)
y_test = model_svm.predict(X_test)
y_test
Prediction6 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction6
Prediction6.to_csv('databizx_submission_svm.csv', index=False)
from keras.models import Sequential
from keras import layers
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
import matplotlib.pyplot as plt
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)
Prediction7 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_test})
Prediction7
Prediction7.to_csv('databizx_submission_krs.csv', index=False)