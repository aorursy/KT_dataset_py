import pandas as pd
train = pd.read_csv('../input/train.csv')
train.isnull().sum()
# Select columns
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
# Fill missing age
train['Age'].fillna(train.groupby(['Sex', 'Pclass'])['Age'].transform('mean'), inplace=True)
# Fill missing embarked
train['Embarked'].fillna(value='S', inplace=True)
# Convert sex to 0 and 1
train.replace({'Sex': {'male': 1, 'female': 0}}, inplace=True)
# Convert embarked to onehot encoding
train = pd.get_dummies(train, columns=['Embarked'])
train.head()
train.shape
train.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = train[features]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg.score(X_train, y_train))
print(logreg.score(X_test, y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))
test = pd.read_csv('../input/test.csv')
test.isnull().sum()
# Select columns
test_cal = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
# Fill missing age
test_cal['Age'].fillna(test_cal.groupby(['Sex', 'Pclass'])['Age'].transform('mean'), inplace=True)
# Fill missing fare
test_cal['Fare'].fillna(test_cal['Fare'].mean(), inplace=True)
# Convert sex to 0 and 1
test_cal.replace({'Sex': {'male': 1, 'female': 0}}, inplace=True)
# Convert embarked to onehot encoding
test_cal = pd.get_dummies(test_cal, columns=['Embarked'])
test_cal = scaler.fit_transform(test_cal)
final = knn.predict(test_cal)
print(final)
test['Survived'] = final
test.tail(20)
test.to_csv('titanic_submit.csv', columns=['PassengerId', 'Survived'])
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=30, batch_size=30, verbose=2)
import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('Train')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
scores = model.evaluate(x=X_test, y=y_test)
print(scores)