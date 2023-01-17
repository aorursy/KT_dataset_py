from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import pandas as pd 

#reading the data

X = pd.read_csv('train.csv')
X_test_full = pd.read_csv('test.csv')
submission = pd.read_csv('gender_submission.csv')

#cleaning the target

X.dropna(axis=0, subset=['Survived'], inplace=True)
y = X.Survived
X.drop(['Survived'], axis=1, inplace=True)


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
#filling the missing data
X['Age'].fillna(X['Fare'].median(), inplace = True) 
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace = True)

#selecting numerical cols
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

#selecting low cardinality cols with objects
low_cardinality_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

#selecting only chosen data
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

#easy one-hot-encoding
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


model = Sequential() #dense constructor NN model
model.add(Dense(64, input_dim=11, activation='sigmoid')) #input dense parameter number must be the same as number of using features
model.add(Dense(48, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) 
early_stopping = EarlyStopping(monitor='val_loss', patience=500) #impelent of early stop of the training procedure, if val_loss numbers will grop up
model.fit(X_train, y_train, epochs = 1000, batch_size=128, validation_data=(X_valid, y_valid), shuffle=True, callbacks=[early_stopping])

preds_test = model.predict(X_test)


#since our problem is  binary classification, the answer must be 0 or 1, so we must convert our predictions to those numbers.
rounded = [int(round(x[0])) for x in preds_test] 


pd.DataFrame({'PassengerId':submission.PassengerId,'Survived':rounded}).to_csv('submission.csv',index=False)
