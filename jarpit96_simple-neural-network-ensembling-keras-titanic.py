#Import Packages
import pandas as pd
import numpy as np

#Keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

#Sklearn
from sklearn.model_selection import StratifiedKFold
#Load Data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.columns.values)
train_df.head()
print(test_df.columns.values)
test_df.head()
#Dropping columns
print("Before: ", train_df.shape, test_df.shape)
train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
print("After: ", train_df.shape, test_df.shape)
def print_empty_cells(dataset):
    print("Empty Cells->")
    cols = dataset.columns.values
    for col in cols:
        print(col, dataset[col].isnull().sum())
#print empty/null number of cells in each column
print_empty_cells(dataset = train_df)
#Fill Null or Empty values with default values
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].dropna().median()) # Median Age
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port) #Most Frequent Port S
#print empty/null number of cells in each column
print_empty_cells(dataset = test_df)
#Fill Null or Empty values with default values
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].dropna().median()) # Median Age
test_df['Fare'] = test_df['Fare'].dropna() 
#Maps for Sex and Embarked
sex_mapping = {'male' : 0, 'female' : 1}
embarked_mapping = {'S' : 0, 'Q' : 1, 'C' : 2}

#Categorical to Numerical Sex and Embarked, Fill NA with Most Frequent Female and S
train_df['Sex'] = train_df['Sex'].map(sex_mapping)
train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
test_df['Sex'] = test_df['Sex'].map(sex_mapping)
test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)

print(train_df.head())
#Extracting Test and Train Sets for NN, Dataframe to Numpy
X_train = train_df.drop(['Survived'], axis=1).values
Y_train = train_df['Survived'].values
X_test = test_df.drop(['PassengerId'], axis = 1).values
#Config Parameters
num_epochs = 200
num_cv_epochs = num_epochs
train_batch_size = 32
test_batch_size = 32
folds = 10
def get_model():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(7,)))
    m.add(Dense(20, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    return m
#Make Keras Model and Fit on Training Data
model = get_model()
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=train_batch_size, verbose=1)
def cross_validation(X_train, Y_train, X_test, num_cv_epochs, k = 5):
    print("------------Cross Validation--------")
    k = max(k, 2) #Minimum 2 folds
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    cvscores = [] #metric scores/accuracy for each iteration
    y_pred = np.zeros((X_test.shape[0], 1)) #sum of predicted probabilities by models trained on different k-1 folds 
    for train, test in kfold.split(X_train, Y_train): #for every iteration 
        model_acc = get_model() #get a new keras model
        model_acc.fit(X_train[train], Y_train[train], epochs=num_cv_epochs, batch_size=train_batch_size, verbose=0) #fit/train on k-1 folds
        scores = model_acc.evaluate(X_train[test], Y_train[test], verbose=0) #evaluate on kth fold
        y_pred += model_acc.predict(X_test) #predict on test dataset for soft voting/ensembling
        print("%s: %.2f%%" % (model_acc.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%%" % (np.mean(cvscores)))
    return y_pred
#Evaluate Model On Train Data To Find Model Accuracy
#KFold Cross Validation
Y_pred = cross_validation(X_train=X_train, Y_train=Y_train, X_test=X_test, num_cv_epochs=num_cv_epochs, k = folds)
#Evaluate Test Data to Get Prediction
Y_pred += model.predict(X_test, batch_size=32)
Y_pred = Y_pred.reshape((Y_pred.shape[0],)) #reshape (418,1) to (418,)
Y_pred = Y_pred / (folds+1) #Average predicted probabilty
Y_pred = [int(p > 0.5) for p in Y_pred] #Converting class probabilities to Binary value
#Make submission Dataframe and Save to file
submission = pd.DataFrame({ "PassengerId": test_df["PassengerId"], "Survived": Y_pred })
submission.to_csv('submission.csv', index=False)
