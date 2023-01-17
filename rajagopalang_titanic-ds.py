import pandas  as pd
from   keras.models import Sequential
from   keras.layers import Dense             

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/titanic/train.csv')
test_data  = pd.read_csv('../input/titanic/test.csv')
#===========================================================================
# select some features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch"]
#===========================================================================
# for the features that are categorical we use pd.get_dummies:
# "Convert categorical variable into dummy/indicator variables."
#===========================================================================
X_train       = pd.get_dummies(train_data[features])
print(X_train)
y_train       = train_data["Survived"]
final_X_test  = pd.get_dummies(test_data[features])
#===========================================================================
# parameters for keras
#===========================================================================
input_dim   = len(X_train.columns) # number of neurons in the input layer
n_neurons   = 50            # number of neurons in the first hidden layer
epochs      = 100           # number of training cycles
#===========================================================================
# keras model
#===========================================================================
model = Sequential()         # a model consisting of successive layers
# input layer
model.add(Dense(n_neurons, input_dim=input_dim, activation='relu'))
# output layer, with one neuron
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#===========================================================================
# train the model
#===========================================================================
model.fit(X_train, y_train, epochs=epochs)
#===========================================================================
# use the trained model to predict 'Survived' for the test data
#===========================================================================
predictions = model.predict(final_X_test)
# set a threshold of 50% for classification, i.e. >0.5 is True
# Note: the '*1' converts the Boolean array into an array containing 0 or 1
predictions = (predictions > 0.5)*1
#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.flatten()})
output.to_csv('submission.csv', index=False)

