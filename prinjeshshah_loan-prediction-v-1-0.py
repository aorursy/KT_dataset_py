import pandas as pd
import numpy as np
data = pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv',delimiter=',')
data.head()
data = data.drop('Loan_ID',axis=1)
list = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
list_2 = ['Male','Yes','3+','Not Graduate','Yes',1000,1000,0,12,0,'Semiurban','-']
for i in range(0,len(list)):
    if data[list[i]].isnull().values.any() == True:
        data[list[i]] = data[list[i]].fillna(list_2[i])
data.head()
data.info()
from sklearn import preprocessing
data.head()
list_3 = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
label_encoder = preprocessing.LabelEncoder()
for i in range(0,len(list_3)):
    data[list_3[i]] = label_encoder.fit_transform(data[list_3[i]].astype(str))

data.head()
data['Dependents'] = data['Dependents'].replace('3+',4)
data['Dependents'] = data['Dependents'].replace('0',0)
data['Dependents'] = data['Dependents'].replace('1',1)
data['Dependents'] = data['Dependents'].replace('2',2)
data['Dependents'].unique()
data.info()
data = pd.get_dummies(data, columns=['Gender','Self_Employed','Property_Area'])
data.head(5)
training_set = data.to_numpy()
np.shape(training_set)
from sklearn.model_selection import train_test_split
X_training = training_set[:,0:7] + training_set[:,9:16]
Y_training = training_set[:,8]
X_training = preprocessing.scale(X_training)
X_train, X_test, y_train, y_test = train_test_split(X_training,Y_training, test_size=0.3, random_state=42)
from keras.models import Sequential
from keras.layers import Dense
# Creates model
model = Sequential()
# 7 Neurons, expects input of 7 features.
model.add(Dense(7, input_dim=7, activation='relu'))
# Add another Densely Connected layer (every neuron connected to every neuron in the next layer)
model.add(Dense(7, activation='relu'))
# Add another Densely Connected layer with 5 Neurons.
model.add(Dense(5, activation='relu'))
# Last layer simple sigmoid function to output 0 or 1 (our label)
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,verbose=2,epochs=50)
from sklearn.metrics import confusion_matrix
test_prediction = model.predict(X_test)
for i in range(len(X_test)):
    print("X=%s, Predicted=%s" % (X_test[i],test_prediction[i]))
prediction_matrix = np.zeros(np.shape(test_prediction))
for i in range(len(test_prediction)):
    if test_prediction[i] >= 0.5:
        prediction_matrix[i] = 1
    else:
        prediction_matrix[i] = 0
confusion_matrix(y_test,prediction_matrix)