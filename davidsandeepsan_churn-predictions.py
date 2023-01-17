#importing libraries

import numpy as np

import pandas as pd
# Importing the data for training

data = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Train File.csv')
data
# Number of columns in data 

data.columns
# Count of null values

data.isnull().sum()
# Summary of Data

data.describe()
data.info()
# Treating the null or NaN Values

data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
data.info()
data.head()
# Encoding categorical data -->>gender Column

from sklearn.preprocessing import LabelEncoder

labelencoder_gender = LabelEncoder()

gender= labelencoder_gender.fit_transform(data['gender'])



# Encoding categorical data -->>Partner Column

labelencoder_Partner = LabelEncoder()

Partner= labelencoder_Partner.fit_transform(data['Partner'])



# Encoding categorical data -->>Dependents Column

labelencoder_Dependents = LabelEncoder()

Dependents= labelencoder_Dependents.fit_transform(data['Dependents'])



# Encoding categorical data -->>PhoneService Column

labelencoder_PhoneService = LabelEncoder()

PhoneService= labelencoder_PhoneService.fit_transform(data['PhoneService'])



# Encoding categorical data -->>MultipleLines Column

labelencoder_MultipleLines = LabelEncoder()

MultipleLines= labelencoder_MultipleLines.fit_transform(data['MultipleLines'])

MultipleLines = pd.get_dummies(MultipleLines,drop_first=True)



# Encoding categorical data -->>InternetService Column

labelencoder_InternetService = LabelEncoder()

InternetService= labelencoder_InternetService.fit_transform(data['InternetService'])

InternetService = pd.get_dummies(InternetService,drop_first=True)



# Encoding categorical data -->>OnlineSecurity Column

labelencoder_OnlineSecurity = LabelEncoder()

OnlineSecurity= labelencoder_OnlineSecurity.fit_transform(data['OnlineSecurity'])

OnlineSecurity = pd.get_dummies(OnlineSecurity,drop_first=True)



# Encoding categorical data -->>OnlineBackup Column

labelencoder_OnlineBackup = LabelEncoder()

OnlineBackup= labelencoder_OnlineBackup.fit_transform(data['OnlineBackup'])

OnlineBackup = pd.get_dummies(OnlineBackup,drop_first=True)



# Encoding categorical data -->>DeviceProtection Column

labelencoder_DeviceProtection = LabelEncoder()

DeviceProtection= labelencoder_DeviceProtection.fit_transform(data['DeviceProtection'])

DeviceProtection = pd.get_dummies(DeviceProtection,drop_first=True)



# Encoding categorical data -->>TechSupport Column

labelencoder_TechSupport = LabelEncoder()

TechSupport= labelencoder_TechSupport.fit_transform(data['TechSupport'])

TechSupport = pd.get_dummies(TechSupport,drop_first=True)



# Encoding categorical data -->>StreamingTV Column

labelencoder_StreamingTV = LabelEncoder()

StreamingTV= labelencoder_StreamingTV.fit_transform(data['StreamingTV'])

StreamingTV = pd.get_dummies(StreamingTV,drop_first=True)



# Encoding categorical data -->>StreamingMovies Column

labelencoder_StreamingMovies = LabelEncoder()

StreamingMovies= labelencoder_StreamingMovies.fit_transform(data['StreamingMovies'])

StreamingMovies = pd.get_dummies(StreamingMovies,drop_first=True)



# Encoding categorical data -->>Contract Column

labelencoder_Contract = LabelEncoder()

Contract= labelencoder_Contract.fit_transform(data['Contract'])

Contract = pd.get_dummies(Contract,drop_first=True)





# Encoding categorical data -->>PaperlessBilling Column

labelencoder_PaperlessBilling = LabelEncoder()

PaperlessBilling= labelencoder_PaperlessBilling.fit_transform(data['PaperlessBilling'])



# Encoding categorical data -->>PaymentMethod Column

labelencoder_PaymentMethod = LabelEncoder()

PaymentMethod= labelencoder_PaymentMethod.fit_transform(data['PaymentMethod'])

PaymentMethod = pd.get_dummies(PaymentMethod,drop_first=True)



# Encoding categorical data -->>Churn Column

labelencoder_Churn = LabelEncoder()

Churn= labelencoder_Churn.fit_transform(data['Churn'])

X_train=pd.concat([pd.DataFrame(gender),

                   data['SeniorCitizen'],

                   pd.DataFrame(Partner),

                  pd.DataFrame(Dependents),

                  data['tenure'],

                  pd.DataFrame(PhoneService),

                  MultipleLines,

                  InternetService,

                  OnlineSecurity,

                  OnlineBackup,

                  DeviceProtection,

                  TechSupport,

                  StreamingTV,

                  StreamingMovies,

                  Contract,

                  pd.DataFrame(PaperlessBilling),

                  PaymentMethod,

                  data['MonthlyCharges'],

                  data['TotalCharges']]

,axis=1)
y_train=Churn
# Feature Scaling for training set

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
#test File

data = pd.read_csv('../input/predict-the-churn-for-customer-dataset/Train File.csv')
data.head()
data.columns
data.info()
# Treating the null or NaN Values

data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
data.info()
# Encoding categorical data -->>gender Column

from sklearn.preprocessing import LabelEncoder

labelencoder_gender = LabelEncoder()

gender= labelencoder_gender.fit_transform(data['gender'])



# Encoding categorical data -->>Partner Column

labelencoder_Partner = LabelEncoder()

Partner= labelencoder_Partner.fit_transform(data['Partner'])



# Encoding categorical data -->>Dependents Column

labelencoder_Dependents = LabelEncoder()

Dependents= labelencoder_Dependents.fit_transform(data['Dependents'])



# Encoding categorical data -->>PhoneService Column

labelencoder_PhoneService = LabelEncoder()

PhoneService= labelencoder_PhoneService.fit_transform(data['PhoneService'])



# Encoding categorical data -->>MultipleLines Column

labelencoder_MultipleLines = LabelEncoder()

MultipleLines= labelencoder_MultipleLines.fit_transform(data['MultipleLines'])

MultipleLines = pd.get_dummies(MultipleLines,drop_first=True)



# Encoding categorical data -->>InternetService Column

labelencoder_InternetService = LabelEncoder()

InternetService= labelencoder_InternetService.fit_transform(data['InternetService'])

InternetService = pd.get_dummies(InternetService,drop_first=True)



# Encoding categorical data -->>OnlineSecurity Column

labelencoder_OnlineSecurity = LabelEncoder()

OnlineSecurity= labelencoder_OnlineSecurity.fit_transform(data['OnlineSecurity'])

OnlineSecurity = pd.get_dummies(OnlineSecurity,drop_first=True)



# Encoding categorical data -->>OnlineBackup Column

labelencoder_OnlineBackup = LabelEncoder()

OnlineBackup= labelencoder_OnlineBackup.fit_transform(data['OnlineBackup'])

OnlineBackup = pd.get_dummies(OnlineBackup,drop_first=True)



# Encoding categorical data -->>DeviceProtection Column

labelencoder_DeviceProtection = LabelEncoder()

DeviceProtection= labelencoder_DeviceProtection.fit_transform(data['DeviceProtection'])

DeviceProtection = pd.get_dummies(DeviceProtection,drop_first=True)



# Encoding categorical data -->>TechSupport Column

labelencoder_TechSupport = LabelEncoder()

TechSupport= labelencoder_TechSupport.fit_transform(data['TechSupport'])

TechSupport = pd.get_dummies(TechSupport,drop_first=True)



# Encoding categorical data -->>StreamingTV Column

labelencoder_StreamingTV = LabelEncoder()

StreamingTV= labelencoder_StreamingTV.fit_transform(data['StreamingTV'])

StreamingTV = pd.get_dummies(StreamingTV,drop_first=True)



# Encoding categorical data -->>StreamingMovies Column

labelencoder_StreamingMovies = LabelEncoder()

StreamingMovies= labelencoder_StreamingMovies.fit_transform(data['StreamingMovies'])

StreamingMovies = pd.get_dummies(StreamingMovies,drop_first=True)



# Encoding categorical data -->>Contract Column

labelencoder_Contract = LabelEncoder()

Contract= labelencoder_Contract.fit_transform(data['Contract'])

Contract = pd.get_dummies(Contract,drop_first=True)





# Encoding categorical data -->>PaperlessBilling Column

labelencoder_PaperlessBilling = LabelEncoder()

PaperlessBilling= labelencoder_PaperlessBilling.fit_transform(data['PaperlessBilling'])



# Encoding categorical data -->>PaymentMethod Column

labelencoder_PaymentMethod = LabelEncoder()

PaymentMethod= labelencoder_PaymentMethod.fit_transform(data['PaymentMethod'])

PaymentMethod = pd.get_dummies(PaymentMethod,drop_first=True)
X_test=pd.concat([pd.DataFrame(gender),

                   data['SeniorCitizen'],

                   pd.DataFrame(Partner),

                  pd.DataFrame(Dependents),

                  data['tenure'],

                  pd.DataFrame(PhoneService),

                  MultipleLines,

                  InternetService,

                  OnlineSecurity,

                  OnlineBackup,

                  DeviceProtection,

                  TechSupport,

                  StreamingTV,

                  StreamingMovies,

                  Contract,

                  pd.DataFrame(PaperlessBilling),

                  PaymentMethod,

                  data['MonthlyCharges'],

                  data['TotalCharges']]

,axis=1)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_test = StandardScaler()

X_test = sc_test.fit_transform(X_test)
# Importing the Keras libraries and packages

# using artificial nueral networks 

# Artificial Neural Network

# Installing Theano

#!pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git



# Installing Tensorflow

#!pip install tensorflow



# Installing Keras

#!pip install --upgrade keras

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))



# Adding the second hidden layer

classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 300,callbacks=[EarlyStopping(patience=2)])
pred = classifier.predict(X_test)
Churn=pd.DataFrame(pred)

submission=pd.concat([data['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']>0.5, 'Yes', 'No')

submission.to_csv('DavidSubmission.csv')
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifierlr = LogisticRegression(random_state = 0)

classifierlr.fit(X_train, y_train)
pred = classifierlr.predict(X_test)
Churn=pd.DataFrame(pred)

submission=pd.concat([data['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']==1, 'Yes', 'No')

submission.to_csv('DavidSubmission1.csv')
# Fitting SVM to the Training set

from sklearn.svm import SVC

classifiersvm = SVC(kernel = 'linear', random_state = 0)

classifiersvm.fit(X_train, y_train)
pred = classifiersvm.predict(X_test)
Churn=pd.DataFrame(pred)

submission=pd.concat([data['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']==1, 'Yes', 'No')

submission.to_csv('DavidSubmission2.csv')
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifiernb = GaussianNB()

classifiernb.fit(X_train, y_train)
pred = classifiernb.predict(X_test)
Churn=pd.DataFrame(pred)

submission=pd.concat([data['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']==1, 'Yes', 'No')

submission.to_csv('DavidSubmission3.csv')
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifierdc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifierdc.fit(X_train, y_train)
pred = classifierdc.predict(X_test)
Churn=pd.DataFrame(pred)

submission=pd.concat([data['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']==1, 'Yes', 'No')

submission.to_csv('DavidSubmission4.csv')
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifierrc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifierrc.fit(X_train, y_train)
pred = classifierrc.predict(X_test)
Churn=pd.DataFrame(pred)

submission=pd.concat([data['customerID'],Churn],axis=1)

submission=submission.set_index("customerID", inplace = False)

submission=submission.rename(columns={0:'Churn'})

submission['Churn'] = np.where(submission['Churn']==1, 'Yes', 'No')

submission.to_csv('DavidSubmission5.csv')