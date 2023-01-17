import pandas as pd
import numpy as np
data = pd.read_csv('../input/bank-marketing-dataset/bank.csv')
data.shape
data.head()
df_pivot = pd.DataFrame({'types': data.dtypes,
                         'nulls': data.isna().sum(),
                          '% nulls': data.isna().sum() / data.shape[0],
                          'size': data.shape[0],
                          'uniques': data.nunique()})
df_pivot
import category_encoders as ce
encoder = ce.BinaryEncoder()
df_binary = encoder.fit_transform(data.loc[:,['job','marital', 'education',
                                              'default', 'housing', 'loan',
                                              'contact','month','poutcome']])
df_binary.head()
int_columns = data.select_dtypes(include=['int'])
int_columns = int_columns.columns.values
columns = np.append(int_columns, 'deposit')
columns
data = pd.concat([df_binary, data.loc[:,columns]], axis=1)
data.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.loc[:,'job_0':'previous'], 
                                                    data.loc[:,'deposit'], test_size=0.2)
from sklearn import preprocessing
preprocessParams = preprocessing.StandardScaler().fit(x_train)
X_train_normalized = preprocessParams.transform(x_train)
X_test_normalized = preprocessParams.transform(x_test)
from keras import Sequential
from keras.layers import Dense
RN = Sequential()
RN.add(Dense(22,input_shape = X_train_normalized.shape[1:], activation = 'sigmoid'))
RN.add(Dense(10, activation = 'sigmoid'))
RN.add(Dense(2,activation = 'sigmoid'))
RN.summary()
# Dummy Transformation
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown = 'ignore')
encoder.fit(pd.DataFrame(y_train))

y_train = encoder.transform(pd.DataFrame(y_train)).toarray()
y_test = encoder.transform(pd.DataFrame(y_test)).toarray()
y_train
RN.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])
history = RN.fit(X_train_normalized,y_train, epochs = 125, validation_split=0.2) 
score = RN.evaluate(X_test_normalized, y_test, verbose = 0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# Graph training: cost train and validation
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss train and validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend();
from sklearn.metrics import confusion_matrix
y_test_predicted = RN.predict(X_test_normalized)
y_test_predicted_indexes = np.argmax(y_test_predicted,axis=1)
y_test_indexes = np.argmax(y_test, axis=1)
#Confusion Matrix
confMatrix = pd.DataFrame(confusion_matrix(y_test_predicted_indexes, y_test_indexes),
                           index=['0 - No','1 - Yes'],columns=['0 - No','1 - Yes'])

confMatrix.index.name = 'Actual'
confMatrix.columns.name= 'Predicted'
print(confMatrix)