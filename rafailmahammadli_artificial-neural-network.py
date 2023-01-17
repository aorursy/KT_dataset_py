import numpy as np
import  pandas as  pd
import tensorflow  as tf
# Importing the dataset
dataset = pd.read_csv('../input/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data

# Label encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

#One Hot Encoding the "Geography"  column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Splitting the dataset to the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#  Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Initializing ANN

ann = tf.keras.models.Sequential()

# Adding the input layer and first hidden layer

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


#Adding the Output layer 
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Training the ANN  on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

print(ann.predict(sc.transform([[1,0,0, 619, 0, 42, 2, 0, 1, 1, 1, 101348]])) > 0.5)



# Predicting the Test set results
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)),1))


# Making Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)

