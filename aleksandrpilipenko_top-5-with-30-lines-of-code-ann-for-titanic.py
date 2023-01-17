# Import modules for data preparation
import numpy as np
import pandas as pd

#load data
dataset = pd.read_csv('train.csv')

# drop empty rows in Age column
dataset = dataset.dropna(subset=['Age'])  

# slice data to X and Y
X = dataset.iloc[:, [2,4,5,6,7,9]].values 
y = dataset.iloc[:, 1].values   

# encode data for transform sex from male\female to 0\1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Import Keras 
import keras
from keras.models import Sequential 
from keras.layers import Dense   
   
#Initialising the ANN
classifier = Sequential()

#Adding Imput layer and the first hidden layer
classifier.add(Dense(output_dim = 4,  init = 'uniform',  activation = 'relu', input_dim = 8))
classifier.add(Dense(output_dim = 4,  init = 'uniform',  activation = 'relu'))
classifier.add(Dense(output_dim = 1,  init = 'uniform', activation = 'sigmoid'  ))
classifier.compile(optimizer = 'adam',  loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Putting the ANN to the test set 
classifier.fit(X_train, y_train, batch_size=10, epochs=500)
import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')

dataset['Age'] = dataset['Age'].fillna(28)

dataset['Cabin'] = dataset['Cabin'].fillna(0)
for i in range(len(dataset)):
    if dataset['Cabin'][i] != 0 :
        dataset['Cabin'][i] = 1

dataset = dataset.dropna(subset=['Embarked'])

X = dataset.iloc[:, [2,4,5,6,7,9,10,11]].values 
y = dataset.iloc[:, 1].values  

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 

labelencoder_X_2 = LabelEncoder()
X[:, 7] = labelencoder_X_2.fit_transform(X[:, 7]) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential 
from keras.layers import Dense      

classifier.add(Dense(output_dim = 8,  init = 'uniform',  activation = 'relu', input_dim = 8))
classifier.add(Dense(output_dim = 8,  init = 'uniform',  activation = 'relu'))
classifier.add(Dense(output_dim = 1,  init = 'uniform', activation = 'sigmoid'  ))
classifier.compile(optimizer = 'adam',  loss = 'binary_crossentropy', metrics = ['accuracy'] )

classifier.fit(X_train, y_train, batch_size=10, epochs=1000)