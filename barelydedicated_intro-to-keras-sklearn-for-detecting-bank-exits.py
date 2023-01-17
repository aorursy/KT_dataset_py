!pip install scikit-learn==0.22.0
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))
#importing the dataset

dataset = pd.read_csv('../input/Churn_Modelling.csv', index_col='RowNumber')

dataset.head()
X_columns = dataset.columns.tolist()[2:12]

y_columns = dataset.columns.tolist()[-1:]

print(f'All columns: {dataset.columns.tolist()}')

print()

print(f'X values: {X_columns}')

print()

print(f'y values: {y_columns}')
X = dataset[X_columns].values # Credit Score through Estimated Salary

y = dataset[y_columns].values # Exited
# Encoding categorical (string based) data. Country: there are 3 options: France, Spain and Germany

# This will convert those strings into scalar values for analysis

print(X[:8,1], '... will now become: ')

from sklearn.preprocessing import LabelEncoder

label_X_country_encoder = LabelEncoder()

X[:,1] = label_X_country_encoder.fit_transform(X[:,1])

print(X[:8,1])
# We will do the same thing for gender. this will be binary in this dataset

print(X[:6,2], '... will now become: ')

from sklearn.preprocessing import LabelEncoder

label_X_gender_encoder = LabelEncoder()

X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])

print(X[:6,2])
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline





pipeline = Pipeline(

    [('Categorizer', ColumnTransformer(

         [ # Gender

          ("Gender Label encoder", OneHotEncoder(categories='auto', drop='first'), [2]),

           # Geography

          ("Geography One Hot", OneHotEncoder(categories='auto', drop='first'), [1])

         ], remainder='passthrough', n_jobs=1)),

     # Standard Scaler for the classifier

    ('Normalizer', StandardScaler())

    ])
X = pipeline.fit_transform(X)
# Splitting the dataset into the Training and Testing set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
print(f'training shapes: {X_train.shape}, {y_train.shape}')

print(f'testing shapes: {X_test.shape}, {y_test.shape}')
from keras.models import Sequential

from keras.layers import Dense, Dropout
# Initializing the ANN

classifier = Sequential()
# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)

classifier.add(Dense(6, activation = 'relu', input_shape = (X_train.shape[1], )))

classifier.add(Dropout(rate=0.1)) 
# Adding the second hidden layer

# Notice that we do not need to specify input dim. 

classifier.add(Dense(6, activation = 'relu')) 

classifier.add(Dropout(rate=0.1)) 
# Adding the output layer

# Notice that we do not need to specify input dim. 

# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)

# We use the sigmoid because we want probability outcomes

classifier.add(Dense(1, activation = 'sigmoid')) 
classifier.summary()
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = classifier.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.1, verbose=2)
plt.plot(np.array(history.history['acc']) * 100)

plt.plot(np.array(history.history['val_acc']) * 100)

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['train', 'validation'])

plt.title('Accuracy over epochs')

plt.show()
y_pred = classifier.predict(X_test)

print(y_pred[:5])
y_pred = (y_pred > 0.5).astype(int)

print(y_pred[:5])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print (((cm[0][0]+cm[1][1])*100)/(len(y_test)), '% of testing data was classified correctly')