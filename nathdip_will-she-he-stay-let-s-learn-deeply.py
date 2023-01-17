import numpy as np

import pandas as pd



# Importing the dataset

hrdata = pd.read_csv('../input/HR_comma_sep.csv')


col_names = ['satisfaction_level',

             'last_evaluation',

             'number_project',

             'average_montly_hours',

             'time_spend_company',

             'Work_accident',

             'salary',

             'promotion_last_5years',

             'sales',

             'left']

hrdata = hrdata.reindex(columns = col_names)

#Randomization

hrdata = hrdata.sample(frac = 1)

X = hrdata.iloc[:, 0:9].values

y = hrdata.iloc[:, 9].values
#Encode all categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X6 = LabelEncoder()

X[:, 6] = labelencoder_X6.fit_transform(X[:, 6])

labelencoder_X8 = LabelEncoder()

X[:, 8] = labelencoder_X8.fit_transform(X[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [6, 8])

X = onehotencoder.fit_transform(X).toarray()



#Spitting into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaling = StandardScaler()

X_train = scaling.fit_transform(X_train)

X_test = scaling.transform(X_test)
from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 20))

#    classifier.add(Dropout(0.1))

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

#    classifier.add(Dropout(0.1))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

cm = confusion_matrix(y_test, y_pred)

# Accuracy, Precision and Recall 

learning_score = precision_recall_fscore_support(y_test, y_pred)

accuracy_cm = (cm[0, 0] + cm[1, 1])/(sum(sum(cm)))

print('The accuracy is : ', accuracy_cm)

print('The precision is : ', learning_score[0][1])

print('The recall is : ', learning_score[1][1])

print('The fscore is : ', learning_score[2][1])