# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing required Libraries

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

dataset = pd.read_csv("../input/titanic/train.csv",header=0)

dataset.head()
dataset = dataset.drop(columns=['Name', 'PassengerId','Ticket'], axis=0)

dataset.head()
ax = sns.countplot(x="Pclass", hue = "Survived", data=dataset)
ax1 = sns.countplot(x="Survived", hue = "Sex", data=dataset)
ageclass = ["0-20","21-40","41-60","61-80","81-100","100+"]

#We make groups of age, in the range of 20 years



agesurvival = pd.DataFrame(index = ageclass, columns = ['0','1'])

agesurvival.loc[:,:] = 0

#Initialising the Dataframe with 0



for index,rows in dataset.iterrows():

    if rows["Age"] <= 20:

        if rows["Survived"] == 0:

            agesurvival.loc["0-20"]['0'] += 1

        else:

            agesurvival.loc["0-20"]['1'] += 1

    elif rows["Age"] > 20 and rows["Age"] <= 40:

        if rows["Survived"] == 0:

            agesurvival.loc["21-40"]['0'] += 1

        else:

            agesurvival.loc["21-40"]['1'] += 1

    elif rows["Age"] > 40 and rows["Age"] <= 60:

        if rows["Survived"] == 0:

            agesurvival.loc["41-60"]['0'] += 1

        else:

            agesurvival.loc["41-60"]['1'] += 1

    elif rows["Age"] > 60 and rows["Age"] <= 80:

        if rows["Survived"] == 0:

            agesurvival.loc["61-80"]['0'] += 1

        else:

            agesurvival.loc["61-80"]['1'] += 1

    elif rows["Age"] > 80 and rows["Age"] <= 100:

        if rows["Survived"] == 0:

            agesurvival.loc["81-100"]['0'] += 1

        else:

            agesurvival.loc["81-100"]['1'] += 1

    elif rows["Age"] > 100:

        if rows["Survived"] == 0:

            agesurvival.loc["100+"]['0'] += 1

        else:

            agesurvival.loc["100+"]['1'] += 1

        

agesurvival.plot.bar(rot=0)



#We get the following graph:
#Separating dependent and independent variables

x = dataset.iloc[:,1:]

y = dataset.iloc[:,0]
x.isna().sum()
del dataset["Cabin"]

#dataset = dataset.drop(columns=['Name', 'PassengerId','Ticket'], axis=0)

dataset.head()
#Separating dependent and independent variables. [Repeating without the 'Cabin' column]

x = dataset.iloc[:,1:]

y = dataset.iloc[:,0]

x
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x.iloc[:, 2:3])

x.iloc[:, 2:3] = imputer.transform(x.iloc[:, 2:3])

x
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer = imputer.fit(x.iloc[:, 6:7])

x.iloc[:, 6:7] = imputer.transform(x.iloc[:, 6:7])

x
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()



x.iloc[:, 0] = labelencoder_X.fit_transform(x.iloc[:,0])

x.iloc[:, 1] = labelencoder_X.fit_transform(x.iloc[:,1])

x.iloc[:, 6] = labelencoder_X.fit_transform(x.iloc[:,6])



columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,6])], remainder='passthrough')

x = columnTransformer.fit_transform(x)

x
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
import keras

#sequential - initialize NN

from keras.models import Sequential



#dense - build NN

from keras.layers import Dense,Dropout



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

#init - 

#activation - 

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))

classifier.add(Dropout(0.3))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.3))

# Adding the third hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.3))

# Adding the fourth hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.3))

# Adding the output layer

#sigmoid - will give probabilities of classes

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6), loss = 'mean_squared_error', metrics = ['accuracy'])





# Fitting the ANN to the Training set

history = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500,validation_split=0.25)

#Preparing the test set

datasetest = pd.read_csv("../input/titanic/test.csv")

datasetest = datasetest.drop(columns=['Name', 'PassengerId','Ticket','Cabin'])



#Separating dependent and independent variables

x_test = datasetest.iloc[:,:]

#y_test = no independent variable in test set



#Taking care of missing values

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x_test.iloc[:, 2:3])

x_test.iloc[:, 2:3] = imputer.transform(x_test.iloc[:, 2:3])



imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer = imputer.fit(x_test.iloc[:, 6:7])

x_test.iloc[:, 6:7] = imputer.transform(x_test.iloc[:, 6:7])





#Taking care of categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()



x_test.iloc[:, 0] = labelencoder_X.fit_transform(x_test.iloc[:,0])

x_test.iloc[:, 1] = labelencoder_X.fit_transform(x_test.iloc[:,1])

x_test.iloc[:, 6] = labelencoder_X.fit_transform(x_test.iloc[:,6])



columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,6])], remainder='passthrough')

x_test = columnTransformer.fit_transform(x_test)





# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_test = sc.fit_transform(x_test)

y_pred = classifier.predict(x_test)

y_pred1 = []

for i in range(len(y_pred)):

    if y_pred[i] > 0.5:

        y_pred1.append(1)

    else:

        y_pred1.append(0)
result = pd.DataFrame(columns=['PassengerId','Survived'])

for i in range(len(y_pred1)):

    result = result.append([{'PassengerId':i+892, 'Survived':y_pred1[i]}], ignore_index = True)
result.head()