# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dataset = pd.read_csv("../input/HR_comma_sep.csv")

print("Dataset Loaded successfully")

# Any results you write to the current directory are saved as output.
# Number of columns in the dataset

dataset.columns
#Shape of the dataset

dataset.shape
#Exploratory data analysis



dataset.describe()
dataset.head()
dataset['left'].value_counts()

# So, the number of people left are 3571
import matplotlib.pyplot as plt

plt.hist(dataset['left'])

plt.show()
import seaborn as sns

sns.countplot(x="sales", data=dataset)

plt.show()

#So from the below picture we can analyze that most of the employees is dataset are in Sales department
sns.boxplot(x = "sales", y = "average_montly_hours" , hue = 'left',data = dataset)

plt.show

#Unable to interpret anything from this
sns.boxplot(x = "sales", y = "satisfaction_level" , hue = 'left',data = dataset)

plt.show

# So from the below plot it is very evident that people with less satisfaction level are leaving the Company
sns.boxplot(x = "sales", y = "time_spend_company" , hue = 'left',data = dataset)

plt.show

# Seems to have some outliers in time_spend_comany 
data = dataset.copy()
# Two of the attributes are Categorical so performing Label encoding along with One hot encoding 

from sklearn.preprocessing import StandardScaler, LabelEncoder

le=LabelEncoder()

for i in data:

    if data[i].dtype == 'object':

        le = LabelEncoder()

        le.fit(data[i].astype(str))

        temp = le.transform(data[i].astype(str))

        if temp.std() == 0:

            print("Dropped attributes with stddev == 0:" , i)

        else:

            one_hot = pd.get_dummies(data[i].astype(str))

            one_hot.columns=[(i+"_"+str(n)) for n in le.classes_]

            one_hot.drop(i +"_"+str(le.classes_[0]),inplace=True,axis=1)

            data = pd.concat([data,one_hot], axis = 1)

            print("Dropped attributes original data after One Hot encoding:", i)

        data.drop([i], inplace = True, axis = 1)
data.head()
print("Shape of dataset before Onehot encoding", dataset.shape)

print("Shape of dataset after Onehot encoding", data.shape)
# So after one hot encoding i have converted the Categorical attributes into Numerical

data.columns


# So taking the target as LEFT i can predict if a employee will leave or stay based on the other features

target = ['left']

X = data.drop(target, axis = 1)

y = pd.DataFrame(data[target])
# Splitting dataset into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.head()
# Scaling helps in converging fast

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data

scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier
# Selecting 4 layers with 100 neurons, 50 neurons, 10 neurons, 5 neurons

mlp = MLPClassifier(activation='relu', alpha=0.001, batch_size='auto',max_iter = 200, solver = 'adam',hidden_layer_sizes=(100,50,10,5))
mlp.fit(X_train, y_train)
# Performing predicitons

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))