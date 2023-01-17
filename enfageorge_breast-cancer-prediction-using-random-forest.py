import pandas as pd
data= pd.read_csv('../input/data.csv')
data
# Missing values

data.isnull().sum()
data.isna().sum() # No missing values
data.head()
data.shape
X = data.iloc[:,2:32].values

Y = data.iloc[:,1].values
Y
#Encoding categorical data values

from sklearn.preprocessing import LabelEncoder #encode categorical features using a one-hot or ordinal encoding scheme

labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y)
Y
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
cm
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_pred)
accuracy