import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
mushrooms = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

mushrooms.describe()

#To check the top few rows

mushrooms.head(5)
#Check the dimensions and shape

mushrooms.ndim

mushrooms.shape
#Plotting different variables to see the distribution

import seaborn as sns

plot1 = sns.countplot(x= 'odor', data = mushrooms)
plot2 = sns.countplot(x= 'class', data = mushrooms)
plot3 = sns.countplot(x= 'cap-surface', data = mushrooms)
plot3 = sns.countplot(x= 'cap-color', data = mushrooms)
#Variablenames Extraction



variable_labels = np.asarray(mushrooms.columns)[0:]
X_var = variable_labels[1:22]
X_var
#Replace Categoricaldata with dummy variable

#Used Label Encoder

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

encoder.fit(mushrooms['cap-shape'].drop_duplicates())

mushrooms['cap-shape']=encoder.transform(mushrooms['cap-shape'])

encoder.fit(mushrooms['cap-surface'].drop_duplicates())

mushrooms['cap-surface']=encoder.transform(mushrooms['cap-surface'])

encoder.fit(mushrooms['cap-color'].drop_duplicates())

mushrooms['cap-color']=encoder.transform(mushrooms['cap-color'])

encoder.fit(mushrooms['bruises'].drop_duplicates())

mushrooms['bruises']=encoder.transform(mushrooms['bruises'])

encoder.fit(mushrooms['odor'].drop_duplicates())

mushrooms['odor']=encoder.transform(mushrooms['odor'])

encoder.fit(mushrooms['gill-attachment'].drop_duplicates())

mushrooms['gill-attachment']=encoder.transform(mushrooms['gill-attachment'])

encoder.fit(mushrooms['gill-spacing'].drop_duplicates())

mushrooms['gill-spacing']=encoder.transform(mushrooms['gill-spacing'])

encoder.fit(mushrooms['gill-size'].drop_duplicates())

mushrooms['gill-size']=encoder.transform(mushrooms['gill-size'])

encoder.fit(mushrooms['gill-color'].drop_duplicates())

mushrooms['gill-color']=encoder.transform(mushrooms['gill-color'])

encoder.fit(mushrooms['stalk-shape'].drop_duplicates())

mushrooms['stalk-shape']=encoder.transform(mushrooms['stalk-shape'])

encoder.fit(mushrooms['stalk-root'].drop_duplicates())

mushrooms['stalk-root']=encoder.transform(mushrooms['stalk-root'])

encoder.fit(mushrooms['stalk-surface-above-ring'].drop_duplicates())

mushrooms['stalk-surface-above-ring']=encoder.transform(mushrooms['stalk-surface-above-ring'])

encoder.fit(mushrooms['stalk-surface-below-ring'].drop_duplicates())

mushrooms['stalk-surface-below-ring']=encoder.transform(mushrooms['stalk-surface-below-ring'])

encoder.fit(mushrooms['stalk-surface-below-ring'].drop_duplicates())

mushrooms['stalk-surface-below-ring']=encoder.transform(mushrooms['stalk-surface-below-ring'])

encoder.fit(mushrooms['stalk-color-above-ring'].drop_duplicates())

mushrooms['stalk-color-above-ring']=encoder.transform(mushrooms['stalk-color-above-ring'])

encoder.fit(mushrooms['stalk-color-below-ring'].drop_duplicates())

mushrooms['stalk-color-below-ring']=encoder.transform(mushrooms['stalk-color-below-ring'])

encoder.fit(mushrooms['veil-type'].drop_duplicates())

mushrooms['veil-type']=encoder.transform(mushrooms['veil-type'])

encoder.fit(mushrooms['veil-color'].drop_duplicates())

mushrooms['veil-color']=encoder.transform(mushrooms['veil-color'])

encoder.fit(mushrooms['ring-number'].drop_duplicates())

mushrooms['ring-number']=encoder.transform(mushrooms['ring-number'])

encoder.fit(mushrooms['ring-type'].drop_duplicates())

mushrooms['ring-type']=encoder.transform(mushrooms['ring-type'])

encoder.fit(mushrooms['spore-print-color'].drop_duplicates())

mushrooms['spore-print-color']=encoder.transform(mushrooms['spore-print-color'])

encoder.fit(mushrooms['population'].drop_duplicates())

mushrooms['population']=encoder.transform(mushrooms['population'])



#Splitting the data into training and testing

from sklearn.model_selection import train_test_split



#use iloc for selecting the class as y variable as this is about mushroom classification and all others as X variable

X = mushrooms.iloc[:,1:22]

y = mushrooms.iloc[:,0]
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 0)
#Method1 - RandomForestClassification

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

error_rf = metrics.accuracy_score(y_test,y_pred_rf)

print(np.sqrt(error_rf))
#Feature Importance Plot

importance = rf.feature_importances_
#Visualize Feature Importance

indices = np.argsort(rf.feature_importances_)[::-1]

names = [X_var[i] for i in indices]
import matplotlib.pyplot as plt

plt.figure()

plt.title('Feature Importance')

plt.bar(range(X.shape[1]), importance[indices])

plt.xticks(range(X.shape[1]), names, rotation=90)

plt.show()

#Method 2 - K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

#From the confusion matrix and the classification report it can be seen that the class e 849 have been predicted correctly and

#3 are predicted wrong. Similarly, for class p all 773 are predicted correctly.
#initiate the error values as a empty matrix. With the loop running from 1 to 20 the error value will get appendend accordingly

error = []



# Calculating error for K values between 1 and 20

for i in range(1, 20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))
error
#From the above we can see mean error is lowest when k=1,2 and k=5,6 . Since k=1 will not classify it we can go for the next best 

#option that produces lowest error which is k=2 which is ideal in this case as 'Class' has 2 levels e and p



#Accuracy

y_pred_knn = classifier.predict(X_test)

error_knn = metrics.accuracy_score(y_test,y_pred_knn)

print(np.sqrt(error_knn))
#From the above 2 methods of knn and random forest we can see that random forest has slightly better accuracy than knn
