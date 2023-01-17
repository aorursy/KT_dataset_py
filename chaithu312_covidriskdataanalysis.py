# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import LinearSVC
# Importing the dataset
dataset = pd.read_csv('../input/riskdataset/Book2.csv',header=[0],  squeeze=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


index = dataset.index
columns = dataset.columns
values = dataset.values
index
columns
values

ds = dataset[1:474][["Gender","Age Bracket","HyperTension","Diabetes","Notes", "Patient Status"]]

df = pd.DataFrame({'Gender' :ds["Gender"].fillna('M'),"Patient Status":ds["Patient Status"].fillna('Deceased'), "Age Bracket": ds["Age Bracket"].fillna(48),"HyperTension" : ds["HyperTension"].fillna('Yes'),"Diabetes" : ds["Diabetes"].fillna('Mumbai')})

dff = df.replace({'Yes/No':'Yes'})
dff = dff.replace({'Deceased': '0'})
dff = dff.replace({'Recovered':'1'})
dff
dataset = dff
X = dataset.iloc[:, [2]].values
X
y = dataset.iloc[:, 1].values
y
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train
X_test = sc.transform(X_test)
X_test
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print (y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
 # Visualising the Training set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train


y_set
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, -1].min() - 0, stop = X_set[:, -1].max() + 1, step = 0.01))
print (X1.ravel(), X2.ravel() ,X1.shape)
ListedColormap(('red', 'green'))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel()]).T).reshape(X1.shape),
             alpha = 0.1, cmap =  ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, -1], X_set[y_set == j, 0],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Covid 9 - Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Patient Status')
plt.legend()
plt.show()



x12 , x34 = X1.shape
y12 , y34 = X1.shape
plt.xlim(900, 1100)
plt.ylim(900, 1100)
plt.scatter([x12 , x34],[y12,y34])
plt.plot([x12 , x34],[y12,y34])
plt.title('Shape')
plt.xlabel('Age')
plt.ylabel('Patient Status')
plt.legend()
plt.show()