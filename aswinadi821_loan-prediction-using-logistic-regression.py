import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data =pd.read_csv("../input/train.csv")
data.head()
data.info()
data.apply(lambda x: sum(x.isnull()),axis=0) # checking missing values in each column of train dataset
data['Gender'].value_counts()
data.Gender = data.Gender.fillna('Male')
data['Married'].value_counts()
data.Married = data.Married.fillna('Yes')
data['Dependents'].value_counts()
data.Dependents = data.Dependents.fillna('0')
data['Self_Employed'].value_counts()
data.Self_Employed = data.Self_Employed.fillna('No')
data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data['Loan_Amount_Term'].value_counts()
data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0)
data['Credit_History'].value_counts()
data.Credit_History = data.Credit_History.fillna(1.0)
data.apply(lambda x: sum(x.isnull()),axis=0)
data.head()
# Splitting traing data

X = data.iloc[:, 1: 12].values

y = data.iloc[:, 12].values
X
y
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
X_train
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
for i in range(0, 5):

    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])



X_train[:,10] = labelencoder_X.fit_transform(X_train[:,10])
# Encoding the Dependent Variable

labelencoder_y = LabelEncoder()

y_train = labelencoder_y.fit_transform(y_train)
X_train
y_train
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

for i in range(0, 5):

    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])

X_test[:,10] = labelencoder_X.fit_transform(X_test[:,10])

# Encoding the Dependent Variable

labelencoder_y = LabelEncoder()

y_test = labelencoder_y.fit_transform(y_test)
X_test
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)

X_test = pca.fit_transform(X_test)

explained_variance = pca.explained_variance_ratio_
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)
y_pred
# Measuring Accuracy

from sklearn import metrics

print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred, y_test))
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
# Visualising the Training set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Logistic Regression (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()



# Visualising the Test set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Logistic Regression (Test set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()

