# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read input files
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
gender_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
header = list(train_data.head())
print(header)

train_data = train_data.dropna()

X = train_data[header[2:]]
y = train_data[header[1]]
del X['Name']
del X['Ticket']
del X['Cabin']


print(X)

# Check correlation between features
import seaborn as sns
plt.figure(figsize=(12,10))
cor = X.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

relevant_features = cor[cor>0.2]
relevant_features
selected_feat = ['Pclass', 'Age', 'SibSp', 'Fare']
# age = X['Age']
# age = age.values
# age = age.reshape((len(age), 1))

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

# age_transformed = scaler.fit_transform(age)
# X['Age'] = age_transformed
# print(X['Age'])
X = X[selected_feat]
X = X.values
y = y.values

X[X =='female'] = 1
X[X == 'male' ] = 0


X[X == 'C'] = 0
X[X == 'Q'] = 1
X[X == 'S'] = 2

X = X.astype(np.float32)

print(X.shape, y.shape)
# Apply PCA on features

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca.fit(X)
projected = pca.fit_transform(X)
print("variance ",np.cumsum(pca.explained_variance_ratio_))

plt.scatter(X[:, 2], X[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('plasma', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
f = open(os.getcwd()+"/decison_tree.pkl",'wb')
pickle.dump(clf, f)
f.close()
# Apply KNN
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)
acc = neigh.score(X_test, y_test)
print("Accuracy : ", acc)
f = open(os.getcwd()+"/knn.pkl",'wb')
pickle.dump(clf, f)
f.close()
# Test data

header = test_data.head()
passenger_id = test_data['PassengerId']

selected_feat = ['Pclass', 'Age', 'SibSp', 'Fare']
test_data = test_data[selected_feat]

print(list(header))
test_data = test_data.values
test_data[test_data =='female'] = 1
test_data[test_data == 'male' ] = 0


test_data[test_data == 'C'] = 0
test_data[test_data == 'Q'] = 1
test_data[test_data == 'S'] = 2

test_data = np.nan_to_num(test_data)

f = open(os.getcwd()+"/decison_tree.pkl",'rb')
DT = pickle.load(f)
f.close()

f = open(os.getcwd()+"/decison_tree.pkl",'rb')
knn = pickle.load(f)
f.close()


# predict
pred_knn = knn.predict(test_data)
pred_dt = DT.predict(test_data)
passenger_id1 = passenger_id.values
pred_knn = pred_knn.reshape((len(pred_knn), 1))
passenger_id1 = passenger_id1.reshape((len(passenger_id1), 1))

out = np.concatenate((passenger_id1, pred_knn), axis = 1)
print(pred_knn.shape, out.shape)
header_out = ["PassengerId", "Survived"]
out_df = pd.DataFrame(out, columns = header_out)
out_df.to_csv(os.getcwd()+"/pred.csv", index=False)
