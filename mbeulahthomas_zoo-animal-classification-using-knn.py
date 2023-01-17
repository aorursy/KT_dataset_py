import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/zoo-animal-classification/zoo.csv')

data.head()
data.shape
data.info()
y=data['class_type'].values

X=data.drop(['class_type','animal_name'],axis=1).values
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)
knn.score(X_train,y_train)
knn.score(X_test,y_test)