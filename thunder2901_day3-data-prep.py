import pandas as pd

df = pd.read_csv("../input/apndcts/apndcts.csv")

df.shape
from sklearn.model_selection import train_test_split



X = df.drop(['class'],axis='columns')

y = df['class']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)



print(X_train.shape)

print(X_test.shape)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(df):

    df_train = df.iloc[train_index]

    df_test = df.iloc[test_index]

    print("Training data: ",df_train.shape)

    print("Testing data: ",df_test.shape)
from sklearn.utils import resample



X = df.iloc[:,0:9]

resample(X, n_samples=100, random_state=0)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier



dct = DecisionTreeClassifier()

dct.fit(X_train,y_train)

dct.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

predict_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test,predict_knn)

sns.heatmap(cm_knn,annot=True)

plt.ylabel("Predicted")

plt.xlabel("Truth")