import pandas as pd
data = pd.read_csv('../input/classification-data-apples-oranges/apples_and_oranges.csv')

data.head()
import matplotlib.pyplot as plt

%matplotlib inline

plt.xlabel('Weight')

plt.ylabel('Size')

plt.scatter(data['Weight'], data['Size'],color="red",marker='*')
from sklearn.model_selection import train_test_split



X = data.drop(['Class'], axis='columns')

Y = data.drop(['Weight','Size'], axis='columns')



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

len(X_train)
len(X_test)
from sklearn.svm import SVC



model = SVC(kernel='rbf')



model.fit(X_train,Y_train)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',

  kernel='rbf', max_iter=-1, probability=False, random_state=None,

  shrinking=True, tol=0.001, verbose=False)
model.score(X_test, Y_test)
model.predict([[62,3.0]])
model.predict([[74, 5.3]])