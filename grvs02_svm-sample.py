import pandas as pd

from sklearn.datasets import load_digits

digits = load_digits()
digits.target
dir(digits)
digits.target_names
df = pd.DataFrame(digits.data,digits.target)

df.head()
df['target'] = digits.target

df.head(20)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.2)
from sklearn.svm import SVC

rbf_model = SVC(kernel='rbf')
len(X_train)
len(X_test)
rbf_model.fit(X_train, y_train)
rbf_model.score(X_test,y_test)
'''Using Linear kernel'''
linear_model = SVC(kernel='linear')

linear_model.fit(X_train,y_train)
linear_model.score(X_test,y_test)