import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
sns.set()
%matplotlib inline

df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df.head()
plt.hist(df['Target'].astype(str))
plt.title('Target histogram')
plt.ylabel('Count')
plt.xlabel('Class')
plt.show()
for feature in df.columns:
    print (feature)
train_null = df.isnull().sum()
train_null_non_zero = train_null[train_null>0] / df.shape[0]
train_null_non_zero
df = df.fillna(df.mean())
test = test.fillna(test.mean())
y = df['Target']
X = df.drop(['Target', 'Id'], axis=1)
test_id = test['Id']
test.drop('Id', axis=1, inplace=True)
train_test_df = pd.concat([X, test], axis=0)
cols = [col for col in train_test_df.columns if train_test_df[col].dtype == 'object']

le = LabelEncoder()
for col in cols:
    le.fit(train_test_df[col])
    X[col] = le.transform(X[col])
    test[col] = le.transform(test[col])
from xgboost import XGBClassifier
from xgboost import plot_importance
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
plt.rcParams["figure.figsize"] = (15,20)
plot_importance(model)
plt.show()
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1)
from sklearn.svm import SVC

start = time.time()

rbf_svm = SVC(kernel='rbf', gamma=0.8, C=12)
rbf_svm.fit(X_train, Y_train)


print("Time: " , time.time() - start )
print("Accuracy: ",rbf_svm.score(X_test, Y_test))
y_predict = model.predict(test)
pred = pd.DataFrame({"Id": test_id, "Target": y_predict})
pred.to_csv('submission.csv', index=False)
pred.head()
