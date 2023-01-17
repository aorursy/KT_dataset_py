import pandas as pd

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/emotions.csv')
df.info()
df.head()
import seaborn as sns

sns.countplot(df['label'],color='lightblue')
dt = df['label']
df = df.drop('label',axis=1)
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(df,dt,test_size=0.3,random_state=42)
params = {

    'objective': 'multi:softprob',

    'max_depth': 5,

    'learning_rate': 1.0,

    'n_estimators': 15

}
%%time

model = XGBClassifier(**params).fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn import metrics
count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
from sklearn import ensemble
%%time

model = ensemble.RandomForestClassifier(n_estimators=15,max_depth=4)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,dt,test_size=0.3,random_state=42)
from sklearn import linear_model
%%time

model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))