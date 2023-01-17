import pandas as pd
df = pd.read_csv('../input/heart.csv')

df.head()
df.isna().sum()
from sklearn.preprocessing import scale
X = df.drop(['target'],axis =1)

#X = scale(X)

y = df.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,random_state = 10)
from xgboost import XGBClassifier
model = XGBClassifier(max_depth=7)

model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
model.feature_importances_*100
X.columns
model.classes_