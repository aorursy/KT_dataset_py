import pandas as pd

df=pd.read_csv("../input/bank_note_data.csv")
df.head()
import seaborn as sns
import matplotlib as plt
%matplotlib inline
sns.countplot(data=df,x="Class")
sns.pairplot(data=df,hue="Class")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop("Class",axis=1))
scaled_fea = scaler.fit_transform(df.drop("Class",axis=1))
df_1=pd.DataFrame(scaled_fea,columns=df.columns[:-1])
df_1.head()
X=df_1
y=df["Class"]
X=X.as_matrix()
y=y.as_matrix()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
import tensorflow as tf
import tensorflow.contrib.learn as learn
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=2)
classifier.fit(X_train, y_train, steps=200, batch_size=20)
note_predictions = list(classifier.predict(X_test))
from sklearn.metrics import classification_report,confusion_matrix
type(y_test)
print(confusion_matrix(y_test,note_predictions))
print(classification_report(y_test,note_predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))