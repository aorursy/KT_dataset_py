import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/voicegender/voice.csv')
df.head()
df.describe()
df.info()
df.isna().sum()
sns.countplot(df['label'])
df.head().T
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df.head().T
x = df.drop('label',axis = 1)
y = df['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
model = RandomForestClassifier(n_estimators = 200,random_state = 42)
model.fit(x_train,y_train)
model.score(x_test,y_test)
y_preds = model.predict(x_test)
print(classification_report(y_test,y_preds))
con_mat = confusion_matrix(y_test,y_preds)
con_mat
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(con_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
svc_model = SVC()
svc_model.fit(x_train,y_train)
svc_model.score(x_test,y_test)
y_pred = svc_model.predict(x_test)
print(classification_report(y_test,y_pred))
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
