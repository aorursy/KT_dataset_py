import pandas as pd
address = '../input/creditcardfraud/creditcard.csv'
df = pd.read_csv(address)
df
df['Class'].value_counts()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = df.drop('Class', axis=1)
y = df.Class
X = scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)
model_svc.score(X_train,y_train)
model_svc.score(X_test,y_test)
y_predict = model_svc.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix
import numpy as np
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
confusion
import seaborn as sns
sns.heatmap(confusion, annot=True)
print(classification_report(y_test, y_predict))