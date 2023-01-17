import pandas as pd
raw = pd.read_csv("../input/creditcard.csv")
raw
raw['Class'].unique()
raw['Class'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x='Class', data=raw, palette='hls')
plt.show()
raw.isnull().sum()
raw.columns.values
features = raw.columns.values.tolist()[1:-1]
features
X = raw[features]
y = raw['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
y_train.value_counts()
X_test.shape
y_test.value_counts()
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf = RandomForestClassifier(max_depth=20, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
roc_auc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()