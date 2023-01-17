import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from statistics import mode, mean

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
data = pd.read_csv("../input/heart-disease-prediction-using-logistic-regression/framingham.csv")
data.head(10)
plt.figure(figsize =(16,9))
sns.heatmap(data.corr() , annot = True)
data.isnull().sum()
data.describe()
data = data.dropna(axis='rows', thresh=15)
data.isnull().sum()
data["education"]=data["education"].fillna(mode(data["education"]))
data["BPMeds"]=data["BPMeds"].fillna(mode(data["BPMeds"]))

data["cigsPerDay"]=data["cigsPerDay"].fillna((data["cigsPerDay"].mean()))
data["totChol"]=data["totChol"].fillna((data["totChol"].mean()))
data["BMI"]=data["BMI"].fillna((data["BMI"].mean()))
data["heartRate"]=data["heartRate"].fillna((data["heartRate"].mean()))
data["glucose"]=data["glucose"].fillna(data["glucose"].mean())

data.isnull().sum()
plt.figure(figsize = (16,9))
sns.pairplot(data)
data = data.drop(columns='currentSmoker')
x = data[['male','age','education','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]
y = pd.Series(data['TenYearCHD'])
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=1)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
model = LogisticRegression()
model.fit(train_x , train_y)
pred = model.predict(test_x)
print(accuracy_score(test_y , pred))
print(confusion_matrix(test_y , pred))
sns.heatmap(confusion_matrix(test_y , pred) , annot = True)
print(classification_report(test_y , pred))
logit_roc_auc = roc_auc_score(test_y, pred)
fpr, tpr, thresholds = roc_curve(test_y, model.predict_proba(test_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()