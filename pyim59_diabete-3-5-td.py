
# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head(20)
df.Outcome.value_counts()
df.columns
cont_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
discrete_features = ['Pregnancies']
sns.pairplot(df, hue="Outcome")
for col in cont_features :
    plt.figure(figsize=[10,5])
    sns.distplot(df[col])
df1 = df.replace(0,np.nan)
df1.Pregnancies = df.Pregnancies
df1.Outcome = df.Outcome
!pip install missingno
import missingno as msno
msno.bar(df1)
msno.matrix(df1)

df1.count()
values={'Glucose':df1.Glucose.mean(), 
        'BloodPressure':df1.BloodPressure.mean(), 
        'SkinThickness':df1.SkinThickness.mean(), 
        'BMI':df1.BMI.mean(),
        'Insulin':df1.Insulin.mean()
       }
for col in cont_features :
    df1=df1.fillna(value={col : np.random.normal(df1[col].mean(),df1[col].std())})
df1.info()
scaler = preprocessing.StandardScaler()
df1[cont_features] = scaler.fit_transform(df1[cont_features])
df1.describe()
X = df1.drop(['Outcome'], axis=1)
y = df1.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
for col in cont_features :
    plt.figure(figsize=[10,5])
    sns.distplot(df1[col])

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
lr_score = metrics.accuracy_score(y_test, y_lr)
print(lr_score)
print(metrics.classification_report(y_test, y_lr))
cm = metrics.confusion_matrix(y_test, y_lr)
print(cm)
probas = lr.predict_proba(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
plt.figure(figsize=(12,12))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe
plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_sample(X_train, y_train)
print(X_rus.shape)
print(y_rus.shape)
lr = LogisticRegression()
lr.fit(X_rus,y_rus)
y_lr = lr.predict(X_test)
print(metrics.classification_report(y_test, y_lr))
cm = metrics.confusion_matrix(y_test, y_lr)
print(cm)
probas = lr.predict_proba(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
plt.figure(figsize=(12,12))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe
plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print(metrics.classification_report(y_test,y_lr))
