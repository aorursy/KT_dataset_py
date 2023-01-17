# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msgn

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

# Lecture des données
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head(15)
df.Outcome.value_counts() #Nombre de positif et négatif par diabètes
df.columns

sns.pairplot(df, hue = "Outcome")
for col in df.columns:
    plt.figure(figsize = [10,5])
    sns.distplot(df[col])
# Changement du 0 des résultats du test en NaN
df_ = df.replace(0,np.nan)
df_.Pregnancies = df.Pregnancies
df_.Outcome = df.Outcome
df_.head(15)
# Comptage par colonnes
df_.count()      
msgn.bar(df_)
df_.info()
values={'Glucose':np.random.normal(df_.Glucose.mean(),df_.Glucose.std()), 
        'BloodPressure':np.random.normal(df_.BloodPressure.mean(),df_.BloodPressure.std()), 
        'SkinThickness':np.random.normal(df_.SkinThickness.mean(),df_.SkinThickness.std()), 
        'BMI':np.random.normal(df_.BMI.mean(),df_.BMI.std())}
values
df_ = df_.fillna(value = values)
df_Insulin_nan = df_[np.isnan(df_.Insulin)]
df_Insulin = df_.drop(df_Insulin_nan.index)
X = df_Insulin.drop(['Insulin'], axis=1)
y = df_Insulin.Insulin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)
from sklearn import ensemble
rf = ensemble.RandomForestRegressor()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print(rf.score(X_test,y_test))
plt.figure(figsize=(12,12))
plt.scatter(y_test, y_rf)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], color='red', linewidth=3)
X_nan = df_Insulin_nan.drop(['Insulin'], axis=1)
y_nan = rf.predict(X_nan)
df_Insulin_nan['Insulin'] = y_nan
df_ = pd.concat([df_Insulin, df_Insulin_nan], ignore_index = True, sort = False)
df_.head()
df_.describe()
scaler = preprocessing.StandardScaler()
df_[['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']] = scaler.fit_transform(df_[['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']])
df_.describe()
scaler = preprocessing.MinMaxScaler()
df_[['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']] = scaler.fit_transform(df_[['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']])
df_.describe()
X = df_.drop(['Outcome'], axis=1)
y = df_.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
for col in df_.columns :
    plt.figure(figsize = [10,5])
    sns.distplot(df_[col])
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
plt.figure(figsize = (12,12))
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
from imblearn.under_sampling import TomekLinks

tl = TomekLinks('auto',True,'majority')
X_tl, y_tl = tl.fit_sample(X_train, y_train)
lr = LogisticRegression()
lr.fit(X_tl,y_tl)
y_lr = lr.predict(X_test)
probas = lr.predict_proba(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
print(metrics.classification_report(y_test,y_lr))
from imblearn.over_sampling import SMOTE

smote = SMOTE('minority')
X_sm, y_sm = smote.fit_sample(X_train, y_train)
lr = LogisticRegression()
lr.fit(X_sm,y_sm)
y_lr = lr.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_lr)
print(cm)
probas = lr.predict_proba(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
print(metrics.classification_report(y_test,y_lr))
from sklearn.model_selection import learning_curve
def plot_learning_curve(est, X_train, y_train) :
    train_sizes, train_scores, test_scores = learning_curve(estimator=est, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=5,
                                                        n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8,10))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
    plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
    plt.grid(b='on')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.6, 1.0])
    plt.show()
plot_learning_curve(lr, X_train, y_train)
