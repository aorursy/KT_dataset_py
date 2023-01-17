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

from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
def scale_feat(df,cont_feat) :
    df1=df
    scaler = preprocessing.RobustScaler()
    df1[cont_feat] = scaler.fit_transform(df1[cont_feat])
    scaler = preprocessing.StandardScaler()
    df1[cont_feat] = scaler.fit_transform(df1[cont_feat]) 
    return df1
from sklearn.model_selection import learning_curve
def plot_learning_curve(est, X_train, y_train) :
    train_sizes, train_scores, test_scores = learning_curve(estimator=est, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 100),
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
def plot_roc_curve(est,X_test,y_test) :
    probas = est.predict_proba(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8,8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe
    plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe
    plt.xlim([-0.05,1.2])
    plt.ylim([-0.05,1.2])
    plt.ylabel('Taux de vrais positifs')
    plt.xlabel('Taux de faux positifs')
    plt.show
def undersample(df, target_col, minority_class) :
    df_minority = df[df[target_col] == minority_class]
    df_majority = df.drop(df_minority.index)
    ratio=len(df_minority)/len(df_majority)
    df_majority = df_majority.sample(frac=ratio)
    df1 = pd.concat((df_majority,df_minority), axis=0)
    return df1.sample(frac=1)
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
df.info()
df.Class.value_counts()
df.columns
discr_feat = []
cont_feat = list(set(df.columns) - set(discr_feat)-{'Class'})
df.isnull().values.sum()
df[cont_feat].describe()
df=scale_feat(df,cont_feat)
for col in cont_feat :
    plt.figure(figsize=[10,5])
    sns.kdeplot(df[col])
df.describe()

from sklearn.model_selection import train_test_split

X = df.drop(['Class'], axis=1)
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)

print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)
print(cm)
# plot_learning_curve(rf, X_train, y_train)
# execution assez lente sur un grand dataset
plot_roc_curve(rf,X_test,y_test)
df.Class.value_counts()
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_sample(X_train, y_train)
rf = ensemble.RandomForestClassifier()
rf.fit(X_rus, y_rus)
y_rf = rf.predict(X_test)

print(classification_report(y_test, y_rf))

print(confusion_matrix(y_test, y_rf))

plot_learning_curve(rf, X_rus, y_rus)

plot_roc_curve(rf,X_test,y_test)
import xgboost as XGB
xgb  = XGB.XGBClassifier()
xgb.fit(X_rus, y_rus)
y_xgb = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_xgb)
print(cm)
print(classification_report(y_test, y_xgb))
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_sm, y_sm = smote.fit_sample(X_train, y_train)

rf = ensemble.RandomForestClassifier()
rf.fit(X_sm, y_sm)
y_rf = rf.predict(X_test)

print(classification_report(y_test, y_rf))

print(confusion_matrix(y_test, y_rf))

# plot_learning_curve(rf, X_sm, y_sm)

plot_roc_curve(rf,X_test,y_test)

importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X.columns[indices])
plt.title('Importance des caracteristiques')

xgb  = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_xgb)
print(cm)
print(classification_report(y_test, y_xgb))
# plot_learning_curve(xgb, X_train, y_train)

plot_roc_curve(xgb, X_test,y_test)
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True)
X_tl, y_tl = tl.fit_sample(X_train, y_train)
rf = ensemble.RandomForestClassifier()
rf.fit(X_tl, y_tl)
y_rf = rf.predict(X_test)

print(classification_report(y_test, y_rf))

print(confusion_matrix(y_test, y_rf))

plot_learning_curve(rf, X_tl, y_tl)

plot_roc_curve(rf,X_test,y_test)


