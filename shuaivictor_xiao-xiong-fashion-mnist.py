# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression



from sklearn import metrics

from sklearn import preprocessing

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score



from sklearn.preprocessing import StandardScaler, MinMaxScaler



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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
df.head()
df.shape
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",

          "Sneaker","Bag","Ankle boot"]

n_samples = len(df.index)

images = np.array(df.drop(['label'],axis=1))

images = images.reshape(n_samples,28,28)
print(images[0])
plt.imshow(images[2])
plt.imshow(images[0], cmap="gray_r")

plt.axis('off')

plt.title('Label: %s' % labels[df.label[0]])
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(images[i], cmap="gray_r")

    plt.title('Label: %s' % labels[df.label[i]])
images[0].reshape(784)
df.label.value_counts()
df.columns
df_train=df.sample(frac=0.08)
df_train.count()
df_train.describe()
X = df_train.drop(['label'], axis=1)

y = df_train.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
lr = LogisticRegression()

lr.fit(X_train,y_train)

y_lr = lr.predict(X_test)
lr_score = metrics.accuracy_score(y_test, y_lr)

print(lr_score)
print(metrics.classification_report(y_test, y_lr))
cm = metrics.confusion_matrix(y_test, y_lr)

print(cm)
plot_learning_curve(lr,X_test,y_test)
from imblearn.under_sampling import TomekLinks



tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, i_tl = tl.fit_sample(X_train, y_train)
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X_train, y_train)
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X_train, y_train)
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
print(accuracy_score(y_test,y_rf))
print(confusion_matrix(y_test,y_rf))
plot_learning_curve(rf, X_train, y_train)
import xgboost as xgb
xgbc=xgb.XGBClassifier()
xgbc.fit(X_train, y_train)
y_xgbc=xgbc.predict(X_test)
print(accuracy_score(y_test,y_xgbc))
pd.crosstab(y_test, y_xgbc, rownames=['Reel'], colnames=['Prediction'], margins=True)