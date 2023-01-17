import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import cufflinks as cf

from plotly.offline import iplot

cf.go_offline('True')





%matplotlib inline
dataset = pd.read_csv('../input/creditcardfraud/creditcard.csv')



dataset.info()
pd.set_option('display.max_columns',31)

dataset.head(10)
dataset.describe()
X = dataset.drop('Class', axis=1)

y = dataset['Class']
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



scaler = StandardScaler()

pca = PCA()



X_scaled = scaler.fit_transform(X)



pca.fit(X_scaled)

display(pca.explained_variance_ratio_.cumsum())



n_features = range(pca.n_components_)

plt.bar(n_features, pca.explained_variance_)

#plt.xticks(features)
var = pca.explained_variance_ratio_

plt.plot(var)
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(32, 24))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(dataset)
sns.countplot(data=dataset, x='Class')
from imblearn.over_sampling import SMOTE



X_bal, y_bal = SMOTE().fit_sample(X_scaled, y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=123, stratify=y_bal)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score



print("ROC AUC score")

print(roc_auc_score(y_test, y_pred))



print("\nTraining Accuracy")

print(lr.score(X_train, y_train))



print("\nTesting Accuracy")

print(lr.score(X_test, y_test))



print(confusion_matrix(y_test, y_pred))



print(classification_report(y_test, y_pred))
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(y_test, y_pred)

class_names = ['GENUINE','FRAUD']

np.set_printoptions(precision=2)





plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 

                      title='Normalized confusion matrix')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()



rf.fit(X_train, y_train)



y_pred_rf = rf.predict(X_test)
print("ROC_AUC score")

print(roc_auc_score(y_test, y_pred_rf))



print("\nTraining Score")

print(rf.score(X_train, y_train))



print("\nTesting Score")

print(rf.score(X_test, y_test))



print(confusion_matrix(y_test, y_pred_rf))



print(classification_report(y_test, y_pred_rf))
cnf_matrix = confusion_matrix(y_test, y_pred_rf)

class_names = ['GENUINE','FRAUD']

np.set_printoptions(precision=2)





plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 

                      title='Normalized confusion matrix')