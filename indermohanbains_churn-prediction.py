import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt
import seaborn as sns
#Click here and press Shift+Enter
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv'
churn_df = pd.read_csv(url)
churn_df.head()
print (churn_df.shape)
print ('\n')
print (churn_df.isnull ().sum ().sum ())
churn_df.info ()
# churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
corr = abs (churn_df.corr ())
plt.figure (figsize = (20,12))
sns.heatmap (corr, annot = True, cmap = 'rainbow', mask = corr < 0.1)
plt.yticks (rotation = 45)
# Highest correlation with target
corr ['churn'].nlargest (20).index [1:]
# multicolinear features
from itertools import product
corr_df = pd.DataFrame (columns = ['i', 'j', 'corr'], index = range (0,100)).fillna (0)

k = -1
for i,j in product (corr.columns, corr.columns):
    if (corr.loc [i,j] > 0.6) & (corr.loc [i,j] < 1):
        k = k + 1
        corr_df.loc [k, 'i'] = [i]
        corr_df.loc [k, 'j'] = [j]
        corr_df.loc [k, 'corr'] = corr.loc [i,j]
corr_df = corr_df.drop_duplicates (subset = ['corr'], keep = 'first')    

print (corr_df ['i'].value_counts ().nlargest (6))
print ('\n')
print (corr_df ['j'].value_counts ().nlargest (6))
y = np.asarray(churn_df['churn'])
y [0:5]
churn_df.columns
from sklearn import preprocessing
poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=False)
X = poly.fit_transform (np.asarray(churn_df [['tenure', 'employ', 'address', 'age', 'loglong', 'callcard', 'longten', 'longmon']]))
X[0:1]
X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR
yhat = LR.predict(X_test)
yhat [0:5]
yhat_prob = LR.predict_proba(X_test)
yhat_prob [0:5]
from sklearn.metrics import jaccard_score
jaccard_score(yhat, y_test)
from sklearn.metrics import classification_report, confusion_matrix
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
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
print (classification_report(y_test, yhat))

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)
