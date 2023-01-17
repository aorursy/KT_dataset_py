import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
df = pd.read_csv('../input/creditcard.csv')
df.Class.sum()/df.shape[0]
#Building a simple model.
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=6,max_features='sqrt')

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
kfold = 5
M =0
for i in range(kfold):
    train,test = train_test_split(df,train_size=0.7,test_size = 0.3)
    y_train,y_test = train['Class'],test['Class']
    del train['Class']; del test['Class']
    rf.fit(train,y_train)    
    Y = rf.predict(test)
    M = M+confusion_matrix(y_pred=Y,y_true=y_test)
plot_confusion_matrix(M,classes=['Legitimate','Fraudulent'])
fraudulent = df[df.Class==1]
legit = df[df.Class==0].sample(frac=1/6)  
df2 = pd.concat([legit,fraudulent]).reset_index(drop=True)
kfold = 5
for i in range(kfold):
    train,test = train_test_split(df2,train_size=0.7,test_size = 0.3)
    y_train,y_test = train['Class'],test['Class']
    del train['Class']; del test['Class']
    model_lgb.fit(train,y_train)    
    Y = model_lgb.predict(test)
    M = M + confusion_matrix(y_pred=Y,y_true=y_test)
plot_confusion_matrix(M,classes=['Legitimate','Fraudulent'])

