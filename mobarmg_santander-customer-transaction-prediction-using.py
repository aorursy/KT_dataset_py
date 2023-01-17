%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import keras as K

from sklearn.preprocessing import MinMaxScaler 

from keras.callbacks import EarlyStopping





from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

tr = pd.read_csv('../input/train.csv',header=0)
from sklearn.utils import resample



df = tr

# Separate majority and minority classes

df_majority = df[df.target==0]

df_minority = df[df.target==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=179902,    # to match majority class

                                 random_state=123) # reproducible results

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts





df_upsampled.target.value_counts()



# 1    576

# 0    576

# Name: balance, dtype: int64
y = df_upsampled.target

x = df_upsampled.drop(['target','ID_code'], axis=1)
def plot_roc(pred,y):

    fpr, tpr, _ = roc_curve(y, pred)

    roc_auc = auc(fpr, tpr)



    plt.figure()

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC)')

    plt.legend(loc="lower right")

    plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.5, random_state = 0)

clf_4 = RandomForestClassifier()

clf_4.fit(xTrain, yTrain)

 

# Predict on training set

pred_y_4 = clf_4.predict(xTest)

 

# Is our model still predicting just one class?

# [0 1]
print( accuracy_score(pred_y_4, yTest) )

plot_roc(pred_y_4,yTest)
tst = pd.read_csv('../input/test.csv',header=0)

tst = tst.set_index('ID_code')

#tst =scaler.fit_transform(tst) 

rslt = clf_4.predict(tst)
tst['target'] = rslt

final = tst[['target']]

final.to_csv('submission.csv')
tst.target.value_counts()
pred_y_4