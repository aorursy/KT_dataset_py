import numpy as np

import pandas as pd
df = pd.read_csv("../input/Chickpea.data.csv")

df.head()
#Na Handling

df.isnull().values.any()
df=df.dropna()
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X = df.drop(['Predictor'], axis=1)

X_col = X.columns
y = df['Predictor']


#Savitzky-Golay filter with second degree derivative.

from scipy.signal import savgol_filter 



sg=savgol_filter(X,window_length=11, polyorder=3, deriv=2, delta=1.0)
sg_x=pd.DataFrame(sg, columns=X_col)

sg_x.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sg_x, y,

                                                    train_size=0.8,

                                                    random_state=23,stratify = y)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components=18)  

X_train = lda.fit_transform(X_train, y_train)  

X_test = lda.transform(X_test)
from sklearn import svm

clf = svm.SVC(kernel="linear")

clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test) 

from sklearn.metrics import confusion_matrix  

from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test, y_pred)  

print(cm)  

print('Accuracy' + str(accuracy_score(y_test, y_pred))) 