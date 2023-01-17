# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import statistics
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/wine-daset/wineDATA.CSV")
print("class1=",len(df[df['Class']==1]), "class2=",len(df[df['Class']==2]), "class3=",len(df[df['Class']==3]))
#pré-processamento para corrigir escalas muito discrepantes.
def corrigir_escalas(df):
    
    for j in range(len(df.columns)):
        for k in range(len(df)):
            valor_min=df.iloc[:,j:j+1].min()
            valor_max=df.iloc[:,j:j+1].max()
            valor_atual=df.iloc[k:k+1,j:j+1].values[0][0]
            df.iloc[k:k+1,:]=(valor_atual- valor_min[0])/(valor_max[0] - valor_min[0])
        return df
x=df.iloc[:,1:]
y=df['Class']
x=corrigir_escalas(x)
x.head()
# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)
#bin_y = np_utils.to_categorical(encoded_Y)
#bin_y 
X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.5)


clf3= MLPClassifier(max_iter=500,learning_rate_init=0.1,verbose=0).fit(X_train, y_train)

y_pred=clf3.predict(X_test)
accuracy_score(y_test,y_pred)*100
def holdout(num):
    lista=[]
    for k in range(num):
        X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.5)
        clf3= MLPClassifier(max_iter=500,learning_rate_init=0.1,verbose=0).fit(X_train, y_train)
        y_pred=clf3.predict(X_test)
        acc=accuracy_score(y_test,y_pred)*100
        lista.append(acc)
    
    return statistics.mean(lista),statistics.stdev(lista)
holdout(30)        
df_spiral=pd.read_csv("/kaggle/input/data-spiral/spiral.csv")
x=df_spiral.iloc[:,:2]
y=df_spiral['class']
#encode class values as integers

X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.3)
#df_spiral=df_spiral.astype(str)

#case 1
clf1= MLPClassifier( max_iter=500,learning_rate_init=0.3,verbose=0,
                   hidden_layer_sizes=(4,),activation='logistic').fit(X_train, y_train)


y_pred=clf1.predict(X_test)
accuracy_score(y_test,y_pred)*100
#clf.score(X_test, y_test)

#case 2
clf2 = MLPClassifier(max_iter=500,learning_rate_init=0.3,verbose=0,
                   hidden_layer_sizes=(4,2),activation='logistic').fit(X_train, y_train)


y_pred=clf2.predict(X_test)
accuracy_score(y_test,y_pred)*100

#case 3
clf3= MLPClassifier(max_iter=500,learning_rate_init=0.3,verbose=0,
                   hidden_layer_sizes=(4,3),activation='logistic').fit(X_train, y_train)


y_pred=clf3.predict(X_test)
accuracy_score(y_test,y_pred)*100

#case 4
clf4= MLPClassifier(max_iter=1000,learning_rate_init=0.3,verbose=0,
                   hidden_layer_sizes=(4,3),activation='logistic').fit(X_train, y_train)


y_pred=clf4.predict(X_test)
accuracy_score(y_test,y_pred)*100

#variação 1
clf3= MLPClassifier(max_iter=1000,learning_rate_init=0.1,verbose=0,
                   hidden_layer_sizes=(9,3),activation='logistic').fit(X_train, y_train)


y_pred=clf3.predict(X_test)
accuracy_score(y_test,y_pred)*100

#variação 2
clf3= MLPClassifier(max_iter=500,learning_rate_init=0.2,verbose=0,
                   hidden_layer_sizes=(7,3),activation='logistic').fit(X_train, y_train)


y_pred=clf3.predict(X_test)
accuracy_score(y_test,y_pred)*100

#variação 2
clf3= MLPClassifier(max_iter=500,learning_rate_init=0.1,verbose=0,
                   hidden_layer_sizes=(8,3),activation='logistic').fit(X_train, y_train)


y_pred=clf3.predict(X_test)
accuracy_score(y_test,y_pred)*100