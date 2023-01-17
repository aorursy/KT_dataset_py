# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/faulty-steel-plates/faults.csv")

raw_data=data.copy()

pd.set_option('display.max_columns', None)

data.head()
data.describe(include='all')
nas=data.isnull().sum()

nas.values
corr=data.corr()

corr.style.background_gradient(cmap='coolwarm')
# from above matrix it is clear that x_minimum,y_minimum is correlated with x_maximum,y_maximum. Pixels_Areas is correlated with X_Perimeter,Y_Perimeter,Sum_of_Luminosity.

#I am taking x_minimum,y_minimum and pixels_areas
data2=data.drop(['X_Maximum','Y_Maximum','X_Perimeter','Y_Perimeter','Sum_of_Luminosity'],axis=1)

data2.shape
x=data2.iloc[:,:22]

y=data2.iloc[:,22:]

targets=(data2.iloc[:,-7:]==1).idxmax(1)

targets







from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y_new=le.fit_transform(targets)

y_new=pd.DataFrame(y_new)

#x.plot(kind="density", layout=(6,5), 

 #            subplots=True,sharex=False, sharey=False, figsize=(15,15))

#plt.show()
#x.hist(figsize=(15,15))

#plt.show()
x.plot(kind='box', layout=(6,5), 

             subplots=True,sharex=False, sharey=False, figsize=(15,15) )

plt.show()
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(x))

print(z)
threshold = 3

drop_rows=np.where(z > 3)

drop_rows
x_without_o = x[(z < 3).all(axis=1)]

x_without_o.shape
y_without_o=y_new.drop(drop_rows[0],axis=0)

y_without_o.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_scaled= sc.fit_transform(x_without_o)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_without_o, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.metrics import accuracy_score



score =accuracy_score(y_test,y_pred)

score
from sklearn import svm



classifier = svm.SVC(kernel='linear') # Linear Kernel

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score



score1 =accuracy_score(y_test,y_pred)

score1
from sklearn import svm



classifier = svm.SVC(kernel='linear') # Linear Kernel

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score



score2 =accuracy_score(y_test,y_pred)

score2
#using newral network
train_counts=int(0.8*x_without_o .shape[0])

validation_counts=int(0.1*x_without_o .shape[0])

train_counts
x_train=x_scaled[:train_counts]

x_validation=x_scaled[train_counts:train_counts+validation_counts]

x_test=x_scaled[train_counts+validation_counts:]

np.shape(x_test)

y_train=y_without_o[:train_counts]

y_train=np.array(y_train)

y_validation=y_without_o[train_counts:train_counts+validation_counts]

y_validation=np.array(y_validation)

y_test=y_without_o[train_counts+validation_counts:]

y_test=np.array(y_test)





import tensorflow as tf
input_size=22

output_size=7

hidden_size=50



model=tf.keras.Sequential([

    

    tf.keras.layers.Dense(hidden_size,activation='relu'),

    tf.keras.layers.Dense(hidden_size,activation='relu'),

    tf.keras.layers.Dense(output_size,activation='softmax')

])
from keras.optimizers import SGD

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
batch_size=128

max_epochs=100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)






model.fit(x_train,

          y_train, 

          batch_size=batch_size,

          epochs=max_epochs, 

          callbacks=[early_stopping], 

          validation_data=(x_validation, y_validation),

          verbose = 2 

          )  
test_loss,test_accuracy=model.evaluate(x_test,y_test)