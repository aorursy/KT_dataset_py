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
df = pd.read_csv('/kaggle/input/voicegender/voice.csv')
df.head()
df.describe().transpose()
l = df.columns
df.isnull().sum()
def encode(x):

    if x == 'male':

        return 1

    else:

        return 0
df['label'].unique()
df['label'] = df['label'].apply(encode)
import matplotlib.pyplot as plt 

import seaborn as sns 
plt.figure(figsize=(16,8))

sns.kdeplot(df['meanfreq'])
sns.countplot(df['label'])
from sklearn.metrics import classification_report, confusion_matrix



from sklearn.linear_model import LogisticRegression



from sklearn.svm import SVC



from sklearn.naive_bayes import MultinomialNB



from sklearn.ensemble import RandomForestClassifier



from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df[l[:-1]],df[l[-1]],random_state=101,test_size=0.33)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



scaled_train = scaler.fit_transform(X_train)

scaled_test = scaler.transform(X_test)
logistic_model = LogisticRegression()
logistic_model.fit(scaled_train,y_train)
preds = logistic_model.predict(scaled_test)
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))
pca = PCA(n_components=10)
reduced_train = pca.fit_transform(scaled_train)

reduced_test = pca.transform(scaled_test)
svc = SVC()
svc.fit(reduced_train,y_train)
preds = svc.predict(reduced_test)
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))
from sklearn.metrics import accuracy_score
accuracy = []

i_vals = []

for i in range(1,21):

    

    pca = PCA(n_components=i)

    reduced_train = pca.fit_transform(scaled_train)

    reduced_test = pca.transform(scaled_test)

    

    svc = SVC()

    svc.fit(reduced_train,y_train)

    pred = svc.predict(reduced_test)

    

    accuracy.append(accuracy_score(y_test,pred))

    i_vals.append(i)
plt.figure(figsize=(12,6))

plt.plot(i_vals, accuracy)
i_vals = np.array(i_vals).reshape(20,1)

accuracy = np.array(accuracy).reshape(20,1)

np.argmax(accuracy)
mnb = MultinomialNB()
from sklearn.preprocessing import MinMaxScaler



mms = MinMaxScaler()



scaled_nn_train = mms.fit_transform(X_train)

scaled_nn_test = mms.transform(X_test)



mnb.fit(scaled_nn_train,y_train)

preds = mnb.predict(scaled_nn_test)



print(classification_report(y_test,preds))
i_vals = []

accu = [] 

for i in range(1,200):

    

    random_forest = RandomForestClassifier(n_estimators=i)

    random_forest.fit(scaled_nn_train,y_train)

    

    preds = random_forest.predict(scaled_nn_test)

    

    i_vals.append(i)

    accu.append(accuracy_score(y_test,preds))
plt.figure(figsize=(15,6))

plt.plot(i_vals,accu)
i_vals = np.array(i_vals)



accu = np.array(accu)
i_vals[np.argmax(accu)]
random_forest = RandomForestClassifier(n_estimators=134)

random_forest.fit(scaled_nn_train,y_train)

    

preds = random_forest.predict(scaled_nn_test)
print(classification_report(y_test,preds))
from tensorflow.keras.layers import Dense



from tensorflow.keras.models import Sequential



from tensorflow.keras.callbacks import EarlyStopping



model = Sequential()



model.add(Dense(10,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(1,activation='sigmoid'))





model.compile(optimizer = 'adam', metrics = ['acc'], loss='binary_crossentropy')

call = [EarlyStopping(monitor='val_acc', patience=25)]
model.fit(reduced_train, y_train, validation_data=(reduced_test,y_test), epochs=700, callbacks=call)
plt.figure(figsize=(16,6))

pd.DataFrame(model.history.history).plot()
preds = model.predict_classes(reduced_test)
print(classification_report(y_test,preds))