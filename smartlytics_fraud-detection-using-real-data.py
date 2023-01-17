# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# The dataset has had PCA applied to the features to provide customer anonymity. 28 Columns (V1-V28) are the resulting principal components.

data = pd.read_csv('../input/creditcard.csv')

data
import numpy as np

import seaborn as sns

fig = plt.figure(figsize=(15,10))

plt.tight_layout()

data2 = data[data.Class==0]

data3 = data[data.Class==1]

sns.set_style('whitegrid')

for i in range(1,5):

    plt.subplot(2,2,i)

    plt.tight_layout()

    sns.kdeplot(np.array(data2['V'+str(i)].values), bw=0.5,label='Not Fraud')

    sns.kdeplot(np.array(data3['V'+str(i)].values), bw=0.5,label='Fraud')

    plt.title('V'+str(i))

    plt.ylabel('Probability Density')

    plt.xlabel('Value')
class_size = data.groupby('Class').size().values

fraud_proportion = class_size/class_size.sum()

fraud_proportion = [str(round(100*f,2))+'%' for f in fraud_proportion]



labels = 'Not Fraud: '+fraud_proportion[0],'Fraud: '+fraud_proportion[1]

plt.pie(class_size,labels=labels)

centre = plt.Circle( (0,0), 0.6, color='white')

p=plt.gcf()

p.gca().add_artist(centre)

plt.axis('equal')

plt.title('Transaction Class')

plt.show()
plt.rcParams['axes.facecolor'] = 'whitesmoke'

plt.figure(figsize = (12,12))



plt.subplot(4,1,1)

plt.hist(data[data.Class==0].Time.values,bins = 50,color='cornflowerblue')

plt.title('Time - Nonfraudulent transactions')

plt.ylabel('Transactions')



plt.subplot(4,1,2)

plt.hist(data[data.Class==1].Time.values,bins = 50,color='lightcoral')

plt.title('Time - Fraudulent transactions')

plt.ylabel('Transactions')

plt.xlabel('Time')



plt.subplot(4,1,3)

plt.hist(data[data.Class==0].Amount.values,bins = 50,color='cornflowerblue')

plt.title('Transaction Amount - Nonfraudulent transactions')

plt.ylabel('Transactions')

plt.xlabel('Transaction Amount')



plt.subplot(4,1,4)

plt.hist(data[data.Class==1].Amount.values,bins = 50,color='lightcoral')

plt.title('Transaction Amount - Fraudulent transactions')

plt.ylabel('Transactions')

plt.xlabel('Transaction Amount')



plt.tight_layout()

plt.show()
# Independent and dependent variables.

X = data.drop(columns=['Class']).values

y = data.Class.values



# Training and Test sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators=100)

rfc_model.fit(X_train,y_train)

y_pred = rfc_model.predict(X_test)



from sklearn.metrics import confusion_matrix, roc_auc_score

cm = confusion_matrix(y_test,y_pred)
roc_auc_score(y_test,y_pred)
fig = plt.figure(figsize=(12,6),)

fig.suptitle("Random Forest Fraud Detection", fontsize="large")

plt.subplot(121)

plt.bar(['Undetected','Detected'],cm[1,:]/np.sum(cm[1,:]),edgecolor='k',linewidth=2.0,color='lightcoral')

plt.ylim([0,1])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.title('Fraud: '+str(round(cm[1,1]/(cm[1,0]+cm[1,1])*100,1))+'% success')

plt.subplot(122)

plt.bar(['Not Classified','Falsely Classifed'],cm[0,:]/np.sum(cm[0,:]),edgecolor='k',linewidth=2.0,color='cornflowerblue')

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.title('Not fraud: '+str(round(cm[0,0]/(cm[0,0]+cm[0,1])*100,2))+'% success')

plt.show()
# Feature Scaling for the remaining methods

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)
# Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs')

lr_model.fit(X_train,y_train)



y_pred = lr_model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
roc_auc_score(y_test,y_pred)
fig = plt.figure(figsize=(12,6),)

fig.suptitle("Logistic Regression Fraud Detection", fontsize="large")

plt.subplot(121)

plt.bar(['Undetected','Detected'],cm[1,:]/np.sum(cm[1,:]),edgecolor='k',linewidth=2.0,color='lightcoral')

plt.ylim([0,1])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.title('Fraud: '+str(round(cm[1,1]/(cm[1,0]+cm[1,1])*100,1))+'% success')

plt.subplot(122)

plt.bar(['Not Classified','Falsely Classifed'],cm[0,:]/np.sum(cm[0,:]),edgecolor='k',linewidth=2.0,color='cornflowerblue')

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.title('Not fraud: '+str(round(cm[0,0]/(cm[0,0]+cm[0,1])*100,2))+'% success')

plt.show()
# ANN Classifier



import keras

from keras.models import Sequential

from keras.layers import Dense

ann_model = Sequential()

ann_model.add(Dense(activation="relu", input_dim=30, units=15, kernel_initializer="uniform"))

ann_model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

ann_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann_model.fit(X_train,y_train,batch_size=10,epochs=4)



y_pred = ann_model.predict(X_test) > 0.5

cm = confusion_matrix(y_test,y_pred)
roc_auc_score(y_test,y_pred)
fig = plt.figure(figsize=(12,6),)

fig.suptitle("ANN Fraud Detection", fontsize="large")

plt.subplot(121)

plt.bar(['Undetected','Detected'],cm[1,:]/np.sum(cm[1,:]),edgecolor='k',linewidth=2.0,color='lightcoral')

plt.ylim([0,1])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.title('Fraud: '+str(round(cm[1,1]/(cm[1,0]+cm[1,1])*100,1))+'% success')

plt.subplot(122)

plt.bar(['Not Classified','Falsely Classifed'],cm[0,:]/np.sum(cm[0,:]),edgecolor='k',linewidth=2.0,color='cornflowerblue')

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.title('Not fraud: '+str(round(cm[0,0]/(cm[0,0]+cm[0,1])*100,2))+'% success')

plt.show()