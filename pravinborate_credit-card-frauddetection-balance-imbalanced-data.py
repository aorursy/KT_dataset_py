# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_auc_score,roc_curve

sns.set()

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data['Class'].value_counts()
plt.figure(figsize=(20,2))

sns.countplot(y=data['Class'])

plt.savefig('countofdata.png')

plt.show()
print(f'Percentage of data where class = 1 is : {(len(data[data.Class == 1])/ len(data[data.Class == 0]))*100}')
#Create train and test dataset

from sklearn.model_selection import train_test_split

X = data.drop('Class',axis=1)

y = data['Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(f'Shape of X_train :{X_train.shape}')

print(f'Shape of X_train :{X_test.shape}')

print(f'Shape of X_train :{y_train.shape}')

print(f'Shape of X_train :{y_test.shape}')
# DummyClassifier to predict only target 0

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')

dummy.fit(X_train,y_train)
dummy_pred = dummy.predict(X_test)

#Check for the unique labels

print(f'Unique predicted labels : {np.unique(dummy_pred)}')
#Check the accuracy for unique predicated labels

print(f'Accuracy for the test dataset : {accuracy_score(dummy_pred,y_test)}')
#Now let's use the Logistic regression model to check the accuracy on inbalanced data set

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
#Predict the value by Logistic Regression

lr.fit(X_train,y_train)
#Predict on test dataset

lr_pred = lr.predict(X_test)
#check the accuracy

accuracy_score(lr_pred,y_test)
# Checking unique values

predictions = pd.DataFrame(lr_pred)

predictions[0].value_counts()
from sklearn.metrics import confusion_matrix,f1_score,classification_report,recall_score
f1_score(y_test,lr_pred)
pd.DataFrame(confusion_matrix(y_test,lr_pred))
sns.heatmap(confusion_matrix(y_test,lr_pred),cmap='viridis',annot=True,fmt='.2f')

plt.savefig('lr.png')

plt.show()
recall_score(y_test,lr_pred)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
random_forest = RandomForestClassifier()

svc = SVC()
random_forest.fit(X_train,y_train)
random_forest_pred = random_forest.predict(X_test)
accuracy_score(random_forest_pred,y_test)
f1_score(random_forest_pred,y_test)
pd.DataFrame(confusion_matrix(random_forest_pred,y_test))
sns.heatmap(confusion_matrix(y_test,random_forest_pred),cmap='viridis',annot=True,fmt='.2f')

plt.savefig('random.png')

plt.show()
recall_score(random_forest_pred,y_test)
print(classification_report(random_forest_pred,y_test))
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dropout

from tensorflow.keras.utils import plot_model
model = Sequential([

    Dense(units=16,input_dim = 30,activation='relu'),

    Dense(units=24,activation='relu'),

    Dropout(0.5),

    Dense(units=20,activation='relu'),

    Dense(units=24,activation='relu'),

    Dense(1,activation='sigmoid')

])
plot_model(model)
model.summary()
##Training of model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=15,epochs=5)
score = model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)

y_test_copy = pd.DataFrame(y_test)
y_pred = y_pred.astype(int)
pd.DataFrame(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='YlOrBr',fmt='.2f')

plt.savefig('deep.png')

plt.show()
from sklearn.utils import resample



#Seprate the input feature and target 

X = data.drop('Class',axis=1)

y= data['Class']


#As told split the data into train and test dataset

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25)

print(f'Shape of X_train :{X_train.shape}')

print(f'Shape of X_train :{X_test.shape}')

print(f'Shape of X_train :{y_train.shape}')

print(f'Shape of X_train :{y_test.shape}')
# concatenate our training data back together

X = pd.concat([X_train, y_train], axis=1)

X.head()
#Seperate minority and majority class:

not_fraud = X[X['Class'] == 0]

fraud = X[X['Class'] == 1]

print(f'Total sample which are not fraud : {len(not_fraud)}')

print(f'Total Fraud samples : {len(fraud)}')
#Now use the oversampling techniques

random_sampling = resample(fraud,

                          replace=True,

                           n_samples = len(not_fraud),

                           random_state = 42

                          )



#combine minority and upsample data

upsample = pd.concat([not_fraud,random_sampling])



#Check new values are balances for the both classes or not

upsample['Class'].value_counts()
X_train = upsample.drop('Class',axis=1)

y_train = upsample['Class']



lr_model = LogisticRegression()

lr_model.fit(X_train,y_train)
upsampled_pred = lr_model.predict(X_test)
#check for the accuracy_score

accuracy_score(y_test,upsampled_pred)
#F1 score is

f1_score(y_test, upsampled_pred)
sns.heatmap(confusion_matrix(y_test,upsampled_pred),annot=True,fmt='.2f',cmap='YlOrBr')

plt.savefig('ligit_after_oversample.png')

plt.show()
print(classification_report(y_test,upsampled_pred))
# downsample majority

not_fraud_downsampled = resample(not_fraud,

                                replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled = pd.concat([not_fraud_downsampled, fraud])



# checking counts

downsampled.Class.value_counts()
# trying logistic regression again with the undersampled dataset



y_train = downsampled.Class

X_train = downsampled.drop('Class', axis=1)



undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)



undersampled_pred = undersampled.predict(X_test)
accuracy_score(y_test, undersampled_pred)

# f1 score

f1_score(y_test, undersampled_pred)
acc
print(classification_report(y_test,undersampled_pred))
#Build the model for the Random Forest

random_forest_undersampled = RandomForestClassifier()

random_forest_undersampled.fit(X_train,y_train)
#Predict the value by using random forest

random_forest_undersampled_pred = undersampled.predict(X_test)
accuracy_score(y_test,random_forest_undersampled_pred)
f1_score(y_test,random_forest_undersampled_pred)
sns.heatmap(confusion_matrix(y_test,random_forest_undersampled_pred),annot=True,fmt='.2f',cmap='YlGnBu')

plt.savefig('rand_after_oversample.png')

plt.show()
print(classification_report(y_test,random_forest_undersampled_pred))
#import the libaray for SMOTE

from imblearn.over_sampling import SMOTE



# Separate input features and target

y = data["Class"]

X = data.drop('Class', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



sm = SMOTE(random_state=27,sampling_strategy=1.0)

X_train, y_train = sm.fit_sample(X_train, y_train)
smote_logistic = LogisticRegression()

smote_logistic.fit(X_train,y_train)
smote_pred = smote_logistic.predict(X_test)
# Checking accuracy

accuracy_score(y_test, smote_pred)
# f1 score

f1_score(y_test, smote_pred)
sns.heatmap(confusion_matrix(y_test,smote_pred),annot=True,fmt='.2f',cmap='YlGnBu')

plt.savefig('rand_after_oversample.png')

plt.show()
smote_random_forest = RandomForestClassifier()

smote_random_forest.fit(X_train,y_train)
smote_rand_pred = smote_random_forest.predict(X_test)
sns.heatmap(confusion_matrix(y_test,smote_rand_pred),annot=True,fmt='.2f',cmap='YlGnBu')

plt.savefig('rand_after_smote.png')

plt.show()
accuracy_score(y_test,smote_rand_pred)
#F1_score is

f1_score(y_test,smote_rand_pred)
print(classification_report(y_test,smote_rand_pred))