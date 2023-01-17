# Importing library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn
# Simulated Class 1 objects

object_id_1 = np.arange(1,501,1 ,dtype=int)

object_weight_1 = np.random.uniform(1,40,500)

object_volume_1 = np.random.randint(10,50,500)

object_freq_per_day_1 =  np.random.randint(1,20,500)

class_1 = np.ones(500,int)*1



# Simulated Class 2 objects

object_id_2 = np.arange(501,1501,1, dtype=int)

object_weight_2 = np.random.uniform(40,60,1000)

object_volume_2 = np.random.randint(50,80,1000)

object_freq_per_day_2 =  np.random.randint(20,30,1000)

class_2= np.ones(1000,int)*2



# Simulated Class 3 objects

object_id_3 = np.arange(1,501,1,dtype=int)

object_weight_3= np.random.uniform(60,80,500)

object_volume_3 = np.random.randint(80,100,500)

object_freq_per_day_3 =  np.random.randint(30,35,500)

class_3= np.ones(500,int)*3

from scipy import vstack

data = vstack(

        (np.hstack((object_id_1.reshape(-1,1), object_weight_1.reshape(-1,1), object_volume_1.reshape(-1,1), object_freq_per_day_1.reshape(-1,1), class_1.reshape(-1,1))),

        np.hstack((object_id_2.reshape(-1,1), object_weight_2.reshape(-1,1), object_volume_2.reshape(-1,1), object_freq_per_day_2.reshape(-1,1), class_2.reshape(-1,1))),

        np.hstack((object_id_3.reshape(-1,1), object_weight_3.reshape(-1,1), object_volume_3.reshape(-1,1), object_freq_per_day_3.reshape(-1,1), class_3.reshape(-1,1))))

    )



dataframe  = pd.DataFrame(data,columns=['object_id','object_weights','object_volume','object_freq_per_day','class'])

dataframe.sample(10)
# Size of the dataset

dataframe.shape
# Distribution of class labe;

sn.countplot(dataframe['class'])
# Disctribution of weight

sn.distplot(dataframe['object_weights'])

plt.xlabel("Feature name: object_weights")

plt.ylabel("PDF")
# Disctribution of object_volume

sn.distplot(dataframe['object_volume'])

plt.xlabel("Feature name: object_volume")

plt.ylabel("PDF")
# Disctribution of object_freq_per_day

sn.distplot(dataframe['object_freq_per_day'],)

plt.xlabel("Feature name: object_freq_per_day")

plt.ylabel("PDF")

sn.pairplot(dataframe.drop(['object_id', 'class'], axis=1),)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
# train test split

X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(['object_id','class'],axis = 1).values, np.array(dataframe['class'].values,int), test_size=0.33, stratify = np.array(dataframe['class'].values,int),random_state=42) 
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_train_normalised = min_max_scaler.fit_transform(X_train)

X_test_normalised = min_max_scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt



"""

y_true : array, shape = [n_samples] or [n_samples, n_classes]

True binary labels or binary label indicators.



y_score : array, shape = [n_samples] or [n_samples, n_classes]

Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of

decisions (as returned by “decision_function” on some classifiers). 

For binary y_true, y_score is supposed to be the score of the class with greater label.



"""



train_loss = []

cv_loss = []

c_range=[10e-5,10e-4,10e-3,10e-2,1,10,10e1,10e2,10e3]

for i in c_range:

    

    clf=LogisticRegression(penalty='l2', C=i,class_weight='balanced')

    clf.fit(X_train_normalised, y_train)

    

    # Predicting

    y_train_pred = clf.predict_proba(X_train_normalised)

    y_cv_pred = clf.predict_proba(X_test_normalised)

    

    # Loss metric storing

    train_loss.append(log_loss(y_train, y_train_pred))

    cv_loss.append(log_loss(y_test, y_cv_pred))

    

    

# Visualising and finding optimal parameter 

#plt.plot(np.arange(1,10,1), train_loss, label='Train loss')

plt.plot(np.arange(1,10,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,10,1), (10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()







## Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_c = c_range[np.argmin(cv_loss)] #optimum 'alpha'



# Logistic regression training

print(f"Traing using optimal alpha:  {c_range[np.argmin(cv_loss)]}\n")

clf=LogisticRegression(penalty='l2', C=optimum_c,class_weight='balanced')

clf.fit(X_train_normalised, y_train)

    

y_pred = clf.predict(X_test_normalised)

y_pred_proba = clf.predict_proba(X_test_normalised)



# Result track

accuracy = accuracy_score(y_test,y_pred)

bal_accuracy = balanced_accuracy_score(y_test,y_pred)

logloss = log_loss(y_test,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')





#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_test),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_test,y_pred)) 