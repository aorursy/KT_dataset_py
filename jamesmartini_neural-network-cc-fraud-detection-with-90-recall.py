import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.info()
df['Class'].value_counts()
sns.set_style('darkgrid') #Set Seaborn plot style

fig, ax = plt.subplots(figsize=(7,5)) #Make the plot a little bigger.

ax.set_yscale('log') #Set the y-scale as logarithmic.

g = sns.countplot(x='Class', data=df, ax=ax) #Place the desired data on the plot.

g.set_xticklabels(['Legit','Fraud']) #Label the x-axis and y-axis correctly, and title the plot.

plt.xlabel('Class')

plt.ylabel('Count')

plt.title('Number of Transactions Per Class')
list_of_correlations= df.corr(method='pearson')

plt.figure(figsize=(11,7))

sns.heatmap(list_of_correlations,linewidths=0.005,linecolor='k')

plt.title('Correlation Heatmap')

plt.show()
np.abs(list_of_correlations['Class']).sort_values(ascending=False).head(5)
from sklearn.model_selection import train_test_split

X = df.drop('Class',axis=1)

y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X.head()
y.head()
type(X_train)
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
type(X_train)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
model = Sequential() #Initialise the model.



model.add(Dense(units=12,activation='relu')) #Add a single input layer.

model.add(Dense(units=1,activation='sigmoid')) #Add an output layer with sigmoid activation, as this is a binary classification.



model.compile(optimizer='rmsprop',loss='binary_crossentropy') #compile the model, choose how the cost function is optimised.
model.fit(X_train,y_train,                   #Pass in the training data

          epochs=10,                         #Specify how long we should train the model for (increasing this could lead to overfitting!)

          validation_data=(X_test,y_test))   #Specify validation data so that the model performance on the test set can be tracked, this is a good way to see if we get any overfitting!
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Loss versus training epoch')

plt.show()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
#Labels for the plot

classes = ['Legit','Fraud']

predicted_classes = ['Predicted ' + item for item in classes]

actual_classes = ['Actually ' + item for item in classes]



#Create a dataframe from confusion_matrix entries for easy plotting

conf_df = pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=predicted_classes,

            index=actual_classes)



#Seaborn heatmap

sns.heatmap(conf_df,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion matrix for the creditcard data set')

plt.show()
X = df.drop(['Class'],axis=1)

y = df['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
np.std(X_train,axis=0) #calculate the standard-deviation within each column
np.mean(X_train,axis=0) #Calculate the mean within each column
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



#Convert to numpy arrays for tensorflow

y_train = y_train.values

y_test = y_test.values
X_train_scaled = scaler.fit_transform(X_train,y_train)

X_test_scaled = scaler.transform(X_test)
np.std(X_train_scaled,axis=0) #Check that the column standard deviations are all 1
np.allclose(np.mean(X_train_scaled,axis=0), 0, rtol=1e-17) #Check that the column means are each very close to 0 (they are actually like 1e-18, so within machine to 0)
from tensorflow.keras.layers import Dropout
model = Sequential()

model.add(Dense(units=12,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')
model.fit(X_train_scaled,y_train,validation_data=(X_test_scaled,y_test),epochs=10)
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.show()
predictions_scaled = model.predict_classes(X_test_scaled)
print(classification_report(y_test,predictions_scaled))
conf_df_scaled = pd.DataFrame(data=confusion_matrix(y_test,predictions_scaled),columns=predicted_classes,

            index=actual_classes)



sns.heatmap(conf_df_scaled,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion Matrix for The Scaled Creditcard Data Set')

plt.show()
from sklearn.utils import resample

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



#Create a dataframe which houses only the training data (so we can use pandas to manipulate it)

df_train = X_train

df_train['Class'] = y_train



df_majority = df_train[df_train['Class']==0] #Seperate the legitimate transactions

df_minority = df_train[df_train['Class']==1] #From the fraudulent transactions



#Downsample using sklearn

df_majority_downsampled = resample(df_majority, 

                                 replace=False,    

                                 n_samples=348,     

                                 random_state=101)



df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled['Class'].value_counts()
X_train = df_downsampled.drop('Class',axis=1)

y_train = df_downsampled['Class']



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train,y_train)

X_test = scaler.transform(X_test)



y_test = y_test.values

y_train = y_train.values
model = Sequential()

model.add(Dense(units=12,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')
model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test),verbose=0)
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.show()
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
conf_df = pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=predicted_classes,

            index=actual_classes)



sns.heatmap(conf_df,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion matrix for the creditcard data set')

plt.show()
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
#Number of Legit transactions, Number of Fraud transactions

sum(y_train == 0), sum(y_train==1)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train,y_train)

X_test = scaler.transform(X_test)



y_train = y_train.values

y_test = y_test.values
from tensorflow.keras.layers import Dropout
model = Sequential()

model.add(Dense(units=12,activation='relu'))

model.add(Dropout(rate=0.25))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')
model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test),verbose=0)
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.show()
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
conf_df = pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=predicted_classes,

            index=actual_classes)



sns.heatmap(conf_df,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion matrix for the creditcard data set')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train,y_train)

X_test = scaler.transform(X_test)



y_train = y_train.values

y_test = y_test.values



model = Sequential()

model.add(Dense(units=12,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),verbose=0)
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.show()
probs = model.predict(X_test)

probs
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
#Plot the Precision-Recall curve and no-skill curve

no_skill = len(y_test[y_test==1]) / len(y_test)



plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='Model with No Skill')

plt.plot(recall, precision, marker='.', label='Our Neural Network Model')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Versus Recall for a Varying Threshold Parameter')



#Calculate the f1-score

fscore = (2 * precision * recall) / (precision + recall)



#Find the index of the largest f1-score and assign best threshold

ix = np.argmax(fscore)

print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

best_thresh = thresholds[ix]



plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Recall/Precision corresponding to optimal f1-score')

plt.legend()

plt.show()



#Assign classes based on the best threshold

classes_on_thresh = np.array([int(p[0] > best_thresh) for p in probs])

predictions = classes_on_thresh
print(classification_report(y_test,predictions))
classes = ['Legit','Fraud']

predicted_classes = ['Predicted ' + item for item in classes]

actual_classes = ['Actually ' + item for item in classes]



conf_df = pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=predicted_classes,

            index=actual_classes)



sns.heatmap(conf_df,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion matrix for the creditcard data set')

plt.show()
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train, y_train = SMOTE().fit_resample(X_train, y_train)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train,y_train)

X_test = scaler.transform(X_test)



y_train = y_train.values

y_test = y_test.values



model = Sequential()

model.add(Dense(units=12,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')

model.fit(X_train,y_train,epochs=15, validation_data = (X_test,y_test),verbose=0)
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.show()
probs = model.predict(X_test)

probs
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, probs)



#Find the best threshold

J = tpr - fpr

ix = np.argmax(J)

best_thresh = thresholds[ix]



#Plot the ROC curve

plt.plot([0,1], [0,1], linestyle='--', label='Model With No Skill')

plt.plot(fpr, tpr, marker='.', label='Our Neural Network Model')

plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='tpr-fpr maximised point (optimal threshold)')



#Label your axes

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()



#Use the optimal threshold to generate predictions

classes_on_thresh = np.array([int(p[0] > best_thresh) for p in probs])

predictions = classes_on_thresh
print(classification_report(y_test,predictions))
conf_df = pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=predicted_classes,

            index=actual_classes)



sns.heatmap(conf_df,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion matrix for the creditcard data set')

plt.show()
from sklearn.utils import resample

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



df_train = X_train

df_train['Class'] = y_train



df_majority = df_train[df_train['Class']==0] #Legit transactions

df_minority = df_train[df_train['Class']==1] #Fraudulent transations



df_majority_downsampled = resample(df_majority, 

                                 replace=False,    

                                 n_samples=348,     

                                 random_state=101)

 

df_downsampled = pd.concat([df_majority_downsampled, df_minority])



X_train = df_downsampled.drop('Class',axis=1)

y_train = df_downsampled['Class']



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train,y_train)

X_test = scaler.transform(X_test)



y_test = y_test.values

y_train = y_train.values



model = Sequential()

model.add(Dense(units=12,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy')



model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test),verbose=0)
losses = pd.DataFrame(model.history.history)

losses.plot()

plt.show()
probs = model.predict(X_test)



fpr, tpr, thresholds = roc_curve(y_test, probs)



#Find the best threshold

J = tpr - fpr

ix = np.argmax(J)

best_thresh = thresholds[ix]



#Plot the ROC curve

plt.plot([0,1], [0,1], linestyle='--', label='Model With No Skill')

plt.plot(fpr, tpr, marker='.', label='Our Neural Network Model')

plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='tpr-fpr maximised point')



#Label your axes

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()



#Use the optimal threshold to generate predictions

classes_on_thresh = np.array([int(p[0] > best_thresh) for p in probs])

predictions = classes_on_thresh
print(classification_report(y_test,predictions))
classes = ['Legit','Fraud']

predicted_classes = ['Predicted ' + item for item in classes]

actual_classes = ['Actually ' + item for item in classes]



conf_df = pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=predicted_classes,

            index=actual_classes)



sns.heatmap(conf_df,annot=True,fmt='g',cbar=False)

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')

plt.xticks(rotation=20)

plt.yticks(rotation=30)

plt.title('Confusion matrix for the creditcard data set')

plt.show()