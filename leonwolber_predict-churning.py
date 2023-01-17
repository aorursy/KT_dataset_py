import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
data = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")
data.head()
# drop unnecessary columns



data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
data.head()
data.info()
# transform vairbales in right format



for col in ['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']:

    data[col] = data[col].astype('category')
data.info()
data.describe().T
plt.figure(figsize=(12,5))



sns.distplot(data['EstimatedSalary'],kde = False)
plt.figure(figsize=(12,5))



sns.distplot(data['CreditScore'],kde = False)
plt.figure(figsize=(12,5))



sns.distplot(data[(data['CreditScore'] <=840) & (data['CreditScore'] >=450)]['CreditScore'],kde = False)
data = data[(data['CreditScore'] <=840) & (data['CreditScore'] >=450)]
plt.figure(figsize=(12,5))



sns.distplot(data['Balance'],kde = False)
balance = [0 if i == 0 else 1 for i in data['Balance']]
pd.Series(balance).value_counts()/len(data)*100
# add the new binary variable and drop the original 



data['has_balance'] = pd.Series(balance)

data = data.drop('Balance', axis = 1)
data['Geography'].value_counts()/len(data)*100
data['NumOfProducts'].value_counts()
data['more_than1product'] = pd.Series([0 if i == 1 else 1 for i in data['NumOfProducts']])

data = data.drop('NumOfProducts', axis=1)
data['more_than1product'].value_counts()
data.head()
data['IsActiveMember'].value_counts()
data['HasCrCard'].value_counts()
data['Tenure'].value_counts()
data['Gender'].value_counts()
data.head()
data.info()
data['has_balance'] = pd.Categorical(data['has_balance'])

data['more_than1product'] = pd.Categorical(data['more_than1product'])
sns.heatmap(data.corr(), annot=True)
len(data)
backup = data.copy()
data = backup
data['Exited'].value_counts()
from sklearn.utils import resample



# # Separate majority and minority classes

# df_majority = data[data.Exited==0]

# df_minority = data[data.Exited==1]

#  

# # Upsample minority class

# df_minority_upsampled = resample(df_minority, 

#                                  replace=True,     # sample with replacement

#                                  n_samples=7600,    # to match majority class

#                                  random_state=404) # reproducible results

#  

# # Combine majority class with upsampled minority class

# df_upsampled = pd.concat([df_majority, df_minority_upsampled])

#  

# # Display new class counts

# df_upsampled.Exited.value_counts()
len(data)
data.head()
data.columns
data.info()
from numpy import array

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten, Dropout



from keras import optimizers

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping



from sklearn.model_selection import train_test_split

from sklearn import metrics



from scipy.stats import zscore
def encode_columns(column, data):

    

    data = pd.concat([data,pd.get_dummies(data[column],prefix=column)],axis=1)

    data.drop(column, axis=1, inplace=True)

    

    return data
data.columns
### ------------- encode categorical columns ----------------



categorical_columns = ['Geography',

                       'Gender',

                       'HasCrCard',

                       'IsActiveMember',

                       'has_balance',

                       'more_than1product']

    

for col in categorical_columns:

    data=encode_columns(col,data)
data.info()
data['CreditScore'] = zscore(data['CreditScore'])

data['Age'] = zscore(data['Age'])

data['Tenure'] = zscore(data['Tenure'])

data['EstimatedSalary'] = zscore(data['EstimatedSalary'])
x = data.drop('Exited', axis=1)

y = data['Exited']
x = np.asarray(x)

y = np.asarray(y)
from tensorflow.keras.callbacks import EarlyStopping
# Split into train/test

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



model = Sequential()

model.add(Dense(100, input_dim=x.shape[1], activation='relu', kernel_initializer='random_normal'))

model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(25,activation='relu'))

model.add(Dense(1,activation='sigmoid'))





# compile the model

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=25, 

                        verbose=1, mode='min', restore_best_weights=True)



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
history = model.fit(X_train, y_train, validation_split=0.2, callbacks=[monitor], verbose=1, epochs=1000)



loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print('Accuracy: %f' % (accuracy*100))

print('\n')
plt.rcParams["figure.figsize"] = (11,5)



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model train vs validation loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
from sklearn.metrics import roc_curve, auc





# Plot an ROC. pred - the predictions, y - the expected output.

def plot_roc(pred,y):

    fpr, tpr, _ = roc_curve(y, pred)

    roc_auc = auc(fpr, tpr)



    plt.figure(figsize=(7,7))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC)')

    plt.legend(loc="lower right")

    plt.show()
prediction_proba = model.predict(X_test)
plot_roc(prediction_proba,y_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
randf = RandomForestClassifier(class_weight={0: 0.60, 1:0.4}, random_state=22, criterion="entropy")



randf.fit(X_train, y_train)



randf_prediction=randf.predict(X_test)



accuracy_score(y_pred = randf_prediction, y_true= y_test)
from sklearn.metrics import  plot_roc_curve

plot_roc_curve(randf, X_test, y_test)

plt.title("ROC for RF")

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report,confusion_matrix
dtree = DecisionTreeClassifier(class_weight={0: 0.60, 1:0.4},random_state=22, criterion='entropy')

dtree.fit(X_train,y_train)
tree_predictions = dtree.predict(X_test)
print(classification_report(y_test,tree_predictions))
pipe_randf=make_pipeline(StandardScaler(), randf)
pipe_dtree = make_pipeline(StandardScaler(), 

                           dtree)
CV_dtree=cross_validate(pipe_dtree,X_train,y_train,scoring=["accuracy","recall","precision"],

                      cv=StratifiedKFold(n_splits=5))
print("The mean accuracy in the Cross-Validation is: {:.2f}%".format((np.mean(CV_dtree["test_accuracy"])*100)))