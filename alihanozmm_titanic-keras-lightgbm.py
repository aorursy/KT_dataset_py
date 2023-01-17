import numpy as np
import pandas as pd 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

import lightgbm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import classification_report, confusion_matrix

import random

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
read_directory = '/kaggle/input/titanic/'
train = pd.read_csv(read_directory+'train.csv')
X_test  = pd.read_csv(read_directory+'test.csv')
y_test  = pd.read_csv(read_directory+'gender_submission.csv')
PassengerID = y_test.PassengerId
test = pd.concat([X_test,y_test],axis=1,levels='PassangerId')


all_columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Survived','Name']

train = train.loc[:,all_columns]
test  = test.loc[:,all_columns]

train.loc[:,'Name'] = [x.split(',')[1].split('.')[0].replace(' ','') for x in train.Name] # Titles of the passangers
test.loc[:,'Name'] = [x.split(',')[1].split('.')[0].replace(' ','') for x in test.Name]

train.loc[:,'Cabin'] = [str(x)[0] for x in train[all_columns]['Cabin']] # Removing Numbers at the end of cabin types
test.loc[:,'Cabin']  = [str(x)[0] for x in test[all_columns]['Cabin']]

train.loc[:,'Sex'] = [0 if x == 'male' else 1 for x in train.Sex] # Label encoding of genders
test.loc[:,'Sex']  = [0 if x == 'male' else 1 for x in test.Sex]


train = pd.get_dummies(train, columns=['Cabin'])  # One Hot Encoding for Cabin column
test  = pd.get_dummies(test, columns=['Cabin'])

train = pd.get_dummies(train, columns=['Pclass'], drop_first='True') # One Hot Encoding for Pclass column
test  = pd.get_dummies(test, columns=['Pclass'], drop_first='True')

train.drop('Cabin_T',axis=1, inplace=True)

concat   = pd.concat([train,test])  # to use in further processes, creating concatted dataframe
columns  = concat.columns

enc = OneHotEncoder(drop='first')
enc_fit = enc.fit(concat[['Name']])  # One Hot Encoding for Name column
train = pd.concat([train,pd.DataFrame(enc_fit.transform(train[['Name']]).toarray(),columns=['Name_'+str(x) for x in range(1,18)])],axis=1)
test = pd.concat([test,pd.DataFrame(enc_fit.transform(test[['Name']]).toarray(),columns=['Name_'+str(x) for x in range(1,18)])],axis=1)

train.drop('Name',axis=1, inplace=True)
test.drop('Name',axis=1, inplace=True)

concat   = pd.concat([train,test]) # to use in further processes, creating concatted dataframe
columns  = concat.columns
imp = SimpleImputer()
imp_fit = imp.fit(concat)
train = pd.DataFrame(imp_fit.transform(train),columns=columns)
test  = pd.DataFrame(imp_fit.transform(test),columns=columns)
X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
X_test  = test.drop('Survived',axis=1)
y_test  = test['Survived']


concat = pd.concat([X_train,X_test])
columns  = concat.columns
std_scaler = StandardScaler()
scale_fit = std_scaler.fit(concat) 
X_train = pd.DataFrame(scale_fit.transform(X_train),columns=columns)
X_test  = pd.DataFrame(scale_fit.transform(X_test),columns=columns)
pd.DataFrame(y_train,columns=['Survived']).groupby('Survived').Survived.count()
oversample = SMOTE() 
X_train, y_train = oversample.fit_resample(X_train, y_train)
pd.DataFrame(y_train,columns=['Survived']).groupby('Survived').Survived.count()
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.1)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

input_dim = X_train.shape[1]
dropout = 0.2

# Initialising the ANN
classifier = Sequential()
classifier.add(Dropout(dropout, input_shape=(input_dim,)))
classifier.add(Dense(units = round(.5*input_dim), kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
classifier.add(Dropout(.8*dropout))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# Fitting the ANN to the Training set
progress = classifier.fit(X_train, y_train, batch_size = 32, epochs = 150,
               callbacks = [callback], validation_data=(X_val, y_val), verbose=0)

results = classifier.evaluate(X_test, y_test, batch_size=16)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, [round(x) for x in [item for sublist in y_pred for item in sublist]]))
file_name = "submission.csv"

y_pred_series = pd.Series([int(round(x)) for x in y_pred.flatten()], name = 'Survived')

file = pd.concat([PassengerID, y_pred_series], axis = 1)

file.to_csv(file_name, index = False)
epochs_range=range(len(progress.history['binary_accuracy']))
fig = plt.figure(figsize=(16,6))
 
def get_graphs():
    xs =[]
    ys =[]
    for i in range(10):
        xs.append(i)
        ys.append(random.randrange(10))
    return xs, ys
 
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
 
x, y = get_graphs()
ax1.plot(epochs_range, progress.history['binary_accuracy'])
ax1.plot(epochs_range, progress.history['val_binary_accuracy'])
ax1.set_ylabel('Accuracy',fontsize=19)
ax1.set_xlabel('Epochs',fontsize=19)
ax1.legend(('Train','Validation'),  loc='lower right', shadow=True, fontsize=12)
ax1.tick_params(axis ='both', which ='both', length = 0, labelsize=12)

x, y = get_graphs()
ax2.plot(epochs_range, progress.history['loss'])
ax2.plot(epochs_range, progress.history['val_loss'])
ax2.set_ylabel('Loss',fontsize=19)
ax2.set_xlabel('Epochs',fontsize=19)
ax2.legend(('Train','Validation'),  loc='upper right', shadow=True, fontsize=12)
ax2.tick_params(axis ='both', which ='both', length = 0, labelsize=12)
     
plt.show()
train_data = lightgbm.Dataset(X_train, label=y_train)
valid_data = lightgbm.Dataset(X_val, label=y_val)

params={
    'learning_rate':0.01,
    'objective':'binary',
    'boosting_type':'gbdt',
    'metric':'binary_logloss'
}

model = lightgbm.train(params,
                       train_data,
                       valid_sets=valid_data,
                       num_boost_round=1000,
                       verbose_eval=50,
                       early_stopping_rounds=20
                       )

predictions = [round(x) for x in model.predict(X_test,axis=1)]

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

metrics = pd.DataFrame([['Accuracy',accuracy_score(list(y_test), predictions)],
                        ['Precision:',precision_score(list(y_test), predictions, average="macro",zero_division=0)],
                        ['Recall:',recall_score(list(y_test), predictions, average="macro")],
                        ['F1 Score:',f1_score(list(y_test), predictions, average="macro")]],columns = ['Metric','Score'])

metrics
feature_importances = pd.DataFrame([model.feature_importance(),[x.split('_')[0] for x in X_train.columns]])
feature_importances = feature_importances.T
feature_importances.columns = ['Importance','Feature']

feature_importances = feature_importances.groupby('Feature')['Importance'].sum()
warnings.simplefilter(action='ignore', category=FutureWarning)

feature_imp = pd.DataFrame({'Value':feature_importances.values,'Feature':feature_importances.index})

feature_imp = feature_imp.groupby('Feature').sum().reset_index()
plt.figure(figsize=(40, 20))
sns.set(font_scale = 5)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                    ascending=False)[0:13])
plt.title('LightGBM Feature Importances')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png')
plt.show()