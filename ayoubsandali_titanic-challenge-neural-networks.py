import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np





data_full=pd.read_csv('../input/titanic/train.csv')





#########Data visualisation



print('Survived per sex\n')

print(data_full[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())





plt.figure(figsize=(6,6))

sns.barplot(x=data_full['Sex'], y=data_full['Survived'])

plt.title('Sex vs Survived')













print('Survived per class\n')

print (data_full[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())



plt.figure(figsize=(6,6))

sns.barplot(x=data_full['Pclass'], y=data_full['Survived'])

plt.title('Class vs Survived')



####Parch vs Survived



plt.figure(figsize=(6,6))

sns.barplot(x=data_full['Parch'], y=data_full['Survived'])

plt.title('Parch vs Survived')
#### SibSp vs Survived

plt.figure(figsize=(6,6))

sns.barplot(x=data_full['SibSp'], y=data_full['Survived'])

plt.title('SibSp vs Survived')
bins=[0,5,12,18,30,60,np.inf]

labels=['Baby','Child','Teenager','Adult(young)','Middle age','Old age']

data_full['AgeGroup'] = pd.cut(data_full["Age"], bins, labels = labels)



plt.figure(figsize=(10,6))

sns.barplot(x=data_full['AgeGroup'], y=data_full['Survived'])

plt.title('Age vs Survived')



print(data_full['AgeGroup'].dtype)
data_full.isna().sum()
import math



y=data_full.Survived

X=data_full.drop('Survived',axis=1)

X=X.drop('Name',axis=1)

X=X.drop('PassengerId',axis=1)

X=X.drop('Ticket',axis=1)

X=X.drop('Cabin',axis=1)

X['SibSp']=X['SibSp'].fillna(0)

X['Parch']=X['Parch'].fillna(0)

X['Embarked']=X['Embarked'].fillna(X['Embarked'].value_counts().idxmax())

X['Pclass']=X['Pclass'].fillna(X['Pclass'].value_counts().idxmax())



num=math.modf(X['Age'].value_counts().mean())[0]

if num>5:

    age=round(X['Age'].value_counts().mean(),0)

else:

    age=round(X['Age'].value_counts().mean(),0)+0.5

X['Age']=X['Age'].fillna(age)



X['AgeGroup'] = pd.cut(X["Age"], bins, labels = labels)

X=X.drop('Age',axis=1)



X_na = (X.isnull().sum() / len(X)) * 100

X_na = X_na.drop(X_na[X_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :X_na})

X.head()









from sklearn.preprocessing import LabelEncoder





categoricals_values=[col for col in X.columns if X[col].dtype=="object" or X[col].dtype.name=="category"  and X[col].nunique()< 10]

print(categoricals_values)

label_X_train = X.copy()

label_encoder = LabelEncoder()

for col in categoricals_values:

    label_X_train[col] = label_encoder.fit_transform(X[col])

    

label_X_train.shape





import tensorflow 

from tensorflow import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Dropout

from numpy.random import seed

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier







def create_model(lyrs=[12,6], act='linear', opt='Adam', dr=0.0):

    

    # set random seed for reproducibility

    seed(42)

    tensorflow.random.set_seed(42)

    

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim=label_X_train.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))  # output layer

    

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    

    return model

model = create_model()

print(model.summary())
training_1 = model.fit(label_X_train, y, epochs=100,validation_split=0.2, batch_size=16, verbose=0)

val_acc = np.mean(training_1.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
plt.plot(training_1.history['accuracy'])

plt.plot(training_1.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model = KerasClassifier(build_fn=create_model, verbose=0)



# define the grid search parameters

batch_size = [16, 32, 64]

epochs = [50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)



# search the grid

grid = GridSearchCV(estimator=model, 

                    param_grid=param_grid,

                    cv=3,

                    verbose=2)  



grid_result = grid.fit(label_X_train, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# create model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=16, verbose=0)



# define the grid search parameters

optimizer = [ 'RMSprop', 'Adagrad', 'Adam']

param_grid = dict(opt=optimizer)



# search the grid

grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

grid_result = grid.fit(label_X_train, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
seed(42)

tensorflow.random.set_seed(42)



# create model

model = KerasClassifier(build_fn=create_model, 

                        epochs=100, batch_size=16, verbose=0)



# define the grid search parameters

layers = [(8),(10),(10,5),(12,6),(12,8,4)]

param_grid = dict(lyrs=layers)



# search the grid

grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

grid_result = grid.fit(label_X_train, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=16, verbose=0)



# define the grid search parameters

drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

param_grid = dict(dr=drops)

grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)

grid_result = grid.fit(label_X_train, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
model = create_model()



print(model.summary())
training = model.fit(label_X_train, y, epochs=100, batch_size=16, 

                     validation_split=0.2, verbose=0)



# evaluate the model

scores = model.evaluate(label_X_train, y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
data_full_test=pd.read_csv('../input/titanic/test.csv')

X_test=data_full_test.copy()

X_test=X_test.drop('Name',axis=1)

X_test=X_test.drop('Ticket',axis=1)

X_test=X_test.drop('Cabin',axis=1)

X_test=X_test.drop('PassengerId',axis=1)

X_test['SibSp']=X_test['SibSp'].fillna(0)

X_test['Parch']=X_test['Parch'].fillna(0)

X_test['Embarked']=X_test['Embarked'].fillna(X_test['Embarked'].value_counts().idxmax())

X_test['Pclass']=X_test['Pclass'].fillna(X_test['Pclass'].value_counts().idxmax())



num=math.modf(X_test['Age'].value_counts().mean())[0]

if num>5:

    age=round(X_test['Age'].value_counts().mean(),0)

else:

    age=round(X_test['Age'].value_counts().mean(),0)+0.5

X_test['Age']=X_test['Age'].fillna(age)



X_test['AgeGroup'] = pd.cut(X_test["Age"], bins, labels = labels)

X_test=X_test.drop('Age',axis=1)



X_test.loc[lambda X_test: X_test['Pclass'] == 1]=X_test.loc[lambda X_test: X_test['Pclass'] == 1].fillna(X_test.loc[lambda X_test: X_test['Pclass'] == 1].mean()) 

X_test.loc[lambda X_test: X_test['Pclass'] == 2]=X_test.loc[lambda X_test: X_test['Pclass'] == 2].fillna(X_test.loc[lambda X_test: X_test['Pclass'] == 2].mean())

X_test.loc[lambda X_test: X_test['Pclass'] == 3]=X_test.loc[lambda X_test: X_test['Pclass'] == 3].fillna(X_test.loc[lambda X_test: X_test['Pclass'] == 3].mean())



X_na_test = (X_test.isnull().sum() / len(X)) * 100

X_na_test = X_na_test.drop(X_na[X_na == 0].index).sort_values(ascending=False)[:30]

missing_data_test = pd.DataFrame({'Missing Ratio' :X_na_test})





label_encoder = LabelEncoder()

for col in categoricals_values:

    X_test[col] = label_encoder.fit_transform(X_test[col])

X_test.head()
preds=model.predict(X_test)

preds=pd.DataFrame(data=np.round(preds,0),columns=['pred']).astype('int64')

output = pd.DataFrame({'PassengerId': data_full_test.PassengerId,

'Survived': preds.pred})

output.to_csv('submission.csv', index=False)




