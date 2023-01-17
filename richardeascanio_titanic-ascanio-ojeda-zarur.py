import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Ignorar warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 



from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import keras

from keras.layers import Dense, Dropout      # Crear capas

from keras.models import Input, Model



# Cargar la data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df = df_train.append(df_test , ignore_index = True)



# Inspecciones

df_train.shape, df_test.shape, df_train.columns.values
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
df['Pclass'].isnull().sum(axis=0)
df.Name.head(10)
df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



# Comparando la cantidad de personas por cada titulo

df['Title'].value_counts()
df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 

             & (df.Title !=  'Mrs')] = 'Others'



# Comparando Title y Survived

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Viendo la cantidad de personas que hay en cada Titulo

df['Title'].value_counts()
df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)
df.Sex.isnull().sum(axis=0)
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
df.Sex = df.Sex.map({'male':0, 'female':1})
df.Age.isnull().sum(axis=0)
df.SibSp.isnull().sum(axis=0), df.Parch.isnull().sum(axis=0)
# Para crear una Family se coloca la columna con los datos de los hermanos/esposos y padres/hijos de la persona

df['Family'] = df['SibSp'] + df['Parch'] + 1



# Comparamos Family y Survived 

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
df['Family'].value_counts()
df.Family = df.Family.map(lambda x: 0 if x > 4 else x)

df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
df['Family'].value_counts()
df.Ticket.isnull().sum(axis=0)
df.Ticket.head(20)
df.Ticket = df.Ticket.map(lambda x: x[0])



# Comparando Ticket y Survived

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df['Ticket'].value_counts()
df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()
df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()
df.Fare.isnull().sum(axis=0)
df.Ticket[df.Fare.isnull()]
df.Pclass[df.Fare.isnull()]
df.Cabin[df.Fare.isnull()]
df.Embarked[df.Fare.isnull()]
sns.boxplot('Pclass','Fare',data=df)

plt.ylim(0, 300) # ignorando la data cuya tarifa > 500

plt.show()
df[['Pclass', 'Fare']].groupby(['Pclass']).mean()
df[['Pclass', 'Fare']].groupby(['Pclass']).std() / df[['Pclass', 'Fare']].groupby(['Pclass']).mean()
sns.boxplot('Ticket','Fare',data=df)

plt.ylim(0, 300) # ignorar la data con la tarifa mayor a 500

plt.show()
df[['Ticket', 'Fare']].groupby(['Ticket']).mean()
df[['Ticket', 'Fare']].groupby(['Ticket']).std() /  df[['Ticket', 'Fare']].groupby(['Ticket']).mean()
sns.boxplot('Embarked','Fare',data=df)

plt.ylim(0, 300) # ignoramos la data cuya tarifa sea mayor que 500

plt.show()
df[['Embarked', 'Fare']].groupby(['Embarked']).mean()
df[['Embarked', 'Fare']].groupby(['Embarked']).std() /  df[['Embarked', 'Fare']].groupby(['Embarked']).mean()
guess_Fare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()

df.Fare.fillna(guess_Fare , inplace=True)



# Vemos los valores de la media de la Tarifa para las personas que murieron y sobrevivieron

df[['Fare', 'Survived']].groupby(['Survived'],as_index=False).mean()
grid = sns.FacetGrid(df, hue='Survived', height=4, aspect=1.5)

grid.map(plt.hist, 'Fare', alpha=.5, bins=range(0,210,10))

grid.add_legend()

plt.show()
df[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean().plot.scatter('Fare','Survived')

plt.show()
df['Fare-bin'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)



# Comparamos Fare-bin y Survived

df[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()
df.Cabin.isnull().sum(axis=0)
df = df.drop(labels=['Cabin'], axis=1)
df.Embarked.isnull().sum(axis=0)
df.describe(include=['O'])
df.Embarked.fillna('S' , inplace=True )
df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()
df = df.drop(labels='Embarked', axis=1)
grid = sns.FacetGrid(df, col='Title', height=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
df[['Title', 'Age']].groupby(['Title']).mean()
df[['Title', 'Age']].groupby(['Title']).std()
grid = sns.FacetGrid(df, col='Fare-bin', height=3, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
df[['Fare-bin', 'Age']].groupby(['Fare-bin']).mean()
df[['Fare-bin', 'Age']].groupby(['Fare-bin']).std()
grid = sns.FacetGrid(df, col='SibSp', col_wrap=4, height=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
df[['SibSp', 'Age']].groupby(['SibSp']).mean()
df[['SibSp', 'Age']].groupby(['SibSp']).std()
grid = sns.FacetGrid(df, col='Parch', col_wrap=4, height=3.0, aspect=0.8, sharey=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))

plt.show()
df[['Parch', 'Age']].groupby(['Parch']).mean()
df[['Parch', 'Age']].groupby(['Parch']).std()
df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','Fare-bin','SibSp']]



X_train  = df_sub.dropna().drop('Age', axis=1)

y_train  = df['Age'].dropna()

X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)



regressor = RandomForestRegressor(n_estimators = 300)

regressor.fit(X_train, y_train)

y_pred = np.round(regressor.predict(X_test),1)

df.Age.loc[df.Age.isnull()] = y_pred



df.Age.isnull().sum(axis=0) # ya no mas null
bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # Esto es algo arbitrario

age_index = (1,2,3,4,5,6,7) #('baby','child','teenager','young','mid-age','over-50','senior')

df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)



df[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df['Ticket'].value_counts()
df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')



# Comparamos nuevamente

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df = pd.get_dummies(df,columns=['Ticket'])
df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1)

y_train = df[0:891]['Survived'].values

X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values

X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values
precisiones_globales=[]

epochs = 200

def graf_model(train_history):

    f = plt.figure(figsize=(15,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()

def precision(model, registrar=False):

    y_pred = model.predict(X_train)

    train_auc = roc_auc_score(y_train, y_pred)

    y_pred = model.predict(val_dfX)

    val_auc = roc_auc_score(val_dfY, y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])
# Normalizamos los inputs

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print("Entrenamiento: ",X_train.shape)

print("Test : ",X_test.shape)
X_train,val_dfX,y_train, val_dfY = train_test_split(X_train,y_train , test_size=0.20, stratify=y_train)

print("Entrenamiento: ",X_train.shape)

print("Validacion : ",val_dfX.shape)
def func_model():   

    inp = Input(shape=(17,))

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(inp)

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model

model = func_model()

print(model.summary())
train_history = model.fit(X_train, y_train, batch_size = 32, epochs = epochs, validation_data=(val_dfX, val_dfY))
graf_model(train_history)

precision(model, True)
def func_model_reg():   

    inp = Input(shape=(17,))

    x=Dropout(0.1)(inp)

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(inp)

    x=Dropout(0.3)(x)

    x=Dense(8, activation="relu", kernel_initializer='glorot_normal')(x)

    x=Dropout(0.3)(x)

    x=Dense(1, activation="sigmoid", kernel_initializer='glorot_normal')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

    return model
modelR = func_model_reg()

print(modelR.summary())

finalTrain = modelR.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)
graf_model(finalTrain)

precision(modelR)
y_pred = model.predict(X_test)

y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})

output.to_csv('prediction.csv', index=False)