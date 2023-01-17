import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pandas as pd

import statsmodels.api as sm

from statsmodels.nonparametric.kde import KDEUnivariate

from statsmodels.nonparametric import smoothers_lowess

from pandas import Series, DataFrame

from patsy import dmatrices

#from KaggleAux import predict as ka # see github.com/agconti/kaggleaux for more details



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# sklearn imports

from sklearn.preprocessing import StandardScaler        

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline



from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split



# Keras imports 

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras.utils import np_utils



# loading data and getting shape

df_treino = pd.read_csv("../input/titanic/train.csv") 

df_teste = pd.read_csv("../input/titanic/test.csv")

df_gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

print(df_treino.shape, df_teste.shape, df_gender_submission.shape)

# Verifying the NaN values

print(f'Treino \n{df_treino.isnull().sum()}')

print('\n')

print(f'Treino \n{df_teste.isnull().sum()}')
# Dropping 'Cabin' because have very NaN values

df_teste = df_teste.drop(['Cabin'], axis = 1)

df_treino = df_treino.drop(['Cabin'], axis = 1)

# Fitting missing values. Using mean of Ages

df_treino['Age'].fillna(df_treino['Age'].mean(), inplace = True)

df_teste['Age'].fillna(df_teste['Age'].mean(), inplace = True)

# Fill only one NaN 'Fare'. Using mean too.

df_teste['Fare'].fillna(df_teste['Fare'].mean(), inplace = True)

# Searching for outliers

sns.boxplot('Survived', 'Fare', data = df_treino)
# Cutting the outliers

df_treino = df_treino[df_treino['Fare']<400]

sns.boxplot('Survived', 'Fare', data = df_treino)
# Creating new columns for Family_size and Family group



df_treino['Family_size'] = df_treino['SibSp'] + df_treino['Parch'] + 1

df_teste['Family_size'] = df_teste['SibSp'] + df_teste['Parch'] + 1



def family_group(size):

    a = ''

    if size <= 1:

        a = 'loner'

    elif size < 4:

        a = 'small'

    else:

        a = 'large'

    return a



df_treino['Family_group'] = df_treino['Family_size'].map(family_group)

df_teste['Family_group'] = df_teste['Family_size'].map(family_group)
# Ploting "Family_group"

sns.barplot( df_treino['Family_group'].value_counts().index, df_treino['Family_group'].value_counts().values)
# Cast Sex and Family_group values to categorical

df_treino['Sex_code'] = pd.Categorical(df_treino.Sex).codes

df_teste['Sex_code'] = pd.Categorical(df_teste.Sex).codes

df_treino['Family_group_code'] = pd.Categorical(df_treino['Family_group']).codes

df_teste['Family_group_code'] = pd.Categorical(df_teste['Family_group']).codes
# Mapping 'Fare' in groups and creating 'Fare_group_code'

def fare_group(fare):

    a = ''

    if fare <=4:

        a = 'very_Low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    else:

        a = 'very_high'

    return a 



df_treino['Fare_group'] = df_treino['Fare'].map(fare_group)

df_teste['Fare_group'] = df_teste['Fare'].map(fare_group)

df_treino['Fare_group_code'] = pd.Categorical(df_treino['Fare_group']).codes

df_teste['Fare_group_code'] = pd.Categorical(df_teste['Fare_group']).codes
# Trying to get correlations between 'Survived' and Name's length

## It is crazy way !!!!

df_treino['len_names'] = df_treino['Name'].apply(len)

df_teste['len_names'] = df_teste['Name'].apply(len)
## Plotting len names

sns.distplot(df_treino['len_names'])
## Mapping the Age in groups and codding

def age_group(age):

    a = ''

    if age <= 5:

        a = 'infant'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'young_adult'

    elif age <= 65:

        a = 'adult'

    else:

        a = 'senior'

    return a 



df_treino['Age_group'] = df_treino['Age'].map(age_group)

df_teste['Age_group'] = df_teste['Age'].map(age_group)

df_treino['Age_group_code'] = pd.Categorical(df_treino['Age_group']).codes

df_teste['Age_group_code'] = pd.Categorical(df_teste['Age_group']).codes
# Dropping missing values in 'Embarked'

df_treino.dropna(inplace = True)

df_treino.info()



# creating a categorical variable with 'Embarked' values

df_treino['Embarked_code'] = pd.Categorical(df_treino['Embarked']).codes

df_teste['Embarked_code'] = pd.Categorical(df_teste['Embarked']).codes



# Specifies the parameters of graphs

fig = plt.figure(figsize=(12, 8))



# Lets plot differents graphs 

ax1 = plt.subplot2grid((2, 3), (0, 0))

# plots the bar graph

df_treino.Survived.value_counts().plot(kind = 'bar', alpha = 0.8)

# setting the range of 'x' values

ax1.set_xlim(-1, 2)

plt.title("Distribution of Survival, (1 = Survived)")



# another graph here

ax2 = plt.subplot2grid((2, 3), (0, 1))

plt.scatter(df_treino.Survived, df_treino.Age, alpha = 0.8)

plt.xlabel = "Age"

plt.grid(axis='y')

plt.title("Survival by Age,  (1 = Survived)")



# another graph here

ax3 = plt.subplot2grid((2, 3), (0, 2))

df_treino.Pclass.value_counts().plot(kind = 'barh', alpha = 0.8 )

plt.title('Class Distribution')

ax3.set_ylim(-1, len(df_treino.Pclass.value_counts()))

plt.grid()



# another graph here

ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)

df_treino[df_treino.Pclass == 1]['Age'].plot(kind = 'kde')

df_treino[df_treino.Pclass == 2]['Age'].plot(kind = 'kde')

df_treino[df_treino.Pclass == 3]['Age'].plot(kind = 'kde')

plt.title("Age Distribution within classes")

plt.legend(['1 Class', '2 Class', '3 Class'], loc  ='best')



# another graph here

ax2 = plt.subplot2grid((2, 3), (1, 2))

df_treino.Embarked.value_counts().plot(kind = 'bar', alpha = 0.8)

ax2.set_xlim(-1, 3)

plt.title("Passengers per boarding location")

fig = plt.figure(figsize=(18, 6))

df_male = df_treino[df_treino.Sex_code == 1]['Survived'].value_counts().sort_index()

df_female = df_treino[df_treino.Sex_code == 0]['Survived'].value_counts().sort_index()



ax1 = fig.add_subplot(121)

df_male.plot(kind = 'barh', label = 'male', alpha = 0.55)

df_female.plot(kind = 'barh', label = 'female', color = 'red', alpha = 0.55)

plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')

ax1.set_ylim(-1, 2)

plt.legend()



ax1 = fig.add_subplot(122)

(df_male/float(df_male.sum())).plot(kind='barh',label='Male', alpha=0.55)  

(df_female/float(df_female.sum())).plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)

plt.title("Who Survived proportionally? with respect to Gender"); plt.legend(loc='best')
# Let's see the correlation among variables

plt.figure(figsize=(12, 8))

sns.heatmap(df_treino.corr(), annot=True)
# values without scaler

X = df_treino.loc[:, ['Pclass', 'Sex_code', 'Age',  'Family_group_code', 'Fare_group_code', 'len_names', 'Embarked_code', 'Age_group_code']].values

y = df_treino.loc[:, 'Survived'].values

X_teste = df_teste.loc[:, ['Pclass', 'Sex_code', 'Age', 'Family_group_code', 'Fare_group_code', 'len_names', 'Embarked_code', 'Age_group_code']].values



# train test without scaler

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25)

print(len(X_train), len(X_valid), len(y_train), len(y_valid))





## with scaler

scaler = StandardScaler()

X_scaler = scaler.fit_transform(X)

X_teste_scaler = scaler.fit_transform(X_teste)

y_scaler = y



# train test with scaler

X_train_scaler, X_valid_scaler, y_train_scaler, y_valid_scaler = train_test_split(X_scaler, y, test_size = 0.25)

print(len(X_train_scaler), len(X_valid_scaler), len(y_train_scaler), len(y_valid_scaler))

# Random Forest with Kfold without scaler



resultado_random = []

fold = KFold(n_splits=5, shuffle=True)

for train_index, valid_index in fold.split(X):

    print("TRAIN:", len(train_index), "TEST:", len(valid_index))

    X_train, X_valid = X[train_index], X[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    modelo_random = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    modelo_random.fit(X_train, y_train)

    p = modelo_random.predict(X_valid)

    resultado_random.append(accuracy_score(y_valid, p))

print(f'Mean Random results without scaler {np.mean(resultado_random)}')

print(f'All results random {resultado_random}')





# Random Forest with Kfold and scaler

resultado_random = []

fold = KFold(n_splits=5, shuffle=True)

for train_index, valid_index in fold.split(X):

    print("TRAIN:", len(train_index), "TEST:", len(valid_index))

    X_train, X_valid = X_scaler[train_index], X_scaler[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    modelo_random_scaler = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    modelo_random_scaler.fit(X_train, y_train)

    p = modelo_random_scaler.predict(X_valid)

    resultado_random.append(accuracy_score(y_valid, p))

print(f'Mean random results with scaler {np.mean(resultado_random)}')

print(f'All results random {resultado_random}')

# Using GridSearchCV with and without scaler



variables = {'X': X, 'X_scaler': X_scaler}

results_random = {}

modelos = ['modelo_random', 'modelo_random_scaler']

for i, j in variables.items():

    parameters = {'n_estimators': [100, 500, 1000, 2000]}

    metrics = ['accuracy']

    for i in modelos:

        modelo = RandomForestClassifier(n_jobs=-1)

        grid_random = GridSearchCV(modelo, parameters, scoring = metrics, refit=False, cv = 10)

        results_random[i] = grid_random.fit(j, y)

    print(results_random)



results_random['modelo_random'].cv_results_
results_random['modelo_random_scaler'].cv_results_


# Neural Network now

# with kfold method with and without scaler

# Changing the activation param



resultado_geral = []

resultado_rede_neural = []



activation = ['identity', 'logistic', 'tanh', 'relu']

variables = {'X': X, 'X_scaler': X_scaler}

results_neural = {}

modelos = ['modelo_random', 'modelo_random_scaler']

for i, j in variables.items():

    for act in activation:

        fold = KFold(n_splits=4, shuffle=True)

        for train_index, valid_index in fold.split(X):

            #print("TRAIN:", train_index, "TEST:", valid_index)

            X_train, X_valid = j[train_index], j[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

            modelo_rede_neural = MLPClassifier(activation = act, max_iter = 5000)

            modelo_rede_neural.fit(X_train, y_train)

            p = modelo_rede_neural.predict(X_valid)

            results_neural[i] = accuracy_score(y_valid, p)

print(results_neural)

        
# Using GridSearchCV with and without scaler (neural network)

# Changing the activation and solver params



variables = {'X': X, 'X_scaler': X_scaler}

results_grid_neural = {}

modelos = ['modelo_neural', 'modelo_neural_scaler']

parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam']}

metrics = ['accuracy']

for i, j in variables.items():

    for mod in modelos:

        modelo = MLPClassifier(max_iter=5000)

        grid = GridSearchCV(modelo, parameters, scoring = metrics, refit=False, cv = 5)

        results_grid_neural[mod] = grid.fit(j, y)



print(results_grid_neural)

results_grid_neural['modelo_neural'].cv_results_


results_grid_neural['modelo_neural_scaler'].cv_results_
# Creating model to sbmit in Kaggle Competition - neural first

modelo_rede_neural = MLPClassifier(activation='logistic', solver='adam', max_iter = 5000)

modelo_rede_neural.fit(X, y)

p_neural = modelo_rede_neural.predict(X_teste)
# This is the format required

rede_neural_submission = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived': p_neural})
# Save to csv file

rede_neural_submission.to_csv('rede_neural_submission.csv', index = False)

rede_neural_submission.head()
# Creating model to sbmit in Kaggle Competition - now random

modelo_random = RandomForestClassifier(n_estimators=500, n_jobs=-1)

modelo_random.fit(X, y)

p_random = modelo_random.predict(X_teste)

random_submission = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived': p_random})

random_submission.to_csv('random_submission.csv', index = False)

random_submission.head()
random_submission['Survived'].sum(), rede_neural_submission['Survived'].sum()
# Classifier comparision

# wide aproach

classifiers = [

    KNeighborsClassifier(3),

    svm.SVC(probability=True),

    DecisionTreeClassifier(),

    XGBClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

    





log_cols = ["Classifier", "Accuracy"]

log= pd.DataFrame(columns=log_cols)
SSplit=StratifiedShuffleSplit(test_size=0.3,random_state=7)

acc_dict = {}



for train_index,test_index in SSplit.split(X,y, ):

    X_train,X_test=X[train_index],X[test_index]

    y_train,y_test=y[train_index],y[test_index]

    

    for clf in classifiers:

        name = clf.__class__.__name__

          

        clf.fit(X_train,y_train)

        predict=clf.predict(X_test)

        acc=accuracy_score(y_test,predict)

        if name in acc_dict:

            acc_dict[name]+=acc

        else:

            acc_dict[name]=acc
log['Classifier']=acc_dict.keys()

log['Accuracy']=acc_dict.values()

#log.set_index([[0,1,2,3,4,5,6,7,8,9]])

%matplotlib inline

sns.set_color_codes("muted")

ax=plt.subplots(figsize=(10,8))

ax=sns.barplot(y='Classifier',x='Accuracy',data=log,color='b')

ax.set_xlabel('Accuracy',fontsize=20)

plt.ylabel('Classifier',fontsize=20)

plt.grid(color='r', linestyle='-', linewidth=0.5)

plt.title('Classifier Accuracy',fontsize=20)
modelo_XGB = XGBClassifier(n_estimators=500)

modelo_XGB.fit(X, y)

p_XGB = modelo_XGB.predict(X_teste)

XGB_submission = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived': p_XGB})

XGB_submission.to_csv('XGB_submission.csv', index = False)
modelo_gradient = GradientBoostingClassifier()

modelo_gradient.fit(X, y)

p_gradient = modelo_gradient.predict(X_teste)

gradient_submission = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived': p_gradient})

gradient_submission.to_csv('gradient_submission.csv', index = False)
XGB_submission['Survived'].sum(), gradient_submission['Survived'].sum()
# working with keras



modelo_keras = Sequential()

modelo_keras.add(Dense(units = 5, activation = 'relu', input_dim = 8))

modelo_keras.add(Dense(units = 5, activation = 'relu'))

modelo_keras.add(Dense(units = 1, activation = 'sigmoid'))

modelo_keras.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Calculation precision in trainnig dataframe

modelo_keras.fit(X_train, y_train, batch_size=10, epochs = 100)

p_keras = modelo_keras.predict(X_valid)

p_keras = (p_keras > 0.75)

print(p_keras.sum())

precisao = accuracy_score(y_valid, p_keras)
modelo_keras.fit(X, y, batch_size=10, epochs = 100)

p_keras_real = modelo_keras.predict(X_teste_scaler)

p_keras_real = [1 if i > 0.75 else 0 for i in p_keras_real]
keras_submission = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived': p_keras_real})

keras_submission.to_csv('keras_submission_simple.csv', index = False)
# Keras with all parameters



modelo_keras = Sequential()

modelo_keras.add(Dense(units = 5, activation = 'relu', input_dim = 8))

modelo_keras.add(Dense(units = 5, activation = 'relu'))

modelo_keras.add(Dense(units = 1, activation = 'sigmoid'))



optimizers = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

losses = ['huber_loss',  'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']



resultados = {}

for opt in optimizers:

    for loss in losses:

        modelo_keras.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])

        modelo_keras.fit(X_train, y_train, batch_size=10, epochs = 100)

        p_best_keras = modelo_keras.predict(X_valid)

        p_best_keras = (p_best_keras > 0.55)

        precisao_cross = accuracy_score(y_valid, p_best_keras)

        resultados[opt, loss] = precisao_cross

print(resultados)





resultados
# applying keras without scaler



modelo_teste_valores = {}

for i in np.arange(0.4, 0.9, 0.05):

    modelo_keras.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

    modelo_keras.fit(X_train, y_train, batch_size=10, epochs = 100)

    p_best_keras = modelo_keras.predict(X_valid)

    p_best_keras = (p_best_keras > i)

    previsao = accuracy_score(y_valid, p_best_keras)

    modelo_teste_valores[i] = previsao

print(modelo_teste_valores)

plt.plot(modelo_teste_valores.keys(), modelo_teste_valores.values())
modelo_keras.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

modelo_keras.fit(X, y, batch_size=10, epochs = 100)

p_best_keras = modelo_keras.predict(X_teste)

p_best_keras = (p_best_keras > 0.55)
p_best_keras = [1 if i == True else 0 for i in p_best_keras]

np.sum(p_best_keras)
keras_submission_best = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived': p_best_keras})
keras_submission_best.to_csv('keras_best.csv', index = False)
### Working with pipeline

steps = [('scaler', StandardScaler()), ('GBC', GradientBoostingClassifier())]

pipeline = Pipeline(steps)
parameters = {'GBC__max_depth':[1, 2, 3, 4], 'GBC__learning_rate': [0.01, 0.001, 0.0001], 'GBC__n_estimators': [100, 200, 300] }
grid = GridSearchCV(pipeline, parameters, cv = 5)
grid.fit(X_train, y_train)
grid.score(X_train, y_train)
grid.best_params_
best = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=300)
best.fit(X, y)
grid_predict = best.predict(X_teste)
grid_predict_series = pd.DataFrame({'PassengerId': df_teste['PassengerId'], 'Survived':grid_predict})
grid_predict_series.to_csv('grid_predict_GBC_best.csv', index = False)