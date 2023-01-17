# importar pacotes necessários

import numpy as np

import pandas as pd
# definir parâmetros extras

pd.set_option('precision', 4)

pd.set_option('display.max_columns', 100)



import warnings

warnings.filterwarnings("ignore")
prefixo_arquivos = '/kaggle/input/titanic/'
# carregar arquivo de dados de treino

train_data = pd.read_csv(prefixo_arquivos + 'train.csv', index_col='PassengerId')
# carregar arquivo de dados de teste

test_data = pd.read_csv(prefixo_arquivos + 'test.csv', index_col='PassengerId')
# unir ambos os dados de treino e teste

data = pd.concat([train_data, test_data])



# mostrar alguns exemplos de registros

data.head()
# extrair títulos das pessoas a partir do nome

data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



# exibir relação entre título e sexo

pd.crosstab(data['Title'], data['Sex']).T
# agregar títulos incomuns

replacements = {

    'Miss': ['Mlle', 'Ms'],

    'Mrs': ['Mme'],

    'Rare': ['Lady', 'Countess', 'Capt', 'Col', 'Don', \

             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

}

for k, v in replacements.items():

    data['Title'] = data['Title'].replace(v, k)

    

# exibir relação entre título e sexo

pd.crosstab(data['Title'], data['Sex']).T
# categorizar os valores dos títulos

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

data['Title'] = data['Title'].map(title_mapping)

data['Title'] = data['Title'].fillna(0)
# categorizar os valores dos sexos

data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
# preencher e categorizar os valores dos portos de embarque

data['Embarked'].fillna(data.Embarked.mode()[0], inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# preencher os valores da passagem

data['Fare'].fillna(data.Fare.mean(), inplace=True)
# criar coluna com tamanho da família

data['FSize'] = data['Parch'] + data['SibSp'] + 1
# criar coluna indicando se estava sozinho

data['Alone'] = 0

data.loc[data.FSize == 1, 'Alone'] = 1
# criar coluna contendo o deque

data['Deck'] = data['Cabin'].str[:1]

data['Deck'] = data['Deck'].fillna('N').astype('category')

data['Deck'] = data['Deck'].cat.codes
# criar coluna contendo o número do quarto

data['Room'] = data['Cabin'].str.extract("([0-9]+)", expand=False)

data['Room'] = data['Room'].fillna(0).astype(int)
data.head()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



def evaluate_regression_model(model, X, y):

    kfold = KFold(n_splits=10, random_state=42)

    results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', verbose=1)

    score = (-1) * results.mean()

    stddev = results.std()

    print(model, '\nScore: %.2f (+/- %.2f)' % (score, stddev))

    return score, stddev
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor



age_models = [

#    ('LR', LinearRegression(n_jobs=-1, fit_intercept=True, normalize=True)),

    ('GBR', GradientBoostingRegressor(random_state=42)),

    ('RFR', RandomForestRegressor(random_state=42)),

    ('XGB', XGBRegressor(random_state=42, objective='reg:squarederror')),

#    ('MLP', MLPRegressor(random_state=42, max_iter=500, activation='tanh',

#                         hidden_layer_sizes=(10,5,5), solver='lbfgs')),

#    ('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=True))

]
# selecionar dados para o treino



cols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Title', 'Age', 'Alone']



data_age = data[cols].dropna()



X_age = data_age.drop(['Age'], axis=1)

y_age = data_age['Age']



data_age.head()
data_age.corr()
names = []

scores = []

lowest = 999

best_model = None



for name, model in age_models:

    

    score, stddev = evaluate_regression_model(model, X_age, y_age)

    names.append(name)

    scores.append(score)

    

    if score < lowest:

        best_model = model

        lowest = score
results = pd.DataFrame({'Age Model': names, 'Score': scores})

results.sort_values(by='Score', ascending=True)
age_model = best_model

age_model.fit(X_age, y_age)
# preencher dados faltantes de idade a partir de uma regressão

data['AgePred'] = age_model.predict(data[cols].drop('Age', axis=1))

data.loc[data.Age.isnull(), 'Age'] = data['AgePred']

data.drop('AgePred', axis=1, inplace=True)

data.head()
# existem colunas com dados nulos?

data[data.columns[data.isnull().any()]].isnull().sum()
data.head()
# realizar normalização nos dados numéricos contínuos

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



cols = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'FSize']



data.loc[:,cols] = scaler.fit_transform(data.loc[:,cols])
data.head()
# importar os pacotes necessários para os algoritmos de classificação

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from datetime import datetime

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



# avalia o desempenho do modelo, retornando o valor da precisão

def evaluate_classification_model(model, X, y):

    start = datetime.now()

    kfold = KFold(n_splits=10, random_state=42)

    results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', verbose=1)

    end = datetime.now()

    elapsed = int((end - start).total_seconds() * 1000)

    score = 100.0 * results.mean()

    stddev = 100.0 * results.std()

    print(model, '\nScore: %.2f (+/- %.2f) [%5s ms]' % (score, stddev, elapsed))

    return score, stddev, elapsed
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold



# faz o ajuste fino do modelo, calculando os melhores hiperparâmetros

def fine_tune_model(model, params, X, y):

    print('\nFine Tuning Model:')

    print(model, "\nparams:", params)

    kfold = KFold(n_splits=10, random_state=42)

    grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=kfold, verbose=1)

    grid.fit(X, y)

    print('\nGrid Best Score: %.2f' % (grid.best_score_ * 100.0))

    print('Best Params:', grid.best_params_)

    return grid
# definir dados de treino

train_data = data[data.Survived.isnull() == False]



# selecionar atributos para o modelo

cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FSize', 'Alone', 'Deck']



X_train = train_data[cols]

y_train = train_data['Survived']



print('Forma dos dados de treino:', X_train.shape, y_train.shape)
train_data.corr()
# definir dados de teste

test_data = data[data.Survived.isnull()]



X_test = test_data[cols]



print('Forma dos dados de teste:', X_test.shape)
names = []

models = []

scores = []

stddevs = []

times = []



def add_model_info(name, model, score, stddev, elapsed):

    names.append(name)

    models.append((name, model))

    scores.append(score)

    stddevs.append(stddev)

    times.append(elapsed)
model = LogisticRegression(random_state=42, solver='newton-cg', C=0.1, multi_class='auto', max_iter=500)



params = dict(

    solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

    C=np.logspace(-3, 3, 7)

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('LR', model, score, stddev, elapsed)
model = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=6, min_samples_split=0.25)



#criterion=’mse’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

#min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 

#min_impurity_decrease=0.0, min_impurity_split=None, presort=False



params = dict(

    criterion=['gini','entropy'],

    max_depth=[4, 6, 8, 10, 12, 14],

    min_samples_split=[0.25, 0.5, 0.75, 1.0]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('DT', model, score, stddev, elapsed)
model = LinearDiscriminantAnalysis(solver='lsqr')



#solver=’svd’, shrinkage=None, priors=None,

#n_components=None, store_covariance=False, tol=0.0001



params = dict(

    solver=['svd', 'lsqr'] #, 'eigen']

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('LDA', model, score, stddev, elapsed)
model = GaussianNB(priors=None, var_smoothing=1e-8)



#priors=None, var_smoothing=1e-09



params = dict(

    priors=[None],

    var_smoothing=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('NB', model, score, stddev, elapsed)
model = KNeighborsClassifier(n_neighbors=11, weights='uniform')



#n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,

#metric_params=None, n_jobs=None



params = dict(

    n_neighbors=[1, 3, 5, 7, 9, 11, 13],

    weights=['uniform', 'distance']

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('KNN', model, score, stddev, elapsed)
model = SVC(random_state=42, C=10, gamma=0.1, kernel='rbf')



#kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001, C=1.0, 

#epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1



params = dict(

    C=[0.001, 0.01, 0.1, 1, 10, 100],

    gamma=[0.001, 0.01, 0.1, 1, 10, 100],

    kernel=['linear', 'rbf']

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('SVM', model, score, stddev, elapsed)
model = MLPClassifier(random_state=42, solver='lbfgs', alpha=1, hidden_layer_sizes=(100,), activation='logistic')

                

#hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, 

#learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 

#random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 

#early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10



params = dict(

    alpha=[1,0.1,0.01,0.001,0.0001,0],

    hidden_layer_sizes=[(100,), (50,), (50,2), (5,5,2), (10,5,2)],

    activation=['identity', 'logistic', 'tanh', 'relu'],

    solver=['lbfgs', 'sgd', 'adam']

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('MLP', model, score, stddev, elapsed)
model = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42, n_estimators=50)



#base_estimator=None, n_estimators=50, learning_rate=1.0,

#algorithm=’SAMME.R’, random_state=None



params = dict(

    n_estimators=[10, 25, 50, 100]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('ABDT', model, score, stddev, elapsed)
from sklearn.ensemble import BaggingClassifier



model = BaggingClassifier(random_state=42, n_estimators=100,

                          max_samples=0.25, max_features=0.8)



#base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0,

#bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,

#n_jobs=None, random_state=None, verbose=0



params = dict(

    n_estimators=[10, 50, 100, 500],

    max_samples=[0.25, 0.5, 0.75, 1.0],

    max_features=[0.7, 0.8, 0.9, 1.0]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('BC', model, score, stddev, elapsed)
model = ExtraTreesClassifier(random_state=42, n_estimators=100, max_depth=7, 

                             min_samples_split=0.25, max_features='auto')



#n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2,

#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, 

#max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 

#bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,

#warm_start=False, class_weight=None



params = dict(

    n_estimators=[10, 50, 100, 500],

    max_depth=[None, 3, 7, 11],

    min_samples_split=[0.25, 0.5],

    max_features=['auto', 0.7, 0.85, 1.0]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('ET', model, score, stddev, elapsed)
model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_features=0.75,

                                   max_depth=4, learning_rate=0.1, subsample=0.6)



#loss=’ls’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2,

#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 

#min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, 

#max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, 

#tol=0.0001



params = dict(

    n_estimators=[100, 250, 500],

    max_features=[0.75, 0.85, 1.0],

    max_depth=[4, 6, 8, 10],

    learning_rate=[0.05, 0.1, 0.15],

    subsample=[0.4, 0.6, 0.8]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('GB', model, score, stddev, elapsed)
model = RandomForestClassifier(random_state=42, n_estimators=100, max_features='auto', max_depth=5)



#n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 

#min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

#min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 

#verbose=0, warm_start=False



params = dict(

    n_estimators=[10, 50, 100, 500],

    max_features=['auto', 'sqrt', 'log2'],

    max_depth=[None, 3, 5, 7, 9, 11, 13]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('RF', model, score, stddev, elapsed)
model = XGBClassifier(max_depth=3, n_estimators=100)



#max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective='reg:squarederror',

#booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 

#colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 

#base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain'



params = dict(

    max_depth=[3, 5, 7, 9],

    n_estimators=[50, 75, 100, 200]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('XGB', model, score, stddev, elapsed)
estimators =  [

    ('RF', RandomForestClassifier(random_state=42, n_estimators=100, max_features='auto', max_depth=5)),

    ('BC', BaggingClassifier(random_state=42, n_estimators=100, max_samples=0.25, max_features=0.8)),

    ('GB', GradientBoostingClassifier(random_state=42, max_depth=4, max_features=0.75,

                                   n_estimators=100, learning_rate=0.1, subsample=0.6)),

#    ('XGB', XGBClassifier(max_depth=3, n_estimators=100)),

]

model = VotingClassifier(estimators, n_jobs=-1, weights=(2,1,1))



#estimators, weights=None, n_jobs=None



params = dict(

    weights=[(1,1,1), (2,1,1), (3,1,1), (3,2,1), (2,2,1), (2,1,2), (5,4,3), (1,2,1), (1,1,2), ]

)

#fine_tune_model(model, params, X_train, y_train)



score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)

add_model_info('VC', model, score, stddev, elapsed)
model = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=100)

model.fit(X_train, y_train)



importances = pd.DataFrame({'feature': X_train.columns,

                            'importance': np.round(model.feature_importances_, 3)})

importances = importances.sort_values('importance', ascending=False).set_index('feature')

importances.head(20)
importances.plot.bar()
results = pd.DataFrame({'Algorithm': names, 'Score': scores, 'Std Dev': stddevs, 'Time (ms)': times})

results.sort_values(by='Score', ascending=False)
# criar diretório para os arquivos de envio

#!test -d submissions || mkdir submissions
prefixo_arquivo = 'titanic-submission'

sufixo_arquivo = '06set'



for name, model in models:

    print(model, '\n')

    

    # treinar o modelo

    model.fit(X_train, y_train)

    

    # executar previsão usando o modelo

    y_pred = model.predict(X_test)

    y_pred_int = y_pred.astype(int)

    #vfunc = np.vectorize(lambda x: 'yes' if x > 0 else 'no')



    # gerar dados de envio (submissão)

    submission = pd.DataFrame({

      'PassengerId': X_test.index,

      'Survived': y_pred_int #vfunc(y_pred)

    })

    submission.set_index('PassengerId', inplace=True)



    # gerar arquivo CSV para o envio

    filename = '%s-p-%s-%s.csv' % (prefixo_arquivo, sufixo_arquivo, name.lower())

    submission.to_csv(filename)
!head titanic-*.csv