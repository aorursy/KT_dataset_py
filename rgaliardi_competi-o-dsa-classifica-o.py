# Importando os módulos das bibliotecas de Data Science



import sys

import IPython 

from IPython import display



import numpy as np

import pandas as pd      



import sklearn as sk

import scipy as sp



import matplotlib as plt   

import seaborn as sns; sns.set(style="ticks", color_codes=True)



# Faz com que os relatórios (plots) sejam apresentados em uma célula e não em uma nova janela

%matplotlib inline       
# Checando as versões para acompanhamento de atualizações



print("Python version: {}". format(sys.version))

print("NumPy version: {}". format(np.__version__))

print("pandas version: {}". format(pd.__version__))

print("matplotlib version: {}". format(plt.__version__))

print("SciPy version: {}". format(sp.__version__)) 

print("scikit-learn version: {}". format(sk.__version__))

print("Seaborn version: {}". format(sns.__version__)) 

print("IPython version: {}". format(IPython.__version__)) 
# Prepara os dados - importando os datasets



dsTrain = pd.read_csv('../input/dataset_treino.csv', 

                      names=['id', 'num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade', 'classe'], 

                      sep=',', header=0)



dsTest  = pd.read_csv('../input/dataset_teste.csv',  

                      names=['id', 'num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade'], 

                      sep=',', header=0)



# Verifica a importação dos dados de treino

print(dsTrain.count())

print('\n')

# Verifica a importação dos dados de teste

print(dsTest.count())
# Verifica valores nulos



dsTrain.isnull().sum()
dsTrain.head()
dsTrain.info()
# Checando a variável Preditora

sns.countplot(x='classe',data=dsTrain)
# Checando as variáveis Independentes

columns = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

sns.pairplot(data=dsTrain[columns])
# Checando as varíaveis - Sumarizado Estatístico

dsTrain.describe()
# Criando a função para Correlação

import matplotlib.pyplot as plt   



def fnCcorrelation_heatmap(data):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        data.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )
# Análise de Correlação de Pearson das variáveis



columns = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

fnCcorrelation_heatmap(dsTrain[columns])

dsTrain[columns].corr().apply(lambda x: x.sort_values(ascending=False).values)
# Criando cópias dos datasets para manipulação e manter os datasets originais



mTrain = dsTrain.copy()

mTest = dsTest.copy()
# Definição das classes de variáveis aplicaveis aos modelos

columns = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']



# X_train - define os dados independentes de treino

X_train = mTrain[columns]



# y_train - define a variável preditora

y_train = mTrain['classe']



# X_test - define os dados independentes de teste

X_test = mTest[columns]



# y_test - define a variável preditora

y_test = mTrain['id']
# Standardization, or mean removal and variance scaling¶

from sklearn import preprocessing



X_scaled = preprocessing.scale(X_train)

X_scaled
# Verificação das variáveis independentes e sua classificação



from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from xgboost import XGBClassifier



#clf = XGBClassifier(n_estimators=50, max_features='sqrt')

#predictions = clf.fit(X_train, y_train)



clf = DecisionTreeClassifier()

classifier = clf.fit(X_train, y_train)



#clf = BernoulliNB()

#classifier = clf.fit(X_train, y_train)



features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = classifier.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)



features.plot(kind='barh', figsize=(25, 25))
# Função para classificação



import warnings

warnings.filterwarnings('ignore')



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Metrics

from sklearn.metrics import accuracy_score, log_loss



#Splits

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold



# Comparação dos classificadores

classifiers = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]



def fnClassifier(splits, target, features):

    cols = ["Classifier", "Accuracy"]

    acc_dict = {}

    log = pd.DataFrame(columns=cols)



    X = features.values

    y = target.values



    #_split = KFold(n_splits=splits, random_state=42, shuffle=True)

    #_split = model_selection.ShuffleSplit(n_splits = splits, test_size = .3, train_size = .6, random_state = 0 )

    _split = StratifiedShuffleSplit(n_splits= splits, test_size=0.1, random_state=0)



    for train_index, test_index in _split.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        for clf in classifiers:

            name = clf.__class__.__name__

            clf.fit(X_train, y_train)

            predictions = clf.predict(X_test)

            acc = accuracy_score(y_test, predictions)

            if name in acc_dict:

                acc_dict[name] += acc

            else:

                acc_dict[name] = acc

                

    for clf in acc_dict:

        acc_dict[clf] = acc_dict[clf] / 10.0

        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=cols)

        log = log.append(log_entry)

    

    # Plot Classifier Accuracy

    sns.set(style="darkgrid")

    sns.barplot(x='Accuracy', y='Classifier', data=log)

    

    return log.groupby(['Classifier', 'Accuracy']).count().sort_values(by=['Accuracy'], ascending=False)
# Classifier(splits, target, features)



classifier = fnClassifier(12, y_train, X_train)

print(classifier.iloc[0])

classifier
# Cria a matriz de Confusao: Confusion Matrix



from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn import datasets, linear_model, tree

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier





print('Confusion Matrix')

#lm = tree.DecisionTreeClassifier()

#lm = naive_bayes.BernoulliNB()

#lm = ensemble.RandomForestClassifier()

lm = discriminant_analysis.LinearDiscriminantAnalysis()



model = lm.fit(X_scaled, y_train)

predictions = lm.predict(X_test)



matrix = cross_val_predict(lm, X_scaled, y_train, cv=10)

pd.DataFrame(confusion_matrix(y_train, matrix), columns=['True', 'False'], index=['True', 'False'])
## output(output, test, name)



df_output = pd.DataFrame()

df_output['id'] = mTest['id'].astype(int)

df_output['classe'] = pd.DataFrame(predictions)



filename = '../output/submission.csv'

df_output[['id', 'classe']].to_csv(filename, sep=',', encoding='utf-8', index=False)  

print('ok - arquivo gerado: ' + filename)