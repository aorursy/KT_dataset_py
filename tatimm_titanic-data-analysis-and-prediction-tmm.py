import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#Classificadores Scikit - https://www.it-swarm.dev/pt/python/lista-de-todos-os-algoritmos-de-classificacao/830155863/

from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import Perceptron

from sklearn.mixture import GaussianMixture

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier 

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier



#Métricas e pontuação de avaliação - https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

from sklearn.model_selection import cross_validate

from sklearn.feature_selection import SelectFromModel





classifiers_name = ["Random Forest", "QDA", "Gaussian Process", 

                    "Perceptron", "Gaussian Mixture", "Naive Bayes Multinomial", "Nearest Neighbors", "Neural Net",

                    "SVC", "Decision Tree"]

classifiers_type = [RandomForestClassifier(), QuadraticDiscriminantAnalysis(), GaussianProcessClassifier(), 

                    Perceptron(), GaussianMixture(), MultinomialNB(), KNeighborsClassifier(), MLPClassifier(),

                    SVC(), DecisionTreeClassifier()]



scoring_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']



#https://scikit-learn.org/stable/modules/feature_selection.html

importance_classifiers = [AdaBoostClassifier(), ExtraTreesClassifier(),RandomForestClassifier()]





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def base_dados_select():

    global dados_treino, dados_teste, df, dados_inicial, rotulos, nome_recursos_inicial

    dados_treino = pd.read_csv('/kaggle/input/titanic/train.csv')

    dados_teste = pd.read_csv('/kaggle/input/titanic/test.csv')

    df = pd.concat([dados_treino, dados_teste])

    dados_treino.rename(columns={'Survived':'target'}, inplace=True)

    dados_teste.rename(columns={'Survived':'target'}, inplace=True)

    df.rename(columns={'Survived':'target'}, inplace=True)

    df['target'].fillna(-1, inplace=True)

    

    # Convert objects in integer by quantity order

    dados_treino['Sex'] = dados_treino['Sex'].replace({'male', 'female'},{0, 1})

    labels = dados_treino['Ticket'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    dados_treino['Ticket'].replace(labels,codes,inplace=True)

    labels = dados_treino['Cabin'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    dados_treino['Cabin'].replace(labels,codes,inplace=True)

    labels = dados_treino['Embarked'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    dados_treino['Embarked'].replace(labels,codes,inplace=True) 

    dados_teste['Sex'] = dados_teste['Sex'].replace({'male', 'female'},{0, 1})

    labels = dados_teste['Ticket'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    dados_teste['Ticket'].replace(labels,codes,inplace=True)

    labels = dados_teste['Cabin'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    dados_teste['Cabin'].replace(labels,codes,inplace=True)

    labels = dados_teste['Embarked'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    dados_teste['Embarked'].replace(labels,codes,inplace=True) 

    df['Sex'] = df['Sex'].replace({'male', 'female'},{0, 1})

    labels = df['Ticket'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    df['Ticket'].replace(labels,codes,inplace=True)

    labels = df['Cabin'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    df['Cabin'].replace(labels,codes,inplace=True)

    labels = df['Embarked'].value_counts(ascending=True).index.tolist()

    codes = range(1,len(labels)+1)

    df['Embarked'].replace(labels,codes,inplace=True)

    

    # Convert float in integer with round up

    dados_treino['Age'] = dados_treino['Age'].apply(np.ceil)

    dados_teste['Age'] = dados_teste['Age'].apply(np.ceil)

    df['Age'] = df['Age'].apply(np.ceil)

    

    # Move target to the end

    df = df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','target']]

    dados_treino = dados_treino[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','target']]

    

    # Separate dataset in independent and dependent attributes

    dados_inicial = dados_treino.iloc[:, :-1].values

    rotulos = dados_treino.iloc[:, -1].values

    nome_recursos_inicial = dados_treino.iloc[:, :-1].columns.tolist()
def base_dados_analyse_simple(nome_recursos, uniq_val):

    # Print data descriptions to understand formats and if total quantity equals unique

    print(df.head())

    print(df.info())

    df_analyse = df.describe()

    df_analyse = df_analyse.transpose()

    df_analyse = df_analyse.assign(c_unique = df.nunique(dropna=False))

    print(df_analyse)



    # Print unique values: to check NaN values and errors

    if (uniq_val == 1):

        for col in df:

            print(col, sorted(df[col].unique()))

    

    # Check outliers

    nr = 0

    # Round Up

    len_recursos = len(nome_recursos)//2 + (len(nome_recursos) % 2 > 0)

    fig, ax = plt.subplots(len_recursos,2, figsize=(20,20))

    for i in range(len(ax)):

        for j in range(len(ax[i])):

            if (nr<len(nome_recursos)):

                ax[i][j] = sns.stripplot(data=df, y=nome_recursos[nr], jitter=True, ax=ax[i][j])

                nr += 1

    fig.suptitle('Checking outliers', position=(.5,1.1), fontsize=20)

    fig.tight_layout()

    fig.show()
def base_dados_analyse_advanced(dataset, nome_recursos, var_graph, size_x, size_y, var_subplot, var_stacked, dado_x , dado_y): 

    # Round Up

    len_recursos = len(nome_recursos)//2 + (len(nome_recursos) % 2 > 0)

    

    # Graphs

    if (var_graph == 'area'): ax = dataset.plot(kind='area', figsize=(size_x, size_y), subplots=var_subplot, stacked=var_stacked, title='Area Dados')

    elif (var_graph == 'line'): ax = dataset.plot(kind='line', figsize=(size_x, size_y), subplots=var_subplot, stacked=var_stacked, title='Linha Dados', x=dado_x, y=dado_y)

    elif (var_graph == 'bar'): ax = dataset.plot(kind='bar', figsize=(size_x, size_y), subplots=var_subplot, stacked=var_stacked, title='Barra Dados')

    elif (var_graph == 'barh'): ax = dataset.plot(kind='barh', figsize=(size_x, size_y), subplots=var_subplot, stacked=var_stacked, title='Barra Horizontal Dados')

    elif (var_graph == 'box'): ax = dataset.plot(kind='box', figsize=(size_x, size_y), subplots=var_subplot, title='Box Dados')

    elif (var_graph == 'hist'): ax = dataset.plot(kind='hist', figsize=(size_x, size_y), subplots=var_subplot, bins=15, alpha=0.5, title='Histograma Dados')

    elif (var_graph == 'kde'): ax = dataset.plot(kind='kde', figsize=(size_x, size_y), subplots=var_subplot, title='Densidade Dados', layout=(len_recursos,2))

    elif (var_graph == 'pie'): ax = dataset.plot(kind='pie', figsize=(size_x, size_y), subplots=var_subplot, title='Pizza Dados')

    elif (var_graph == 'hexbin'): ax = dataset.plot(kind='hexbin', figsize=(size_x, size_y), subplots=var_subplot, title='HexBin Dados', x=dado_x, y=dado_y)

    elif (var_graph == 'scatter'): ax = dataset.plot(kind='scatter', figsize=(size_x, size_y), subplots=var_subplot, title='Dispersão Dados', x=dado_x, y=dado_y)
def base_dados_treat():

    global df, dados_treino, dados_teste, dados_inicial, rotulos, nome_recursos_inicial

    #Delete Name column because of PassengerID and Cabin column for having most (1014/1309) of NaN values 

    dados_treino = dados_treino.drop(['Name', 'Cabin'], axis=1)

    dados_teste = dados_teste.drop(['Name', 'Cabin'], axis=1)

    df = df.drop(['Name', 'Cabin'], axis=1)

    

    # Check NaN and errors: Replaces NaN by mean

    dados_treino = dados_treino.fillna(dados_treino.mean())

    dados_teste = dados_teste.fillna(dados_teste.mean())

    df = df.fillna(df.mean())

    

    # Check outliers: Delete values

    dados_treino = dados_treino[(dados_treino.Age < 75)]

    dados_treino = dados_treino[(dados_treino.Fare < 300)]

    dados_treino = dados_treino[(dados_treino.Parch < 5)]

    dados_teste = dados_teste[(dados_teste.Age < 75)]

    dados_teste = dados_teste[(dados_teste.Fare < 300)]

    dados_teste = dados_teste[(dados_teste.Parch < 5)]

    df = df[(df.Age < 75)]

    df = df[(df.Fare < 300)]

    df = df[(df.Parch < 5)]

    

    # Re separate dataset in independent and dependent resources

    dados_inicial = dados_treino.iloc[:, :-1].values

    rotulos = dados_treino.iloc[:, -1].values

    nome_recursos_inicial = dados_treino.iloc[:, :-1].columns.tolist()
def classificadores( dados, rotulos, cross_value, var_predict ):

    global clf_name, avg_scores

    clf_name = classifiers_name

    avg_scores = []

    avg_scores_1 = []

    avg_scores_2 = []

    avg_scores_3 = []

    print('Number of metrics: ', len(scoring_list))

    print('Number of classifiers evaluated: ', len(classifiers_type))

    

    # Evaluation of all classifiers on the list

    for clf in classifiers_type:

        clf.fit(dados, rotulos)

        scores = cross_validate(clf, dados, rotulos, cv=4, scoring=scoring_list)

        for score in scoring_list:

            avg_scores.append(scores['test_'+score].mean())

            

    # Rank 3 first positions for classifier evaluation

    for j in range(0, 2*len(scoring_list), len(scoring_list)):

        for i in range(j, len(avg_scores)-len(scoring_list), len(scoring_list)):

            if (avg_scores[j] < avg_scores[i+len(scoring_list)]):

                if (i ==len(avg_scores)-2*len(scoring_list)):

                    avg_scores[:len(scoring_list)],avg_scores[-len(scoring_list):] = avg_scores[-len(scoring_list):],avg_scores[j:len(scoring_list)]

                    clf_name[:len(scoring_list)//len(scoring_list)],clf_name[-len(scoring_list)//len(scoring_list):] = clf_name[-len(scoring_list)//len(scoring_list):],clf_name[:len(scoring_list)//len(scoring_list)]

                else:

                    avg_scores[j:len(scoring_list)],avg_scores[(i+len(scoring_list)):(i+2*len(scoring_list))] = avg_scores[(i+len(scoring_list)):(i+2*len(scoring_list))],avg_scores[j:len(scoring_list)]

                    clf_name[j//len(scoring_list):len(scoring_list)//len(scoring_list)],clf_name[(i+len(scoring_list))//len(scoring_list):(i+2*len(scoring_list))//len(scoring_list)] = clf_name[(i+len(scoring_list))//len(scoring_list):(i+2*len(scoring_list))//len(scoring_list)],clf_name[j//len(scoring_list):len(scoring_list)//len(scoring_list)]

    avg_scores_1 = avg_scores[:len(scoring_list)]

    avg_scores_2 = avg_scores[len(scoring_list):(2*len(scoring_list))]

    avg_scores_3 = avg_scores[(2*len(scoring_list)):(3*len(scoring_list))]

    

    # Print classifiers ranking

    print('Classifiers ranking:')

    print(clf_name[0]+': ', avg_scores_1)

    print(clf_name[1]+': ', avg_scores_2)

    print(clf_name[2]+': ', avg_scores_3)

    

    # Comparative chart

    df_clf = pd.DataFrame({clf_name[0]: avg_scores_1,

                       clf_name[1]: avg_scores_2,

                       clf_name[2]: avg_scores_3}, index=scoring_list)

    ax = df_clf.plot(kind='bar' , rot=0, title='Compare classifiers')

    

    # Predict test data

    if (var_predict == 1):

        print('CLF_LOOKUP: ', clf_name[0])

        clf_index = classifiers_name.index(clf_name[0])

        print('CLF_INDEX: ', clf_index)

        clf = classifiers_type[clf_index]

        print('CLF: ', clf)

        clf.fit(dados, rotulos)

        test_predict = clf.predict(dados_teste)

        test_predict = test_predict.astype('int64')

        submission = pd.DataFrame()

        submission['PassengerId'] = dados_teste['PassengerId']

        submission['Survived'] = test_predict

        print(submission['Survived'].value_counts())

        submission.to_csv(r'Submission.csv', index = False, header = True)
def atributos( dados, rotulos, numero_imp, var_print, var_graph ):

    global dados_novo, nome_recursos_novo

    if (numero_imp >= 0):

        # Evaluation of the chosen importance classifier

        clf = importance_classifiers[numero_imp]

        clf.fit(dados, rotulos)

        importance = clf.feature_importances_

        print("Attributes importance: ", importance)

        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        indices = np.argsort(importance)[::-1]

        print("Attributes indices: ", indices)

        # Print attributes ranking

        if (var_print == 1):

            print("Attributes ranking:")

            for f in range(dados.shape[1]):

                print("%d. atributo %d (%f)" % (f + 1, indices[f], importance[indices[f]]))

                

        # Attributes importance graph

        if (var_graph == 1):

            plt.figure()

            plt.title("Attributes importance")

            plt.bar(range(dados.shape[1]), importance[indices], align="center")

            plt.xticks(range(dados.shape[1]), indices)

            plt.xlim([-1, dados.shape[1]])

            plt.show()

            

        # Select most important attributes

        model = SelectFromModel(clf, prefit=True)

        dados_novo = model.transform(dados)

        n_attrs = dados_novo.shape[1]

        idx_most_important = importance.argsort()[-n_attrs:]

        print(idx_most_important)

        nome_recursos_novo = np.array(nome_recursos_inicial)[idx_most_important]

        print(nome_recursos_novo)
base_dados_select()

base_dados_analyse_simple(nome_recursos_inicial, True)
base_dados_treat()

base_dados_analyse_simple(nome_recursos_inicial, False)
base_dados_analyse_advanced(df, nome_recursos_inicial, 'bar', 20, 20, True, False, None , None)
base_dados_analyse_advanced(df, nome_recursos_inicial, 'kde', 20, 20, True, False, None , None)
classificadores(dados_inicial, rotulos, 4, 0)
atributos(dados_inicial, rotulos, 1, 1, 1)
pwd
classificadores(dados_inicial, rotulos, 4, 1)