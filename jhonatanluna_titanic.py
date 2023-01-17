# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# train: https://raw.githubusercontent.com/qodatecnologia/titanic-dataset/master/train.csv
# test: https://raw.githubusercontent.com/qodatecnologia/titanic-dataset/master/test.csv
import pandas as pd

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

print(df_train.info())
# Bibliotecas
import pandas as pd 
import matplotlib 
import numpy as np 
import scipy as sp 
import IPython
from IPython import display 
import sklearn 
import random
import time
import warnings
warnings.filterwarnings('ignore')
# Modelos preditivos
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
# Auxiliares
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# DataViz
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# Dataviz Config
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
data_raw = pd.read_csv('../input/titanic/train.csv')
data_val  = pd.read_csv('../input/titanic/test.csv')

# criamos copia destes dados
data1 = data_raw.copy(deep = True)
# e concatenamos os datasets para altera-los juntos, caso necessário
data_cleaner = [data1, data_val]
print('Null values TREINO:\n', data1.isnull().sum())
print()
print('Null values TESTE:\n', data_val.isnull().sum())
# LIMPEZA DE DADOS
for dataset in data_cleaner:    
    #MEDIANA
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #MODA (em caso de uso da mediana, erro de tipo: str)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #MEDIANA
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
# dropar colunas para dataset de treino
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())
# FEATURE ENGINEERING: criação de features
for dataset in data_cleaner:    
    # nova feature: FamilySize
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    
    #nova feature: IsAlone
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    # nova feature: "Title"
    # Divisão rápida para "Name" com método .split()
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


    # nova feature: FareBin, "cortamos os dados em 4(quartil)"; Segmentamos/classificamos em 4 grupos
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #nova feature: AgeBin para melhor distribuir os dados de idade por grupos, neste caso, 5. Segmentamos/classificamos em 5 grupos
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

data1.sample(10)
# Limpar Title names raros(diferentes de Mr, Miss, Mrs, Master)
print(data1['Title'].value_counts())
title_names = (data1['Title'].value_counts() < 10) # Série de true/false do TitleName como index, quando aparece menos que 10x no dataset
title_names
# Função lambda para economizar linhas de código e substituir por "Misc" qualquer TitleName raro, com menos de 10 aparições
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
# Visualizar limpeza
data1.info()
print("====================================================================================")
data_val.info()
print("====================================================================================")
data1.sample(10)
# Converter

label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
# Definir variavel target
Target = ['Survived']
# Definir features ja alteradas
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] # Nomes para gráficos
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] # modelos preditivos
data1_xy =  Target + data1_x
data1_xy
# Variáveis a substituir, com LabelEncoder
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
data1_xy_bin
# Variáveis dummy
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
data1_dummy.head()
# Divisão TREINO/TESTE

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Dataset: {}".format(data1.shape))
print("Treino: {}".format(train1_x.shape))
print("Teste: {}".format(test1_x.shape))

train1_x_bin.head()
data1_x
# Correlação por sobrevivência
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Correlação sobrevivência:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        

#crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
print(pd.crosstab(data1['Title'],data1[Target[0]]))
data1[[x, Target[0]]]
data1[[x, Target[0]]].groupby(x, as_index=False).mean()
# Distribuição por idade: sobreviventes/não-sobreviventes
a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
a.add_legend()
# Comparativo Sexo/Classe/Idade
h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
# Heatmap de Correlação
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Heatmap Correlação Pearson', y=1.05, size=15)

correlation_heatmap(data1)
# Instanciar diversos algoritmos de CLASSIFICAÇÃO
MLA = [
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
# splitter http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 

# comparar métricas
MLA_columns = ['MLA Nome', 'MLA Parametros','Acurácia média TREINO', 'Acurácia média TESTE', 'TEMPO UTILIZADO(média)']
MLA_compare = pd.DataFrame(columns = MLA_columns)

# comparar predições
MLA_predict = data1[Target]

# salvamos performance na tabela
row_index = 0
for alg in MLA:

    # setamos nome do algoritmo e parametros
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Nome'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parametros'] = str(alg.get_params())
    
    # resultados com cross validation
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split, return_train_score=True)

    MLA_compare.loc[row_index, 'TEMPO UTILIZADO(média)'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'Acurácia média TREINO'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'Acurácia média TESTE'] = cv_results['test_score'].mean()     

    # salvamos as predições, algoritmo por algoritmo
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index+=1
MLA_compare.sort_values(by = ['Acurácia média TESTE'], ascending = False, inplace = True)
MLA_compare
MLA_predict
#barplot seaborn https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='Acurácia média TESTE', y = 'MLA Nome', data = MLA_compare, color = 'm')
#barplot seaborn https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='Acurácia média TESTE', y = 'MLA Nome', data = MLA_compare, color = 'm')

# pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Score por Algoritmos \n')
plt.xlabel('Accurácia(%)')
plt.ylabel('Algoritmo')
submit_xgb = XGBClassifier()
submit_xgb.fit(data1[data1_x_bin], data1[Target])
data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])
#submit file
submit = data_val[['PassengerId','Survived']]
submit.to_csv("submit.csv", index=False)
