import pandas as pd
import sklearn
adult = pd.read_csv("../input/datapmr/train_data.csv",
        engine='python',
        na_values="?")
adult.shape
adult.head()
import matplotlib.pyplot as plt
adult['income'].value_counts().plot(kind='bar')
adult['occupation'].value_counts().plot(kind='bar')
adult['native.country'].value_counts().plot(kind='pie')
sexo = pd.crosstab(adult['income'], adult['sex'], margins = True, normalize = 'columns')
sexo
sexo.plot(kind='bar')
#marital.status
estado_civil = pd.crosstab(adult['income'], adult['marital.status'], margins = True,
                           normalize = 'columns')
estado_civil
#occupation
ocupacao = pd.crosstab(adult['income'], adult['occupation'], margins = True,
                           normalize = 'columns')
ocupacao
#relationship
relacao = pd.crosstab(adult['income'], adult['relationship'], margins = True,
                           normalize = 'columns')
relacao
#workclass
workclass = pd.crosstab(adult['income'], adult['workclass'], margins = True,
                           normalize = 'columns')
workclass
#race
raca = pd.crosstab(adult['income'], adult['race'], margins = True,
                           normalize = 'columns')
raca
adult['race'].value_counts().plot(kind='bar')
#education
educacao = pd.crosstab(adult['income'], adult['education'], margins = True,
                           normalize = 'columns')
educacao
#education
educacao = pd.crosstab(adult['income'], adult['education.num'], margins = True,
                           normalize = 'columns')
educacao
dado_n_num_sexo = ['Female', 'Male']
dado_num_sexo = [10.9, 30.5]

dado_n_num_estciv = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
              'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
dado_num_estciv = [10.4, 43.5, 44.7, 8.1, 4.6, 6.4, 8.6]

dado_n_num_ocup = ['Adm-clerical', 'Armed-forces', 'Craft-repair', 'Exec-managerial',
                     'Farming-fishing', 'Handlers-cleaner', 'Machine-op-inspct', 'Other-service',
                    'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support',
                    'Transport-moving']
dado_num_ocup = [13.4, 11.1, 22.7, 48.4, 11.6, 6.3, 12.5, 4.2, 0.7, 44.9, 32.5, 26.9, 30.5, 20.0]

dado_n_num_rela = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
dado_num_rela = [44.9, 10.3, 3.8, 1.3, 6.3, 47.5]

dado_n_num_work = ['Federal-gov','Local-gov', 'Never-worked','Private', 'Self-emp-inc',
                   'Self-emp-not-inc', 'State-gov', 'Without-pay']
dado_num_work = [38.6, 29.5, 0, 21.9, 55.7, 28.5, 27.2, 0]

dado_n_num_race = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
dado_num_race = [11.6, 26.6, 12.4, 9.2, 25.6]

dado_n_num_inc = ['>50K', '<=50K']
dado_num_inc = [0, 1]

dado_n_num =( dado_n_num_sexo + dado_n_num_estciv + dado_n_num_ocup + dado_n_num_rela 
             +dado_n_num_work+dado_n_num_race+dado_n_num_inc)
dado_num =( dado_num_sexo + dado_num_estciv + dado_num_ocup + dado_num_rela +dado_num_work
           +dado_num_race +dado_num_inc)
#Função para substituir os valores não numericos
def num_func(label):
    for i in range(len(dado_n_num)):
        if label == dado_n_num[i]:
            return dado_num[i]
    return label
adult['sex'] = adult['sex'].apply(num_func)
adult['marital.status'] = adult['marital.status'].apply(num_func)
adult['occupation'] = adult['occupation'].apply(num_func)
adult['relationship'] = adult['relationship'].apply(num_func)
adult['workclass'] = adult['workclass'].apply(num_func)
adult['race'] = adult['race'].apply(num_func)
#>50K e <=50k também são valores não numericos

adult['income'] = adult['income'].apply(num_func)
nadult = adult.dropna()
nadult
nadult.corr(method='pearson').income.sort_values(ascending=True)
testadult = pd.read_csv("../input/datapmr/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testadult['sex'] = testadult['sex'].apply(num_func)
testadult['marital.status'] = testadult['marital.status'].apply(num_func)
testadult['occupation'] = testadult['occupation'].apply(num_func)
testadult['relationship'] = testadult['relationship'].apply(num_func)
testadult['workclass'] = testadult['workclass'].apply(num_func)
testadult['race'] = testadult['race'].apply(num_func)
ntestadult = testadult.dropna()
Xadult = nadult[['relationship', 'marital.status', 'education.num', 'age','hours.per.week',
                  'capital.gain', 'sex', 'race', 'capital.loss']]
Yadult = nadult.income
Xtadult = testadult[['relationship', 'marital.status', 'education.num', 'age', 'hours.per.week',
                       'capital.gain', 'sex', 'race', 'capital.loss']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(Xtadult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(Yadult,knn.predict(Xadult))
resultado = pd.DataFrame(testadult.index)
resultado["income"] = YtestPred
#Lembrando da função para substituir os valores não numericos,
#podemos usar ela para o oposto
def subs(label):
    if label == 0:
        return "<=50K"
    else:
        return ">50K"
resultado["income"]=resultado["income"].apply(subs)
resultado.columns = ['Id', 'income']
resultado.to_csv('resultado.csv', index=False)
