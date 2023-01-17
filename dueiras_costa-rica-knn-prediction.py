import pandas as pd
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

data_treino1 = pd.read_csv('../input/train.csv', sep=r'\s*,\s*',
        engine='python',
        na_values="?")

data_teste1 = pd.read_csv('../input/test.csv', sep=r'\s*,\s*',
        engine='python',
        na_values="?")

data_treino = data_treino1.drop('Id', axis=1)
#data_treino = data_treino.drop('edjefe', axis=1)
#data_treino = data_treino.drop('edjefa', axis=1)
data_treino = data_treino.drop('idhogar', axis=1)


data_teste = data_teste1.drop('Id', axis=1)
#data_teste = data_teste.drop('edjefe', axis=1)
#data_teste = data_teste.drop('edjefa', axis=1)
data_teste = data_teste.drop('idhogar', axis=1)
data_treino.head()
data_treino.shape
data_treino['v18q1'] = data_treino['v18q1'].fillna(0)
data_teste['v18q1'] = data_teste['v18q1'].fillna(0)
# If individual is over 19 or younger than 7 and missing years behind, set it to 0
data_treino.loc[((data_treino['age'] > 19) | (data_treino['age'] < 7)) & (data_treino['rez_esc'].isnull()), 'rez_esc'] = 0

data_teste.loc[((data_teste['age'] > 19) | (data_teste['age'] < 7)) & (data_teste['rez_esc'].isnull()), 'rez_esc'] = 0


"""data_treino.loc[(data_treino['edjefe'] == 'yes') ,'edjefe'] = int(1)
data_treino.loc[(data_treino['edjefe'] == 'no') ,'edjefe'] = int(0)
data_teste.loc[(data_teste['edjefe'] == 'yes') ,'edjefe'] = int(1)
data_teste.loc[(data_teste['edjefe'] == 'no') ,'edjefe'] = int(0) 
                
data_treino.loc[(data_treino['edjefa'] == 'yes') ,'edjefa'] = int(1)
data_treino.loc[(data_treino['edjefa'] == 'no') ,'edjefa'] = int(0)
data_teste.loc[(data_teste['edjefa'] == 'yes') ,'edjefa'] = int(1)
data_teste.loc[(data_teste['edjefa'] == 'no') ,'edjefa'] = int(0)"""

mapping = {"yes": 1, "no": 0}


for df in [data_treino, data_teste]:

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)


data_treino["edjefe"].value_counts().plot(kind="bar")
depend = []
for dependency, children, olds, total in zip(data_treino['dependency'], data_treino['hogar_nin'], data_treino['hogar_mayor'], data_treino['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

data_treino['dependency'] = depend

chw = []
for nin, adul in zip(data_treino['hogar_nin'], data_treino['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin / adul]

data_treino.shape
depend = []
for dependency, children, olds, total in zip(data_teste['dependency'], data_teste['hogar_nin'], data_teste['hogar_mayor'], data_teste['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

data_teste['dependency'] = depend

chw = []
for nin, adul in zip(data_teste['hogar_nin'], data_teste['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin / adul]
data_teste.head()
data_treino.shape
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    corr_matrix = data_treino.corr()
    display(corr_matrix['Target'].sort_values(ascending=False))
    

data_treino.loc[(data_treino['tipovivi1'] == 1), 'v2a1'] = 0
data_teste.loc[(data_teste['tipovivi1'] == 1), 'v2a1'] = 0
data_treino.loc[(data_treino['tipovivi5'] == 1), 'v2a1'] = 0
data_teste.loc[(data_teste['tipovivi5'] == 1), 'v2a1'] = 0
elec = []

for i, row in data_treino.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)
        
data_treino['elec'] = elec

data_treino = data_treino.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
elec = []

for i, row in data_teste.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)
        
data_teste['elec'] = elec

data_teste = data_teste.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
data_treino.info(verbose=True, null_counts=True)
data_treino = data_treino.dropna()
data_teste = data_teste.fillna(0)
xdata_treino = data_treino.drop("Target" ,axis=1)
Xtreino = xdata_treino[:]
Ytreino = data_treino.Target

y = []
x = range(8, 30)
for i in x:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, Xtreino, Ytreino, cv=15)
    y.append(scores.mean())
plt.scatter(x, y)

knn = KNeighborsClassifier(n_neighbors=28)
scores = cross_val_score(knn, Xtreino, Ytreino, cv=15)
display(scores)
scores.mean()

knn.fit(Xtreino,Ytreino)
Xteste = data_teste[:]
YtestePred = knn.predict(Xteste)
display(YtestePred)
arq = open ("prediction.csv", "w")
arq.write("id,Target\n")
for i, j in zip(data_teste1["Id"], YtestePred):
    arq.write(str(i)+ "," + str(j)+"\n")
arq.close()
