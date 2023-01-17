import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
train_d = pd.read_csv('../input/train.csv', index_col='Id')
test_d = pd.read_csv('../input/test.csv', index_col='Id')
train_d.shape
train_d.head()
train_d.info()
test_d.info()
test_d.select_dtypes('object').head()
display(train_d.isnull().sum()[train_d.isnull().sum() > 0])
simnao = {"yes":1,"no":0}
test_d['Target'] = np.nan
data = train_d.append(test_d)
data['dependency'] = data['dependency'].replace(simnao).astype(np.float64)
data['edjefa'] = data['edjefa'].replace(simnao).astype(np.float64)
data['edjefe'] = data['edjefe'].replace(simnao).astype(np.float64)
data['v2a1'].fillna(0, inplace=True)
data.loc[(data['v18q'] == 0), 'v18q1'] = 0
data = data.drop('v18q',axis=1)
data['rez_esc'].fillna(0, inplace=True)
data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5
data.loc[data['meaneduc'].isnull(), 'meaneduc'] = data.loc[data['meaneduc'].isnull(), 'escolari']
data = data.drop('SQBmeaned',axis=1)
fit_data = data.loc[data['Target'].notnull()].drop('idhogar',axis=1).copy()
fit_data = fit_data.dropna()
fit_data.shape
knn = KNeighborsClassifier(n_neighbors=16)
scores = cross_val_score(knn,
                         fit_data.loc[:,:'Target'],
                         fit_data.loc[:,'Target'],
                         cv=10)
display(scores)
display(scores.mean())
train_d['Target'].value_counts(normalize=True).plot(kind='bar')
correlacao = fit_data.corr()

corr_superior = correlacao.where(np.triu(np.ones(correlacao.shape), k=1).astype(np.bool))

drop = [column for column in corr_superior.columns if any(abs(corr_superior[column]) > 0.9)]

drop
data = data.drop(columns = ['SQBescolari', 'SQBage',
            'SQBhogar_total', 'SQBedjefe',
            'SQBovercrowding', 'SQBdependency','SQBhogar_nin','agesq'])
data = data.drop('male',axis=1)
correlacao.loc[correlacao['tamhog'].abs() > 0.9, correlacao['tamhog'].abs() > 0.9]
data = data.drop(columns = ['r4t3','tamhog',
 'hogar_total'], axis=1)
correlacao.loc[correlacao['coopele'].abs() > 0.9, correlacao['coopele'].abs() > 0.9]
elecc = ['noelec', 'coopele', 'public', 'planpri']

elecvalues=pd.Series([fit_data['noelec'].sum(),fit_data['coopele'].sum(),fit_data['public'].sum(),fit_data['planpri'].sum()])
display(elecvalues.sum())

elecc[elecvalues.idxmax()]
elec = []

for i, row in data.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(2)

data['elec'] = elec

data = data.drop(columns = elecc)
data = data.drop('area2', axis=1)
n=0
for v in list(zip(fit_data['mobilephone'],fit_data['qmobilephone'])):
    if v[0]==0 and v[1]!=0: n+=1
n
data = data.drop('mobilephone', axis=1)
aguac = ['abastaguano', 'abastaguafuera', 'abastaguadentro']

n=pd.Series([fit_data[c].sum() for c in aguac])

display(n.sum())
aguac[n.idxmax()]
agua = []
for i, row in data.iterrows():
    if row['abastaguano'] == 1:
        agua.append(0)
    elif row['abastaguafuera'] == 1:
        agua.append(1)
    elif row['abastaguadentro'] == 1:
        agua.append(2)
    else:
        agua.append(2)

data['agua'] = agua

data = data.drop(columns = aguac)
instc = ['instlevel1','instlevel2','instlevel3',
        'instlevel4','instlevel5','instlevel6',
        'instlevel7','instlevel8','instlevel9']

n=pd.Series([fit_data[c].sum() for c in instc])

display(n.sum())
n.idxmax()
instlevel = []

for i, row in data.iterrows():
    if row['instlevel1'] == 1:
        instlevel.append(0)
    elif row['instlevel2'] == 1:
        instlevel.append(1)
    elif row['instlevel3'] == 1:
        instlevel.append(2)
    elif row['instlevel4'] == 1:
        instlevel.append(3)
    elif row['instlevel5'] == 1:
        instlevel.append(4)
    elif row['instlevel6'] == 1:
        instlevel.append(5)
    elif row['instlevel7'] == 1:
        instlevel.append(6)
    elif row['instlevel8'] == 1:
        instlevel.append(7)
    elif row['instlevel9'] == 1:
        instlevel.append(8)
    else:
        instlevel.append(2)

data['instlevel'] = instlevel

data = data.drop(columns = instc)
elimbasuc = ['elimbasu1','elimbasu2','elimbasu3',
             'elimbasu4','elimbasu5','elimbasu6']

n=pd.Series([fit_data[c].sum() for c in elimbasuc])

display(n.sum())
elimbasuc[n.idxmax()]
elimbasu = []

for i, row in data.iterrows():
    j=0
    for c in elimbasuc:
        if row[c] == 1:
            elimbasu.append(j)
        j+=1
        
data['elimbasu'] = elimbasu

data = data.drop(columns = elimbasuc)
ecasac = ['epared1', 'epared2', 'epared3', 
          'etecho1', 'etecho2', 'etecho3',
          'eviv1', 'eviv2', 'eviv3']
n=0
for c in ecasac:
    n+=fit_data[c].sum()
n/3
ecasa = []

for i, row in data.iterrows():
    j=1
    nota = 0
    for c in ecasac[:3]:
        if row[c] == 1:
            nota+=j
        j+=1
    j=1
    for c in ecasac[3:6]:
        if row[c] == 1:
            nota+=j
        j+=1
    j=1
    for c in ecasac[6:]:
        if row[c] == 1:
            nota+=j
        j+=1
    ecasa.append(nota)
    
data['ecasa'] = ecasa

data = data.drop(columns = ecasac)
nfamilia = []
for i, row in data.iterrows():
    nfamilia.append(row['tamviv']-row['hhsize'])
data['nfamilia'] = nfamilia
data = data.drop(columns = ['tamviv'])
tipovivic = ['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']

n=pd.Series([fit_data[c].sum() for c in tipovivic])

display(n.sum())
tipovivic[n.idxmax()]
tipovivi = []

for i, row in data.iterrows():
    if row['tipovivi1'] == 1:
        tipovivi.append(0)
    elif row['tipovivi2'] == 1:
        tipovivi.append(1)
    elif row['tipovivi3'] == 1:
        tipovivi.append(2)
    elif row['tipovivi4'] == 1:
        tipovivi.append(3)
    elif row['tipovivi5'] == 1:
        tipovivi.append(4)
    else:
        tipovivi.append(0)
                            
data['tipovivi'] = tipovivi
data = data.drop(columns=tipovivic)
insalubrec = ['pisonotiene','sanitario1','energcocinar1','refrig','cielorazo']
insalubre = []

for i, row in data.iterrows():
    nota = 0
    for c in insalubrec[:3]:
        if row[c] == 1:
            nota+=1
    for c in insalubrec[3:]:
        if row[c] == 0:
            nota+=1
    insalubre.append(nota)
                            
data['insalubre'] = insalubre
data = data.drop(columns=insalubrec)
correlacao = data.dropna().corr()

corr_superior = correlacao.where(np.triu(np.ones(correlacao.shape), k=1).astype(np.bool))

drop = [column for column in corr_superior.columns if any(abs(corr_superior[column]) > 0.9)]

drop
correlacao.loc[correlacao['instlevel'].abs() > 0.9, correlacao['instlevel'].abs() > 0.9]
correlacao.loc[correlacao['sanitario3'].abs() > 0.9, correlacao['sanitario3'].abs() > 0.9]
correlacao.loc[correlacao['energcocinar3'].abs() > 0.9, correlacao['energcocinar3'].abs() > 0.9]
fit_drop = ['instlevel','sanitario2','sanitario3',
            'sanitario5','sanitario6','lugar1',
            'lugar2','lugar3','lugar4','lugar5',
            'lugar6','parentesco2','parentesco3',
            'parentesco4','parentesco5','parentesco6',
            'parentesco7','parentesco8', 'parentesco9',
            'parentesco10','parentesco11','parentesco12',
            'paredblolad','paredzocalo','paredpreb',
            'paredmad','paredzinc','paredfibras',
            'paredother','pisomoscer','pisocemento',
            'pisoother','pisonatur','pisomadera',
            'techozinc','techoentrepiso','techocane',
            'techootro','energcocinar2','energcocinar3',
            'energcocinar4','estadocivil2','estadocivil1',
            'estadocivil3','estadocivil4','estadocivil5',
            'estadocivil6', 'estadocivil7','r4h1','r4h2',
            'r4m1','r4m2','r4t2','idhogar','Target']

fit_set1 = data.drop(columns=fit_drop).copy()
fit_set1['Target'] = data['Target'].copy()

fit_set1 = fit_set1.dropna()
fit_set1.info()
fit_set1.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
plt.xlabel('Number of Unique Values'); plt.ylabel('Count');
plt.title('Count of Unique Values in Integer Columns');
knn = KNeighborsClassifier(n_neighbors=15)
scores = cross_val_score(knn,
                         fit_set1.loc[:,:'Target'],
                         fit_set1.loc[:,'Target'],
                         cv=10)
display(scores)
display(scores.mean())
scores_array = []
for n in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn,
                             fit_set1.loc[:,:'Target'],
                             fit_set1.loc[:,'Target'],
                             cv=10)
    scores_array.append(scores.mean())
    
plt.plot(range(1,30),scores_array, 'ro')
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
cv_score = cross_val_score(knn, fit_set1.loc[:,:'Target'],fit_set1.loc[:,'Target'], cv = 10, scoring = scorer)

print(f'F1 Score = {round(cv_score.mean(), 4)}, std = {round(cv_score.std(), 4)}')
cols = list(fit_set1.columns)
cols.pop()

means = {}
stds = {}
fit_norm = fit_set1[cols]

for c in cols:
        means[c] = fit_norm[c].mean()
        stds[c] = fit_norm[c].std()

fit_norm = (fit_norm - fit_norm.mean()) / (fit_norm.std())
fit_norm['Target'] = fit_set1['Target']

fit_norm.head()
fit_norm.info()
fit_norm.select_dtypes(np.float64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
plt.xlabel('Number of Unique Values'); plt.ylabel('Count');
plt.title('Count of Unique Values in Float Columns');
knn = KNeighborsClassifier(n_neighbors=14)
scores = cross_val_score(knn,
                         fit_norm.loc[:,:'Target'],
                         fit_norm.loc[:,'Target'],
                         cv=10)
display(scores)
display(scores.mean())
scores_array = []
for n in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn,
                             fit_norm.loc[:,:'Target'],
                             fit_norm.loc[:,'Target'],
                             cv=10)
    scores_array.append(scores.mean())
plt.plot(range(1,30),scores_array, 'ro')
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
knn = KNeighborsClassifier(n_neighbors=14)
cv_score = cross_val_score(knn, fit_norm.loc[:,:'Target'],fit_norm.loc[:,'Target'], cv = 10, scoring = scorer)

print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')
target0 = []

for i,row in fit_norm.iterrows():
    if row['Target'] == 4:
        target0.append(4)
    else:
        target0.append(0)

fit_norm0 = fit_norm.drop(columns=['Target']).copy()
fit_norm0['target0'] = target0

fit_norm0['Target'] = fit_norm['Target'].copy()
scores_array = []
cols = list(fit_norm.columns)
cols.pop()
cols.pop()

for n in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn,
                             fit_norm0.loc[:,cols],
                             fit_norm0.loc[:,'target0'],
                             cv=10)
    scores_array.append(scores.mean())
    
plt.plot(range(1,30),scores_array, 'ro')
scores_array = []
for n in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn,
                             fit_norm.loc[fit_norm0['target0'] == 0,:'Target'],
                             fit_norm.loc[fit_norm0['target0'] == 0,'Target'],
                             cv=10)
    scores_array.append(scores.mean())
    
plt.plot(range(1,30),scores_array, 'ro')
data_f = data.drop(columns=fit_drop).copy()
data_f['Target'] = data['Target'].copy()
data_f = data_f.loc[data_f['Target'].isnull()]
cols = list(data_f.columns)
cols.pop()

for c in cols:
        data_f[c] = data_f[c].subtract(means[c]).divide(stds[c])
        
data_f.info()
Target = fit_norm.loc[:,'Target'].copy()
fit_norm = fit_norm.drop('Target', axis=1)
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(kind='svm').fit_sample(fit_norm, Target)
knn = KNeighborsClassifier(n_neighbors=14)
scores = cross_val_score(knn,
                         X_resampled, y_resampled,
                         cv=10)
display(scores)
display(scores.mean())
scores_array = []
for n in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    scores = cross_val_score(knn,
                             X_resampled, y_resampled,
                             cv=10)
    scores_array.append(scores.mean())
plt.plot(range(1,30),scores_array, 'ro')
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_resampled, y_resampled)
data_test = data_f.drop('Target', axis=1)
testPred = knn.predict(data_test)
data_test.head()
arq = open ("prediction.csv", "w")
arq.write("Id,Target\n")
for i, j in zip(data_test.index, testPred):
    arq.write(str(i)+ "," + str(int(j))+"\n")
arq.close()
