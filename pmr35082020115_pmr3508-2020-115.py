import pandas as pd

import sklearn
# Importação da base de dados alterando os nomes das colunas e admitindo os valores "?" como nulo.

adult = pd.read_csv('../input/adult-pmr3508/train_data.csv',names=[

        'Id',"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        header = 0)
adult.shape
adult.head()
adult.set_index('Id', inplace=True)
adult.head()
adult.info()
adult.describe()
adult["Country"].value_counts()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
plt.figure(figsize=(10,3))

adult['Martial Status'].value_counts().plot(kind = 'bar')

plt.title('Martial Status')
plt.figure(figsize=(10,3))

adult.Relationship.value_counts().plot(kind ='bar', color ='#d963db')

plt.title('Relationship')
plt.figure(figsize=(10,3))

adult.Race.value_counts().plot(kind='bar', color='#e87909')

plt.title('Race')
bins = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure(figsize=(10,3))

plt.hist(adult.Age, bins=bins, color = '#ebeb50', edgecolor='gray')

plt.xlabel("Idade")

plt.ylabel("Frequência")

plt.title('Age Histogram')

plt.xticks(bins)

plt.show()
bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

plt.figure(figsize=(10,3))

plt.hist(adult['Education-Num'], bins=bins, color='#79eb50', edgecolor='gray')

plt.title('Education')

plt.xlabel('Years')

plt.ylabel('Frequency')

plt.xticks(bins)

plt.show()
plt.figure(figsize=(10,3))

adult.Education.value_counts().plot(kind='bar', color='#eb50a0')

plt.title('Education Grade')
plt.figure(figsize=(10,3))

adult["Capital Gain"].hist(color='#50ebb5')

plt.title('Capital Gain')

plt.show()
plt.figure(figsize=(10,3))

adult["Capital Loss"].hist( color='#db6363')

plt.title('Capital Loss')

plt.show()
plt.pie(adult.Sex.value_counts(), labels = ["Male", "Female"], autopct ='%.2f %%', colors =['#6b9ff2','#e8affa'])

plt.title('Sex Distribuition')
fig, axes = plt.subplots(nrows=1, ncols=2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)



sexes = adult.Sex.unique()

i=0

for ax in axes:

    ax.pie(adult[adult['Sex']==sexes[i]]['Target'].value_counts(), autopct = '%.2f %%', colors=['#faa7a7', '#88f7ab'])

    ax.legend(adult.Target.unique())

    ax.set(title='Income for {}'.format(sexes[i]))

    i+=1



plt.show()
corr_mat = adult.corr()

sns.set()

plt.figure(figsize=(10,8))

sns.heatmap(corr_mat, annot=True,vmin = -1., vmax = 1., cmap = plt.cm.RdYlGn_r)
data = pd.concat([adult['Education-Num'], adult['Target']], axis=1)



f, ax = plt.subplots(figsize=(5,5))



colors=['#faa7a7', '#88f7ab']

sns.boxplot(x='Target', y='Education-Num', data=data, notch = True, palette=colors)

plt.title('Boxplot of Years of Education over Income')

plt.show()
targets = adult.Target.unique()

plt.figure(figsize=(10,3))

colors=['#faa7a7', '#88f7ab']

i=0

for target in targets:

    plt.hist(adult[adult['Target']==target]['Education-Num'], alpha = 0.5, color=colors[i], edgecolor = 'gray')

    i+=1

plt.legend(adult.Target.unique())

plt.title('Histogram of Years of Education by Income')



plt.show()
targets = adult.Target.unique()

plt.figure(figsize=(10,3))

colors=['#faa7a7', '#88f7ab']

i=0

for target in targets:

    plt.hist(adult[adult['Target']==target]['Age'], alpha = 0.5, color=colors[i], edgecolor= 'gray')

    i+=1

plt.legend(adult.Target.unique())

plt.title('Histogram of Age by Income')



plt.show()
targets = adult.Target.unique()

plt.figure(figsize=(10,3))

colors=['#faa7a7', '#88f7ab']

i=0

for target in targets:

    plt.hist(adult[adult['Target']==target]['Hours per week'], alpha = 0.5, color=colors[i], edgecolor= 'gray')

    i+=1

plt.legend(adult.Target.unique())

plt.title('Histogram of Hours per week by Income')



plt.show()
data = pd.concat([adult['Education-Num'], adult['Race']], axis=1)



f, ax = plt.subplots(figsize=(10,5))



sns.boxplot(x='Race', y='Education-Num', data=data, notch = True)

plt.title('Boxplot of Years of Education over race')

plt.show()
nadult = adult.dropna()
nadult
testadult = pd.read_csv('../input/adult-pmr3508/test_data.csv',

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        header = 0)
testadult.set_index('Id', inplace=True)
ntestadult = testadult.dropna()
ntestadult.head()
Y_adult= nadult['Target']



X_adult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]



X_testadult = testadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
from sklearn.neighbors import KNeighborsClassifier
KNNclf = KNeighborsClassifier(n_neighbors=10)
from sklearn.model_selection import cross_val_score
score = cross_val_score(KNNclf, X_adult, Y_adult, cv = 5, scoring="accuracy")
score
print("Acurácia com cross validation:", score.mean())
scores_mean = []

scores_std = []



k_max = None

max_std = 0

max_acc = 0



i = 0

print('Finding best k...')

for k in range(1,31):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k)

    

    score = cross_val_score(KNNclf, X_adult, Y_adult, cv = 5)

    

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

        max_std = scores_std[i]

    i += 1

    if not (k%3):

        print('   K = {0} | Best CV acc = {1:2.2f}% +/-{3:4.2f}% (best k = {2})'.format(k, max_acc*100, k_max, max_std*100))

print('\nBest k: {}'.format(k_max))
KNNclf = KNeighborsClassifier(n_neighbors=16)

KNNclf.fit(X_adult,Y_adult)
Y_testPred = KNNclf.predict(X_testadult)

Y_testPred
prediction = pd.DataFrame()
prediction[0] = testadult.index

prediction[1] = Y_testPred

prediction.columns = ['Id','income']
prediction.head()
prediction.to_csv('prediction.csv',index = False)