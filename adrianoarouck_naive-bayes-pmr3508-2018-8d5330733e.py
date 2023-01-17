import pandas as pd
import sklearn
import matplotlib.pyplot as plt
spamdata = pd.read_csv("../input/dataspam/train_data.csv",
        engine='python')
spamdata.shape
spamdata.head()
spamdata.describe()
spamdata['ham'].value_counts()
spamdata['ham'].value_counts().plot(kind='bar')
#Corelação simples
correlation = pd.DataFrame(spamdata.corr(method='pearson').ham)
#Correlação caso dado != 0
verificadores = list(spamdata.columns)
verificadores.remove('ham')
verificadores.remove('Id')
cor = []
count = []
a=0
mapper = {}
for i in verificadores:
    tempframe = spamdata[[i, 'ham']][spamdata[i] != 0]
    cor.append(tempframe.corr(method='pearson').iat[0, 1])
    count.append(tempframe.shape[0])
    mapper[a] = i
    a += 1
cor_not0 = pd.DataFrame({'ham': cor, 'Not 0': count})
#Neste dataframe colocamos não somente a correlação, mas a quantidade de valores não 0,
#isso se dá para termos uma melhor analise dos dados.

cor_not0 = cor_not0.rename(mapper, axis='index')
#Correlação dado que palavra ou caractere occorre
ocorre = spamdata.drop(['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'], axis=1)
ocorre[ocorre > 0] = 1
cor_ocorre = pd.DataFrame(ocorre.corr(method='pearson').ham)
correlation.join(cor_not0, rsuffix = '_not0').join(cor_ocorre, rsuffix = '_ocorre').sort_values(by='ham', ascending=True)
cs = spamdata[['word_freq_cs', 'ham']][spamdata['word_freq_cs'] != 0]
cs.groupby('ham').size()
notcs = spamdata[['word_freq_cs', 'ham']][spamdata['word_freq_cs'] == 0]
notcs.groupby('ham').size()
good_ham = spamdata[spamdata['ham'] == True]
bad_spam = spamdata[spamdata['ham'] == False]
m_ham = pd.DataFrame(good_ham.mean(), columns = ['ham'])
m_spam = pd.DataFrame(bad_spam.mean(), columns = ['spam'])
means = m_ham.join(m_spam)
means = means.div(means.sum(axis=1), axis=0).sort_values(by='ham', ascending =True)
plt.rcParams['figure.figsize'] = [15, 5]
means.plot(kind='bar')
ocorre.head()
good_o_ham = ocorre[ocorre['ham'] == True]
bad_o_spam = ocorre[ocorre['ham'] == False]
m_o_ham = pd.DataFrame(good_o_ham.mean(), columns = ['ham'])
m_o_spam = pd.DataFrame(bad_o_spam.mean(), columns = ['spam'])
means_o = m_o_ham.join(m_o_spam)
means_o = means_o.div(means_o.sum(axis=1), axis=0).sort_values(by='ham', ascending =True)
plt.rcParams['figure.figsize'] = [15, 5]
means_o.plot(kind='bar')
compare = means_o.join(means, lsuffix='o')
compare = compare[['ham', 'hamo', 'spam', 'spamo']]
compare.loc['word_freq_remove':'Id', :].plot(kind='bar')
compare.loc['Id': , :].plot(kind='bar')
treino = spamdata.drop(columns=['ham', 'Id', 'word_freq_will', 'word_freq_address', 'word_freq_you', 'char_freq_(',
                               ]).join(spamdata[['word_freq_hp', 'word_freq_000', 'word_freq_george',
                                                 'word_freq_3d']], rsuffix = '_2')
treino_alvo = spamdata['ham'] 
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(treino, treino_alvo)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(bnb, treino, treino_alvo, cv=10)

print(scores)
## Carregando a base de testes

df_test = pd.read_csv('../input/dataspam/test_features.csv')
df_test2 = df_test.drop(columns=['Id', 'word_freq_will', 'word_freq_address', 'word_freq_you', 'char_freq_(',
                               ]).join(df_test[['word_freq_hp', 'word_freq_000', 'word_freq_george',
                                                 'word_freq_3d']], rsuffix = '_2')
## Realizando as predições

predictions = bnb.predict(df_test2)

str(predictions)

## Transformando predictions em um Panda DataFrame

df_entrega = pd.DataFrame(predictions, columns=['ham'])
df_final = df_test[['Id']].join(df_entrega)

#Usando a certeza de que CS = Ham
cs = df_test[['word_freq_cs']][df_test['word_freq_cs'] != 0]
for v in cs.index:
    df_final.iloc[v, 1] = True
## salvando as predições num arquivo CSV
df_final.to_csv('predictions.csv', index=False)
cs
treino = spamdata.drop(columns=['ham', 'Id', 'word_freq_will', 'word_freq_address'])
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
scores = cross_val_score(knn, treino, treino_alvo, cv=10)
scores
knn.fit(treino, treino_alvo)
df_test = pd.read_csv('../input/dataspam/test_features.csv')
df_test2 = df_test.drop(columns=['Id', 'word_freq_will', 'word_freq_address'])
## Realizando as predições

predictions = knn.predict(df_test2)

str(predictions)

## Transformando predictions em um Panda DataFrame

df_entrega = pd.DataFrame(predictions, columns=['ham'])
df_final = df_test[['Id']].join(df_entrega)
## salvando as predições num arquivo CSV
df_final.to_csv('predictions2.csv', index=False)

