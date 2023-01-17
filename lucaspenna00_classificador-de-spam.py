import pandas as pd

df = pd.read_csv('train_data.csv')
df.head(10)
df.info()
df.describe()
'''

Com o método Query, conseguimos dividir o dataframe filtrando-o pela variável target "ham";

'''

# Separando o dataframe (apenas emails não spam):

df_nospam = df.query('ham == 1')

# Separando o dataframe (apenas emails spam):

df_spam = df.query('ham == 0')
import numpy as np

# Calculando as médias de frequência do caractere $

nospam_cifrao_mean = np.mean(df_nospam['char_freq_$'])
spam_cifrao_mean = np.mean(df_spam['char_freq_$'])

# Criando um histograma para comparar a frequência de $ em emails spam e não spam
from matplotlib import pyplot as plt

%matplotlib inline

locations = [1, 2]
heights = [nospam_cifrao_mean, spam_cifrao_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média de caractere $ por tipo de email')
plt.ylabel('Taxa média de caractere $')
plt.xlabel('Tipo de email')

# Calculando as médias da taxa de frequẽncia do caractere "!"

nospam_exclamacao_mean = np.mean(df_nospam['char_freq_!'])
spam_exclamacao_mean = np.mean(df_spam['char_freq_!'])

# Gerando o gráfico
 
locations = [1, 2]
heights = [nospam_exclamacao_mean, spam_exclamacao_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média de caractere ! por tipo de email')
plt.ylabel('Taxa média de caractere !')
plt.xlabel('Tipo de email')

## PLOTANDO HISTOGRAMA PARA HAM

no_spam_means = np.zeros(shape=54)

for i in range(1, 54):

   nome = df_nospam.columns[i]
   no_spam_means[i] = np.mean(df_nospam[nome])

plt.figure(figsize=(20, 10))
locations = df_nospam.columns[:54]
heights = no_spam_means
plt.bar(locations, heights)
plt.title('Taxa média de caractere em email HAM')
plt.ylabel('Taxa média de caractere')
plt.xlabel('Tipo de caractere')
plt.xticks(df_nospam.columns[:54], rotation='vertical')

## PLOTANDO HISTOGRAMA PARA SPAM

spam_means = np.zeros(shape=54)

for i in range(1, 54):

   nome = df_spam.columns[i]
   spam_means[i] = np.mean(df_spam[nome])

plt.figure(figsize=(20, 10))
locations = df_spam.columns[:54]
heights = spam_means
plt.bar(locations, heights)
plt.title('Taxa média de caractere em email SPAM')
plt.ylabel('Taxa média de caractere')
plt.xlabel('Tipo de caractere')
plt.xticks(df_spam.columns[:54], rotation='vertical')

plt.show()
from sklearn.naive_bayes import GaussianNB
features_train = df.drop(columns=['ham'])
target_train = df['ham']
gnb = GaussianNB()

gnb.fit(features_train, target_train)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, features_train, target_train, cv=10)

print(scores)
print("\n")
print("A media da Validacao Cruzada eh: %s" % scores.mean())
## Carregando a base de testes

df_test = pd.read_csv('test_features.csv')

## Realizando as predições

predictions = gnb.predict(df_test)

## salvando as predições num arquivo CSV

# corrigindo o formato de submissão

sample = pd.read_csv('sample_submission_1.csv')
submit = sample.drop(['ham'], axis = 1)
submit['ham'] = predictions
submit = submit.set_index('Id')

submit.to_csv('predictions.csv')