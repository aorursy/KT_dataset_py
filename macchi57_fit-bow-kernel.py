%%capture

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('../input/fakenewsvortexbsb/train_df.csv', sep=';', error_bad_lines=False, quoting=3);
train.head(4) #primeiros 4 registros
train.shape #numero de registros do dataset
train.columns.values #colunas do dataset
train["manchete"][80] #amostra do dataset
train["Class"][80] #amostra do dataset
train['Class'].unique()#tipos diferentes de labels 
from unidecode import unidecode

example = train["manchete"][1]

print(unidecode(example)) #teste com amsotra do dataset
import re

# aplicamos ambas bibliotecas propostas, unidecode aplica o unicode no texto

# re retira as pontuações.

letters_only=re.sub("[^a-zA-Z]"," ",unidecode(example)) 

print(letters_only)
lower_case=letters_only.lower() #aplica as letras minusculas para todas as letras nos textos

words=lower_case.split()
from nltk.corpus import stopwords

#lista de stop words da lingua portuguêsa

print (stopwords.words("portuguese"))
stop = stopwords.words("portuguese") #criar um arry com as stop words do português
# não é possivel aplicar o unicode em uma lista, então vamos percorrer o array aplciando emc ada registro

lista_stop = [unidecode(x) for x in stop]  

print (lista_stop)
print(words)
#Filtro das palavras não stop words presentes no texto

words=[w for w in words if not w in lista_stop] 

print(words)
# Função review_to_words realiza todas as transformações que foram realizadas antes

def review_to_words(raw_review):

    raw_review = unidecode(raw_review)

    raw_review.lstrip('Jovem Pan')

    letters_only=re.sub("[^a-zA-Z]"," ",raw_review)

    words=letters_only.lower().split()

    meaningful_words=[w for w in words if not w in lista_stop]

    return(' '.join(meaningful_words))
#testando a função

clean_review=review_to_words(train['manchete'][1])

print(clean_review)
# Pegando o valor da dimenção dos dados para passar no for logo abaixo

num_reviews=train['manchete'].size

print (num_reviews)
# loop para aplicar as transformações em cada registro da coluna manifestacao_clean do dataset

clean_train_review=[]

for i in range(0,num_reviews):

    clean_train_review.append(review_to_words(train['manchete'][i]))
from sklearn.feature_extraction.text import CountVectorizer

# configurar os parametros do WordtoVec/Tokeninzação e criar o objeto

vectorizer=CountVectorizer(analyzer='word',tokenizer=None,preprocessor = None, stop_words = None,max_features = 7000)

# aplicar WordtoVec/Tokeninzação

train_data_features=vectorizer.fit_transform(clean_train_review)

# aplicar a estrutura de dados numpy array

train_data_features=train_data_features.toarray()
train_data_features.shape
train_data_features[1]
# vocabulario de todas das palavaras mais importantes de todas requisições

vcab=vectorizer.get_feature_names()

#print(vcab)
train_y = train["Class"]
from sklearn.model_selection import train_test_split

# Split dos dados em treino e validação

X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_y, test_size=0.25, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

# Criação do objeto que corresponde o modelo com o hiperparametros especificados

model = KNeighborsClassifier(n_neighbors=3)

#Treinamento do modelo

%time model = model.fit( X_train, y_train )
# Prevendo os dados de teste com o modelo treinado

result = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

# Acurrácia absoluta dos resultados

accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
from sklearn.metrics import confusion_matrix

confusion_matrix(result, y_test, labels=y_train.unique())
import seaborn as sn

import matplotlib.pyplot as plt

array = confusion_matrix(result, y_test, labels=y_train.unique())

array = array.astype('float') / array.sum(axis=1)[:, np.newaxis] #normalização dos valores 

df_cm = pd.DataFrame(array, index = y_train.unique(), #cria um data frame para base ao gráfico

                  columns = y_train.unique())

plt.figure(figsize = (10,7)) 

sn.heatmap(df_cm, annot=True, cmap=sn.light_palette((210, 90, 60), input="husl"))
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

%time nb = nb.fit( X_train, y_train )

result = nb.predict(X_test)
accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
from sklearn.tree import DecisionTreeClassifier

clf2 = DecisionTreeClassifier(random_state=42)

%time clf2 = clf2.fit( X_train, y_train )

result = clf2.predict(X_test)
# Acurrácia absoluta dos resultados

accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=42)

%time forest = forest.fit( X_train, y_train )

result = forest.predict(X_test)
# Acurrácia absoluta dos resultados

accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
from sklearn.ensemble import GradientBoostingClassifier

clf3 = GradientBoostingClassifier(random_state=42)

%time clf3 = clf3.fit( X_train, y_train )

result = clf3.predict(X_test)
# Acurrácia absoluta dos resultados

accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42, solver='lbfgs')

%time clf = clf.fit( X_train, y_train )

result = clf.predict(X_test)
# Acurrácia absoluta dos resultados

accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
from sklearn.svm import SVC

clf4 = SVC(random_state=42)

%time clf4 = clf4.fit( X_train, y_train )

result = clf4.predict(X_test)
# Acurrácia absoluta dos resultados

accuracy_score(y_test, result)
# Essa função realiza testes de recall e F1-score do modelo, importante não apenas se basear na acurrácia e precisão.

print (classification_report(y_test, result))
def prevendo_noticias(string, model):

    to_array=[]

    to_array.append(review_to_words(string))

    sample_final=vectorizer.transform(to_array)

    sample_final=sample_final.toarray()

    result = model.predict(sample_final)

    if  result[0] == 1:

        label = 'Fake News'

    else:

        label = 'Verdadeira'

        

    return label, string
prevendo_noticias('Aras: decisão do STF não deveria valer para casos concluídos', forest)
prevendo_noticias('Bolsonaro pessoalmente incendêia a amazonia e mata as girafas', forest)
prevendo_noticias('Jornalista joga água benta em Temer e ele admite que impeachment foi golpe', forest)
# Criação do objeto que corresponde o modelo com o hiperparametros especificados

from sklearn.ensemble import RandomForestClassifier

model_final = RandomForestClassifier(random_state=42)

#Treinamento do modelo

#%time model_final = model_final.fit( train_data_features, train_y )
param_grid = { 

    'n_estimators': [100, 300, 500, 800, 1000],

    'criterion': ['gini', 'entropy'],

    'bootstrap': [True, False]

}
from sklearn.model_selection import GridSearchCV

CV_rf = GridSearchCV(estimator=model_final, param_grid=param_grid, cv= 5, scoring='accuracy', n_jobs=-1)

CV_rf = CV_rf.fit( train_data_features, train_y )
CV_rf.best_params_
model_fit = RandomForestClassifier(random_state=42, bootstrap= True, criterion= 'entropy', n_estimators= 800)#.fit( train_data_features, train_y )
%time model_fit = model_fit.fit( X_train, y_train )

result = model_fit.predict(X_test)
accuracy_score(y_test, result)
%time model_final =  model_fit.fit( train_data_features, train_y )
# Importando o dados de teste

test = pd.read_csv('../input/fakenewsvortexbsb/sample_submission.csv', sep=';', error_bad_lines=False, quoting=3);
test.head(5)
#Filtro das palavras não stop words presentes no texto

num_reviews, = test['Manchete'].shape

print(num_reviews)
# loop para aplicar as transformações em cada registro da coluna manifestacao_clean do dataset

clean_test_review=[]

for i in range(0,num_reviews):

    clean_test_review.append(review_to_words(test['Manchete'][i]))
# aplicar WordtoVec/Tokeninzação

test_data_features = vectorizer.transform(clean_test_review)

test_data_features=test_data_features.toarray()
# Prevendo os dados de teste com o modelo treinado

result_test = model_final.predict(test_data_features)
# Criando um dataframe com os resultados obtidos para submiter na competição

minha_sub = pd.DataFrame({'index': test.index, 'Category': result_test})

# Criando um arquivo csv com os resultados

minha_sub.to_csv('submission.csv', index=False)