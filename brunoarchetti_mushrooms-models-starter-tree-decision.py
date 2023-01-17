# Bibliotecas para carregar/manipular os dados
import pandas as pd
import numpy as np

# Bibliotecas para geração de gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Bibliotecas de machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# Carrega a base de dados mushrooms disponível no próprio Kaggle e adicionado neste notebook
mushrooms = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

mushrooms.head() #exibe as 5 (cinco) primeiras linhas do dataset
# Breve análise exploratória do dataset
# Quantidade de linhas agrupadas pelas classes comestiveis e venenosas
print(mushrooms.groupby('class').size(), '\n')

# View geral do dataset
mushrooms.info()
fig, ax = plt.subplots(figsize=(15, 8))

# Não consegui colocar o jitter como parâmetro da função scatterplot
sns.scatterplot(x='odor',  y='veil-color', data=mushrooms, hue='class', size='class', sizes=(500, 1000))

ax.set(xlabel='odor',
       ylabel='veil-color',
       title='Gráfico de Dispersão - Base de Dados Mushrooms',
       )

plt.show()
# É possível identificar por ex. que o odor igual a 'n' e o veil_color igual a 'w'
# é o único local onde há interseção entre as classes venenosas e comestiveis
features = mushrooms.iloc[:, 1:23] # Seleciona os dados sem as classes na variável features
labels = mushrooms.iloc[:,0] # Seleciona os dados das classes na variável labels


# Conversão das features para valores inteiros
features = pd.get_dummies(features)

# Realiza a divisão do dataset inserindo 20% dos dados para teste e 80% para treino
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)
print(y_train.value_counts(), '\n') # Valores de classe utilizados para treino
print(y_test.value_counts()) # Valores de classe utilizados para teste
# A variável X_train recebe a porção de dados usado para treinar o modelo.
# Este contém apenas os dados (80% deles), sem as classes:
X_train
# As classes dos dados de treino estão na variável y_train:
y_train
# Já a variável X_test estão os dados (20% deles) usados para testar o modelo 
# (dados que o modelo não conhece)
X_test
# As classes dos dados de teste estão na variável y_test
y_test
model = DecisionTreeClassifier(criterion = 'gini', random_state = None) # Instanciando uma árvore para classificação 

model = model.fit(X_train, y_train) # Constrói o modelo a partir da base de treinamento
#score = model.score(X_train, y_train)
#print(score)

y_pred = model.predict(X_test) # Utiliza o modelo treinado para realizar
                               # previsões sobre a base teste desconhecida 

accuracy = accuracy_score(y_test, y_pred) # Verifica a acurácia do modelo

print(f'Acurária obtida por meio do modelo árvore de Decisão: {accuracy:.2%}')
pd.crosstab(y_pred,
            y_test,
            rownames=['Previsto'],
            colnames=['Real'],
            margins=True)
# Relatório de classificação
print(classification_report(y_test, y_pred))
feature_names = features.columns.to_list() # Lista contendo o nome dos atributos 

plot_data = export_graphviz(model,
                           #max_depth=10,
                           feature_names=feature_names, 
                           filled=True, rounded=True, 
                           special_characters=True,
                          leaves_parallel=True) 

graphviz.Source(plot_data)