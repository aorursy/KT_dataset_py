import pandas as pd

import seaborn as sns

import sklearn.metrics

import matplotlib.pyplot as plt

import numpy as np

import warnings



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



warnings.filterwarnings("ignore")

rnd_state = 65
data = pd.read_csv("../input/Admission_Predict.csv", index_col=0)

data = data.rename(columns={'GRE Score' : 'GRE_Score',

                            'TOEFL Score' : 'TOEFL_Score',

                            'University Rating' : 'University_Rating',

                            'Chance of Admit ' : 'Chance_of_Admit',

                            'LOR ' : 'LOR'})



data.loc[data.Chance_of_Admit >= 0.90, 'Chance_of_Admit_Class'] = "High"

data.loc[(data.Chance_of_Admit >= 0.75) & (data.Chance_of_Admit < 0.90), 'Chance_of_Admit_Class'] = "Average"

data.loc[data.Chance_of_Admit <= 0.75, 'Chance_of_Admit_Class'] = "Low"



del data['Chance_of_Admit']



print (data['Chance_of_Admit_Class'].value_counts())
print(data.info())
print("Início da tabela de dados\n{0}".format(data.head()))
print("Informações sobre a tabela\n{0}".format(data.describe(include='all')))
print("Formato do conjunto de dados {0}".format(data.shape))
research_w_dirt = data["Research"] + 1e-12*np.random.rand(data["Research"].shape[0])

plottableData = data.copy()

plottableData["Research"] = research_w_dirt

sns.pairplot(data, kind="scatter", hue="Chance_of_Admit_Class")

plt.show()
X = data[["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]]

Y = data[["Chance_of_Admit_Class"]]



splitted = train_test_split(X, Y, test_size=0.20, random_state=rnd_state)



train = { "x" : splitted[0], "y" : splitted[2] }

test = { "x" : splitted[1], "y" : splitted[3] }



print(train['x'].shape)

print(train['y'].shape)



print(test['x'].shape)

print(test['y'].shape)
accuracy = { 'train' : [], 'test' : []}

decision_trees = []

leaf_node_range = range(2,50)



for mln in leaf_node_range:

  modelo = DecisionTreeClassifier(max_leaf_nodes=mln, criterion='entropy', random_state=rnd_state)

  scores = cross_val_score(modelo, train['x'], train['y'], cv=10)

  accuracy['train'].append(scores.mean())

  

  modelo.fit(train['x'], train['y'])

  ypred = modelo.predict(test['x'])

  accuracy['test'].append(sklearn.metrics.accuracy_score(test['y'], ypred))

  print("Nº de nós folha: {0}".format(mln))

  print("Score {0}\n".format(modelo.score(test['x'], test['y'])))

  decision_trees.append(modelo)
plt.plot(leaf_node_range, accuracy['train'], label='Treino')

plt.plot(leaf_node_range, accuracy['test'], label='Teste')

plt.ylabel('Acurácia')

plt.xlabel('Máximo de Nós Folha')

plt.legend()



plt.show()
decision_tree = decision_trees[np.argmax(accuracy['test'])]
ypred = decision_tree.predict(test['x'])

print(ypred)
y_scores_arvore = decision_tree.predict_proba(test['x'])

print(y_scores_arvore)
acuracia = sklearn.metrics.accuracy_score(test['y'], ypred)

best_mln = leaf_node_range[np.argmax(accuracy['test'])]

print("Acurácia: {0:.2f}%".format(acuracia * 100))

print("Nº de nós folhas: {0}".format(best_mln))
importance_frame = pd.DataFrame()

importance_frame['Columns'] = data.columns[:7]

importance_frame['Importance'] = modelo.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)



plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7], importance_frame['Columns'])

plt.xlabel('Ganho de informação')

plt.title('Contribuição dos atributos')

plt.show()
conf_mat = sklearn.metrics.confusion_matrix(test['y'], ypred)



df_cm = pd.DataFrame(conf_mat, index = [i for i in ["High", "Average", "Low"]],

                  columns = [i for i in ["High", "Average", "Low"]])



cmap = sns.light_palette("navy", as_cmap=True)

plt.figure()

sns.heatmap(df_cm, annot=True, cmap=cmap)

plt.show()
sklearn.tree.export_graphviz(decision_tree, out_file='tree.dot')