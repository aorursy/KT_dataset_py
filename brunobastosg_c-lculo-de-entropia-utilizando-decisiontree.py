import pandas as pd 

import graphviz



from sklearn.tree import export_graphviz

from sklearn.tree import DecisionTreeClassifier
# os dados são numéricos pois precisamos treinar uma árvore de decisão

# Marca: 0 = Honda, 1 = Cherry

# Quilometragem: 0 = Baixa, 1 = Média, 2 = Alta

# Quebrou?: 0 = Não, 1 = Sim

data = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 2, 1], [1, 0, 1], [1, 1, 1]] 

df = pd.DataFrame(data, columns = ['Marca', 'Quilometragem', 'Quebrou?']) 



df
X = df[['Marca', 'Quilometragem']]

y = df['Quebrou?']



clf = DecisionTreeClassifier(criterion='entropy')

clf = clf.fit(X, y)
dot_data = export_graphviz(clf, out_file=None, 

                      feature_names=['Marca', 'Quilometragem'],  

                      class_names=['Sim', 'Não'],  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)
graph