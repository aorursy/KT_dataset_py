import pandas as pd
import matplotlib.pyplot as plt
def plot_tree(fitted_tree, feature_names, label_names):
    """Função auxiliar para a visualização de uma árvore de decisão treinada
    
    Args:
        fitted_tree (sklearn.tree.DecisionTreeClassifier): Árvore de decisão treinada
        feature_names (list): Lista com o nome dos atributos que estão na árvore
        label_names (list): Lista com o nome das labels que estão presentes nos dados
    Returns:
        graphviz.Source: Objeto com imagem gerada
    See:
        Código adaptado da documentação do sklearn. Confira em: https://scikit-learn.org/stable/modules/tree.html
    """
    
    import graphviz
    from sklearn.tree import export_graphviz
    
    dot_data = export_graphviz(clf, out_file=None, 
                     feature_names=feature_names,  
                     class_names=label_names,  
                     filled=True, rounded=True,  
                     special_characters=True)
    
    graph = graphviz.Source(dot_data)
    return graph
from sklearn import datasets

wine = datasets.load_wine()
x_original, y_original = wine.data, wine.target
wine.feature_names
wine.target_names
winedf = pd.DataFrame(x_original, y_original, columns = wine.feature_names)
winedf['label'] = wine.target
winedf
import seaborn as sns

sns.pairplot(winedf, hue = 'label')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_original, y_original, train_size = 0.7)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(x_train, y_train)
plot_tree(clf, wine.feature_names, wine.target_names)
y_pred = clf.predict(x_test)

y_pred
from sklearn.metrics import plot_confusion_matrix, accuracy_score
accuracy_score(y_test, y_pred)
fig, ax = plt.subplots(dpi = 110)

disp = plot_confusion_matrix(clf, x_test, y_test, cmap = plt.cm.Blues, 
                                 display_labels = wine.target_names, ax = ax)
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(x_train, y_train)
plot_tree(clf, wine.feature_names, wine.target_names)
y_pred = clf.predict(x_test)

y_pred
accuracy_score(y_test, y_pred)
fig, ax = plt.subplots(dpi = 110)
plot_confusion_matrix(clf, x_test, y_test, cmap = plt.cm.Blues, 
                                 display_labels = wine.target_names, ax = ax)
clf = DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes=3)
clf = clf.fit(x_train, y_train)
plot_tree(clf, wine.feature_names, wine.target_names)
y_pred = clf.predict(x_test)

y_pred
accuracy_score(y_test, y_pred)
fig, ax = plt.subplots(dpi = 110)
plot_confusion_matrix(clf, x_test, y_test, cmap = plt.cm.Blues, 
                                 display_labels = wine.target_names, ax = ax)