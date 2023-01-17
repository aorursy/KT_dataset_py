# Import de algumas bibliotecas que serão utilizadas
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
def scatter_bylabel(xdata, ydata, label, color, pointsize = 150, figdpi = 300, figsize = (10, 10)) -> None:
    """Função auxiliar para facilitar a criação de um scatter plot, que colore os pontos com base em suas *labels*
        
    Args:
        xdata (pandas.Series): Dados do eixo X
        ydata (pandas.Series): Dados do eixo Y
        label (list): Lista com as labels presentes nos dados
        color (list): Lista com as cores que devem ser utilizadas para cada label
        pointsize (int): Tamanho do ponto
        figdpi (int): DPI da figura gerada
        figsize (tuple): Tupla com as dimensões da figura
    Returns:
        Esta função não tem retorno
    """
    fig = plt.figure(figsize = figsize, dpi = figdpi)
    plt.scatter(xdata, ydata, pointsize, c = label, cmap = matplotlib.colors.ListedColormap(
        color
    ), edgecolors='black')
def scatter_point_to_classify(xdata, ydata, label, color, newpoint, pointsize = 150, 
                                                          figdpi = 300, figsize = (10, 10)):
    """Função auxiliar que adiciona um novo pontos antes de realizar um scatter plot. Utiliza como base a função
    auxiliar scatter_bylabel
        
    Args:
        xdata (pandas.Series): Dados do eixo X
        ydata (pandas.Series): Dados do eixo Y
        label (list): Lista com as labels presentes nos dados
        color (list): Lista com as cores que devem ser utilizadas para cada label
        newpoint (dict): Dicionário com os campos (pointx, pointy, color)
        pointsize (int): Tamanho do ponto
        figdpi (int): DPI da figura gerada
        figsize (tuple): Tupla com as dimensões da figura
    Returns:
        Esta função não tem retorno
    """
    import numpy as np
        
    color = np.append(color, newpoint['color'])
    xdata = np.append(xdata, newpoint['pointx'])
    ydata = np.append(ydata, newpoint['pointy'])
    label = np.append(label, newpoint['color'])
    
    scatter_bylabel(xdata, ydata, label, color, pointsize = pointsize, figdpi = figdpi, figsize = figsize)
df = pd.read_csv('../input/generatedrandompoints/points.csv')
df.head()
color = df['label'].str.lower().unique()
scatter_bylabel(df['pointx'], df['pointy'], df['label'], color, figdpi = None, figsize = (7, 7))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
pontos_attr = df.iloc[:, 0:-1]
pontos_attr.head()
pontos_labels = df['label']
pontos_labels.head()
knn.fit(pontos_attr, pontos_labels)
newpoint = {
    'color': 'red',
    'pointx': 3,
    'pointy': 6.5
}
scatter_point_to_classify(df['pointx'], df['pointy'], df['label'], 
                          color, newpoint, figdpi = None, figsize = (7, 7))
knn.predict([[newpoint['pointx'], newpoint['pointy']]])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], df['label'], train_size = 0.7)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, y_pred)
from sklearn.metrics import plot_confusion_matrix
confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(dpi = 110)

disp = plot_confusion_matrix(knn, x_test, y_test, cmap = plt.cm.Blues, 
                                 display_labels = df['label'].unique(), ax = ax)
from sklearn import datasets

iris = datasets.load_iris()
iris.data
iris.target
irisdf = pd.DataFrame({
    'sepal_length': iris.data[:, 0],
    'sepal_width': iris.data[:, 1],
    'petal_length': iris.data[:, 2],
    'petal_width': iris.data[:, 3],
    'label': iris.target
})

irisdf.head()
import seaborn as sns
sns.pairplot(irisdf, hue = 'label')
knn = KNeighborsClassifier(n_neighbors = 2)
x_train, x_test, y_train, y_test = train_test_split(irisdf.iloc[:, 0:-1], irisdf['label'], train_size = 0.7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred
fig, ax = plt.subplots(dpi = 110)

disp = plot_confusion_matrix(knn, x_test, y_test, cmap = plt.cm.Blues, 
                                 display_labels = iris.target_names, ax = ax)