import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))
# Code zum Einlesen der ARFF-Datei: 

def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")

# Lesen der Daten
org_data = read_data("../input/kiwhs-comp-1-complete/train.arff")
print('datapoints:', len(org_data))
print(len(org_data),"Daten wurden eingelesen")
# Pandas-Dataframe aus den Daten zusammenbauen:

df_data = pd.DataFrame({'x':[item[0] for item in train], 'y':[item[1] for item in train], 'Category':[item[2] for item in train]})

df_data.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# X,Y Frames definieren und Farben bestimmen
X = df_data[["x","y"]].values
Y = df_data["Category"].values
colors = {-1:'red',1:'blue'}

# Daten aufsplitten
train_x, test_x, train_y, test_y = train_test_split(X,Y, random_state=0, test_size = 0.2)

scaler = StandardScaler()
scaler.fit(train_x)

# Gesplittete Daten skalieren, damit diese bei Visualisierung geordnet aussehen
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


%matplotlib inline
# Trainingsdaten visualisieren und Farben zuweisen
plt.scatter(X[:,0],X[:,1],c=df_data["Category"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#Source: Competition --> Discussion
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Compare
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA']) 
cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000']) 

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()
#Sklearn bereitgestellte Funktion KNeighborsClassifier wird verwendet.
#Getestet wird mit verschiedenen Modelle mit unterschiedlichen n_neighbor-Parametern. Die werte werden als Graph ausgegeben.

from sklearn.neighbors import KNeighborsClassifier

test_accuracy = []

neighbors_range = range(1,50)

for n_neighbors in neighbors_range:
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(train_x, train_y)
    test_accuracy.append(clf.score(test_x, test_y))    
    
plt.plot(neighbors_range, test_accuracy, label='Genauigkeit bei den Testdaten')
plt.ylabel('Genauigkeit')
plt.xlabel('Anzahl der Nachbarn')
plt.legend()
#Modell wird trainiert und mit dem KNeighborsClassifier=13 berechnet, um den höchsten Peakwert zu erhalten. Im Anschluss wird die genaue Trennlinie angezeigt, welches das Modell gelernt hat. 
#Die Punkte werden in zwei "Bereiche" eingeteilt.

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(13) #Anzahl betrachteter Nachbarn
model.fit(train_x, train_y)
plot_decision_boundary(model, train_x, train_y)

print ('train accuracy: {}'.format(model.score(train_x, train_y)))
print ('test accuracy: {}'.format(model.score(test_x, test_y)))
# Vorhersagen
testdf = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

testX = testdf[["X","Y"]].values
model.predict(testX)

# Speichern der Vorhersagen
submissions_knn = pd.DataFrame()
id = []
for i in range(len(testX)):
    id.append(i)
    i = i + 1
submissions_knn["Id (String)"] = id 
submissions_knn["Category (String)"] = model.predict(testX).astype(int)
print(submissions_knn[:10])
submissions_knn.to_csv("submissions_knn.csv", index=False, header=True)