#Als erstes lesen wir die .arff-Datei ein, die unsere Trainingsdaten enthält
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")
import numpy as np
import pandas as pd
df_data = pd.DataFrame({'x':[item[0] for item in train], 'y':[item[1] for item in train], 'Category':[item[2] for item in train]})

df_data.head()

from sklearn.model_selection import train_test_split

X = df_data[["x","y"]].values
Y = df_data["Category"].values
colors = {-1:'red',1:'blue'}

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_Train)

X_Train = scaler.transform(X_Train)
X_Test = scaler.transform(X_Test)
%matplotlib inline
import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],c=df_data["Category"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
from sklearn.neighbors import KNeighborsClassifier

test_accuracy = []

neighbors_range = range(1,20)

for n_neighbors in neighbors_range:
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_Train, Y_Train)
    test_accuracy.append(clf.score(X_Test, Y_Test))    
    
plt.plot(neighbors_range, test_accuracy, label='Genauigkeit bei den Testdaten')
plt.ylabel('Genauigkeit')
plt.xlabel('Anzahl der Nachbarn')
plt.legend()
model = KNeighborsClassifier(n_neighbors = 14)
model.fit(X_Train, Y_Train)

print(model.score(X_Test,Y_Test))
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X, Y.astype(np.integer), clf=model, legend=2, colors=('#aa0000,#0000aa,#00ff00'))


######### hier versuchen wir nun das Vorhersagen#######
testdf = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

testX = testdf[["X","Y"]].values
model.predict(testX)
######################################################


######## Anschließend Speichern wir unsere Vorhersage ab #######
prediction = pd.DataFrame()
id = []
for i in range(len(testX)):
    id.append(i)
    i = i + 1
prediction["Id (String)"] = id 
prediction["Category (String)"] = model.predict(testX).astype(int)
print(prediction[:40])
prediction.to_csv("predict.csv", index=False)
##################### ENDE ####################################

