import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/pulsar_stars.csv')
X_data = dataset.iloc[:, 0:-1].values
Y_data = dataset.iloc[:,-1].values
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data)
model = Sequential()
model.add(Dense(6, input_dim = 8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
from graphviz import Digraph

g = Digraph(format = 'png')
g.attr(rankdir = 'LR')
g.attr(splines = 'false')
g.attr(pad = '0.5')
g.attr(nodesep = '0.1')
g.attr(ranksep = '2.5')

with g.subgraph(name='cluster_0') as c:
    for i in range(1, 9):
        c.node('I_' + str(i), label = '', shape = 'circle')
    c.attr(label = 'Input')
    c.attr(color = 'white')

with g.subgraph(name='cluster_1') as c:
    for i in range(1, 7):
        c.node('N_' + str(i), label = '', shape = 'circle')
    c.attr(label = 'Hidden')
    c.attr(color = 'white')

for x in range(1, 9):
    for y in range(1, 7):
        g.edge('I_'+str(x), 'N_'+str(y))

with g.subgraph(name='cluster_2') as c:
    c.node('O_1', label = '', shape = 'circle')
    c.attr(label = 'Output')
    c.attr(color = 'white')

for y in range(1, 7):
    g.edge('N_'+str(y), 'O_1')

g
model.compile(loss = 'binary_crossentropy' , optimizer='adam' , metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 200 , batch_size=100)
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

auc_keras = auc(fpr, tpr)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='ROC Curve (Area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()