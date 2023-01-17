import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import neighbors
#from sklearn.neighbors import  NearestNeighbors
from sklearn import model_selection
import os
print(os.listdir("../input"))
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
org_data = read_data("../input/train.arff")
data = [(x,y,c) for x,y,c in org_data]
data = np.array(data)
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)
print(len(train_x),len(test_x))
print(test_y)
#Auswertungsfunktion
#Freie Interpretation von KneighborsClassifier.score
def score(testsamples,truelabelx):
    accuracy =0
    correct = 0
    for i in range(0, len(truelabelx)):
        accuracy+=1
        if testsamples[i] == truelabelx[i]:
            correct+=1
    return correct / accuracy        
neighbor = neighbors.KNeighborsClassifier()
neighbor.fit(train_x,train_y)
pred = neighbor.predict(test_x)
print("Score = %f "%(score(pred,test_y)))

def optimize(weight,algo):
    maximum = 0
    for i in range(1, 100,1):
        for j in range(0,len(weight),1):
            for k in range(0, len(algo), 1):
                for l in range(1,100,1):
                    neighbor = neighbors.KNeighborsClassifier(i,weight[j],algo[k],l)
                    neighbor.fit(train_x,train_y)
                    pred = neighbor.predict(test_x)
                    blubb = score(pred,test_y)
                    print("Score", score(pred,test_y),i,weight[j],algo[k],l)
                    if blubb >= maximum:
                        maximum = blubb
                    
    return maximum   
weight = ['uniform','distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
print(optimize(weight,algorithm))