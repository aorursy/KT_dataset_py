import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier as RF

from matplotlib import pyplot as plt
%matplotlib inline
i = 3168
x = np.zeros((i,21))
t = 0
with open('../input/voice.csv', newline='') as csvfile:
    file_ = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in file_:
        if t == 0:
            labels = row[0].split(',')
            t += 1
            continue
        else:
            if row[0].split(',')[-1] == '"male"':
                x[t-1,-1] = 1
            elif row[0].split(',')[-1] == '"female"':
                x[t-1,-1] = 0
            x[t-1,:-1] = row[0].split(',')[:-1]
            t += 1
idx = np.array([3,5,12])
male = x[x[:,-1] == 1]
female = x[x[:,-1] == 0]
plt.figure(figsize=(10,10))
plt.xlabel(labels[3][1:-1])
plt.ylabel(labels[5][1:-1])
plt.scatter(male[:,3],male[:,5], c = 'b', s = 20,label = "male")
plt.scatter(female[:,3],female[:,5], c = 'r',s = 20,label = "female")
plt.legend()
plt.show()
plt.figure(figsize=(10,10))
plt.xlabel(labels[3][1:-1])
plt.ylabel(labels[12][1:-1])
plt.scatter(male[:,3],male[:,12], c = 'b', s = 20,label = "male")
plt.scatter(female[:,3],female[:,12], c = 'r',s = 20,label = "female")
plt.legend()
plt.show()
plt.figure(figsize=(10,10))
plt.xlabel(labels[5][1:-1])
plt.ylabel(labels[12][1:-1])
plt.scatter(male[:,5],male[:,12], c = 'b', s = 20,label = "male")
plt.scatter(female[:,5],female[:,12], c = 'r',s = 20,label = "female")
plt.legend()
plt.show()
plt.figure(figsize=(10,10))
plt.xlabel(labels[3][1:-1])
plt.ylabel(labels[5][1:-1])
plt.scatter(np.log(male[:,3]),np.log(male[:,5]), c = 'b', s = 20,label = "male")
plt.scatter(np.log(female[:,3]),np.log(female[:,5]), c = 'r',s = 20,label = "female")
plt.legend()
plt.show()
plt.figure(figsize=(10,10))
plt.xlabel(labels[3][1:-1])
plt.ylabel(labels[12][1:-1])
plt.scatter(np.log(male[:,3]),np.log(male[:,12]), c = 'b', s = 20,label = "male")
plt.scatter(np.log(female[:,3]),np.log(female[:,12]), c = 'r',s = 20,label = "female")
plt.legend()
plt.show()
plt.figure(figsize=(10,10))
plt.xlabel(labels[5][1:-1])
plt.ylabel(labels[12][1:-1])
plt.scatter(np.log(male[:,5]),np.log(male[:,12]), c = 'b', s = 20,label = "male")
plt.scatter(np.log(female[:,5]),np.log(female[:,12]), c = 'r',s = 20,label = "female")
plt.legend()
plt.show()