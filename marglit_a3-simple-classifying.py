import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt
import random

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
def split_and_plot(pointset): # Work-Around um ungewollten automatischen Konsolen-Output zu verhindern
    # Trennen wir die Punkte nach "Farbe"...
    red = [[i[0],i[1]] for i in pointset if i[2] == 1]
    blue = [[i[0],i[1]] for i in pointset if i[2] == -1]
    # ...und plotten das Ganze.
    plt.plot([i[0] for i in red], [i[1] for i in red], 'ro', [i[0] for i in blue], [i[1] for i in blue], 'bo')
    return red, blue 

plt.title("Die Trainingsdaten")
red, blue = split_and_plot(train)
def avrg_point(pointset, plotstring):
    avrg = np.array([0.0,0.0])
    for point in pointset:
        avrg += point
    avrg /= len(pointset)
    plt.plot(avrg[0], avrg[1], plotstring)
    return avrg

plt.title("Unsere Hot-Spots")
red = avrg_point(red, 'ro')
blue = avrg_point(blue, 'bo')
def predict(point, result = 0):
    if np.linalg.norm(point - red) < np.linalg.norm(point - blue):
        plt.plot(point[0], point[1], 'rx' if (result == -1) else 'ro')
        return 1
    else:
        plt.plot(point[0], point[1], 'bx' if (result == 1) else 'bo')
        return -1
plt.title("Testergebniss")
correct = 0.
incorrect = 0.
for entry in train:
    if predict([entry[0], entry[1]], entry[2]) == entry[2]:
        correct += 1
    else:
        incorrect += 1
print("Fehlerquote: " + str(100 * incorrect / correct) + "%")
#Submission
plt.title("Unsere Abgabe")
test = pd.read_csv('../input/kiwhs-comp-1-complete/test.csv', index_col='Id', header=0)
with open('submission.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['Id (String)', 'Category (String)'])
    for tid,x,y in test.itertuples():
        spamwriter.writerow([tid, predict([x, y])])