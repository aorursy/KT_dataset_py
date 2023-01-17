import numpy as np

import itertools

import matplotlib.pyplot as plt



def confusion_matrix(yt, yp, classes):

    instcount = yt.shape[0]

    n_classes = classes.shape[0]

    mtx = np.zeros((n_classes, 4))

    for i in range(instcount):

        for c in range(n_classes):

            mtx[c,0] += 1 if yt[i,c]==1 and yp[i,c]==1 else 0

            mtx[c,1] += 1 if yt[i,c]==1 and yp[i,c]==0 else 0

            mtx[c,2] += 1 if yt[i,c]==0 and yp[i,c]==0 else 0

            mtx[c,3] += 1 if yt[i,c]==0 and yp[i,c]==1 else 0

    mtx = [[m0/(m0+m1), m1/(m0+m1), m2/(m2+m3), m3/(m2+m3)] for m0,m1,m2,m3 in mtx]

    plt.figure(num=None, figsize=(5, 15), dpi=100, facecolor='w', edgecolor='k')

    plt.imshow(mtx, interpolation='nearest',cmap='Blues')

    plt.title("title")

    tick_marks = np.arange(n_classes)

    plt.xticks(np.arange(4), ['1 - 1','1 - 0','0 - 0','0 - 1'])

    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(n_classes), range(4)):

        plt.text(j, i, round(mtx[i][j],2), horizontalalignment="center")



    #plt.tight_layout()

    plt.ylabel('labels')

    plt.xlabel('Predicted')

    plt.show()



# y1 = np.genfromtxt('y_true.csv', delimiter=",")

# y2 = np.genfromtxt('y_pred.csv', delimiter=",")

# labels = np.genfromtxt('labels.csv', delimiter=",", dtype="|S")

# confusion_matrix(y1, y2, labels)