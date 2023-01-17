import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools
import matplotlib.pyplot as plt
!wget 'https://www.dropbox.com/s/6b1ervg883khysg/train_true.csv'
!wget 'https://www.dropbox.com/s/y2g55iywusndq7h/train_preds.csv'
train_true = pd.read_csv('train_true.csv')
train_preds = pd.read_csv('train_preds.csv')
labels  = ['Nucleoplasm','Nuclear membrane','Nucleoli',
'Nucleoli fibrillar center','Nuclear speckles','Nuclear bodies',
'Endoplasmic reticulum ','Golgi apparatus','Peroxisomes',
'Endosomes','Lysosomes','Intermediate filaments ',
'Actin filaments','Focal adhesion sites','Microtubules',
'Microtubule ends','Cytokinetic bridge','Mitotic spindle',
'Microtubule organizing center','Centrosome','Lipid droplets',
'Plasma membrane','Cell junctions','Mitochondria',
'Aggresome','Cytosol','Cytoplasmic bodies', 'Rods & rings']
yt = train_true.values[:, 1:]
yp = (train_preds.values[:, 1:] > 0).astype('int')
# based on https://www.kaggle.com/nikolaikopernik/confusion-matrix
def confusion_matrix(yt, yp, labels):
    instcount = yt.shape[0]
    n_classes = len(labels)
    fp = ((yt - yp) < 0).sum(axis = 0)
    fn = ((yt - yp) > 0).sum(axis = 0)
    tp = (yt*yp).sum(axis = 0)
    tn = ((yt==0)*(yp==0)).sum(axis = 0)
    mtx = np.vstack([tp/(tp + fn), fn/(tp + fn), tn/(tn + fp), fp/(tn + fp)]).T
    plt.figure(num=None, figsize=(5, 15), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(mtx, interpolation='nearest',cmap='Blues')
    plt.title("Confusion matrix")
    tick_marks = np.arange(n_classes)
    plt.xticks(np.arange(4), ['1 - 1','1 - 0','0 - 0','0 - 1'])
    plt.yticks(tick_marks, labels)
    for i, j in itertools.product(range(n_classes), range(4)):
        plt.text(j, i, round(mtx[i][j],2), horizontalalignment="center")

    plt.ylabel('labels')
    plt.xlabel('True-Predicted')
    plt.show()
confusion_matrix(yt, yp, labels)
def miss_classification(yt, yp, classes, percent=False, multi_label=True):
    '''
    Params:
        yt, yp : binary numpy array
        classes : list of names classes
        percent : display persent of miss classification
        multi_label : target can have multy labels
    '''

    n_classes = len(classes)
    fp = ((yt - yp) < 0).astype('int')
    fn = ((yt - yp) > 0).astype('int')
    mtc = (fn.T @ fp)
    if multi_label:
        mtc = np.hstack([mtc, (fp.sum(axis = 0) - mtc.sum(axis = 0)).clip(0)[:,None]])
        classes.append('As extra class')
    if percent:
        mtc = (mtc / mtc.sum(axis = 1)[:,None])
        mtc = np.nan_to_num(mtc).clip(0,1) * 100
        
    plt.figure(num=None, figsize=(12, 12), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(mtc, interpolation='nearest',cmap='Blues')
    plt.title("Miss-Classification table")

    plt.xticks(np.arange(mtc.shape[1]), classes, rotation=90)
    plt.yticks(np.arange(mtc.shape[0]), classes, rotation=0)
    
    for i in range(mtc.shape[0]):
        for j in range(mtc.shape[1]):
            plt.text(j, i, int(mtc[i][j]), horizontalalignment="center")
    plt.ylabel('labels')
    plt.xlabel('Miss classed')
    plt.show()
miss_classification(yt, yp, labels)
miss_classification(yt, yp, labels, percent=True)