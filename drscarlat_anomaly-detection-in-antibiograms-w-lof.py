# IMPORT MODULES

import numpy as np
from numpy import ma
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
%matplotlib inline
from matplotlib import ticker, cm
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import random
import seaborn as sns

from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import sklearn.datasets, sklearn.decomposition
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor

print("Modules imported...YAY !")

import os
print(os.listdir("../input"))
PATH = "../input/"
print(os.listdir(PATH))
# Load MIMIC3 data on Organisms, Antibiotics and their Resistance/Sensitivity ANTIBIOGRAMS

data_raw = pd.read_csv(PATH + 'Antibiograms.csv')
data = data_raw.copy()
data2 = data_raw.copy()
dataTest = data_raw.copy()
print(data.shape)
print(data.info())
print(data.describe())
print(data.head(10))
data
# GENERAL antibiogram - All Ab and All Orgs
data4chart = data.groupby('Organism').mean()
data4chart
# Prep for viz

x4chart = data4chart.iloc[0:,0:0]
print(x4chart)

y4chart = data4chart.columns
print(y4chart)

for x in range(len(x4chart)):
    for y in range(len(y4chart)):
        z = data4chart.iat[x,y]


print(x,y,z)

z4chart = data4chart.iat[4,1]
print(z4chart)
# Chart the general antibiogram

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for xs in range(len(x4chart)):
    for ys in range(len(y4chart)):
        zs = data4chart.iat[xs,ys]
        if zs < -0.001:
            c = 'r'
            m = 'o'
        elif zs > 0.001:
            c = 'g'
            m = '^'   
        else:
            c = 'w'
            m = ''
        ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Organisms')
ax.set_ylabel('Antibiotics')
ax.set_zlabel('Sensitivity / Resistance')

fig.set_size_inches(12,12)
plt.show()
# Chart the general antibiogram perpendicular to antibiotics axis = 
# For EACH Antibiotic - how sensistive / resistant are ALL the orgs
# or...Perpendicular to the Organisms axis = 
# For EACH Organism - how sensistive / resistant it is to ALL the antibiotics

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for xs in range(len(x4chart)):
    for ys in range(len(y4chart)):
        zs = data4chart.iat[xs,ys]
        if zs < -0.001:
            c = 'r'
            m = 'o'
        elif zs > 0.001:
            c = 'g'
            m = '^'   
        else:
            c = 'w'
            m = ''
        ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Organisms')
ax.set_ylabel('Antibiotics')
ax.set_zlabel('Sensitivity / Resistance')

ax.view_init(0, 0) # angle of view on the 3d
plt.draw()

fig.set_size_inches(12,12)
plt.show()
SigmaOrg = data.groupby(['Organism']).count()
#print(SigmaOrg)
SigmaOrg = SigmaOrg.sort_values(by=['Amikacin'], ascending=False)
print(SigmaOrg)
# Search for an organism eith a LIKE mechanism ... part of the name of the org
data.query('Organism.str.contains("STAPH")', engine='python')
#data2.loc[data2['Organism'] == SearchOrg]
# ONE organism and its average antibiogram

SearchOrg = 'STAPH AUREUS COAG +'

# ANTIBIOGRAM summarized for ONE organism
OneOrg = data.loc[data['Organism'] == SearchOrg]
print('Normalized summary antibiogram for ',SearchOrg, ' based on ', OneOrg.shape, ' samples' )
OneOrgAbSigma = round((OneOrg.sum(axis=0, numeric_only=int)/OneOrg.shape[0]),2)
OneOrgAbSigma
# Chart ONE organism antibiogram in 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for xs in range(1):
    for ys in range(len(OneOrgAbSigma)):
        #print(OneOrg.iat[xs,ys])
        zs = OneOrgAbSigma[ys]
        
        if zs < -0.001:
            c = 'r'
            m = 'o'
        elif zs > 0.001:
            c = 'g'
            m = '^'   
        else:
            c = 'w'
            m = ''
        ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Organisms')
ax.set_ylabel('Antibiotics')
ax.set_zlabel('Sensitivity / Resistance')
ax.view_init(0, 0)          # angle of view on the 3d
plt.draw()

fig.set_size_inches(12,12)
plt.show()
# Chart antibiogram for one org

AllAbs = data.columns[1:]
signal = OneOrgAbSigma
pos_signal = signal.copy()
neg_signal = signal.copy()

pos_signal[pos_signal <= 0] = np.nan
neg_signal[neg_signal > 0] = np.nan


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(AllAbs))

ax.barh(y_pos, pos_signal, color='green')
ax.barh(y_pos, neg_signal, color='red')

ax.set_yticks(y_pos)
ax.set_yticklabels(AllAbs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Red = Resistant ... Green = Sensitive')
ax.set_xlim(-1,1)
myTitle = 'Average antibiogram for '+str(SearchOrg)+ ' based on '+str(OneOrg.shape[0])+ ' samples'
ax.set_title(myTitle)
plt.show()
# One Hot Encoder
print('Before One Hot Encoder',data.shape)
data = pd.get_dummies(data)
# Make numbers float 
data.astype(float)
print('After One Hot Encoder',data.shape)
# Function for plotting confusion matrix

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,
                          normalize=False):
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.show()
# kNN - FIT = Showing the model what is considered NORMAL

n_neighbors = 20
X = data.copy()
lof = LocalOutlierFactor(n_neighbors = n_neighbors,novelty = True)
MyModel = lof.fit(X) 
print('Model fit w novelty ON and ',n_neighbors, 'neighbors')
# Identify ONE organism
# Get the ID # of rows of the organism, so only these rows will be modified below
dataT = dataTest.copy()
SearchOrg = 'STAPH AUREUS COAG +'
dataTix = dataT.loc[dataT['Organism'] == SearchOrg].index
print('Organism to be modified ', SearchOrg, ' with ', dataTix.shape, ' samples')
# Modify the antibiogram of one org
# First zero the ab column...then insert a % RandPerc of modifications
# Set the % of random mods

RandPerc = 1
# Value to enter 0,-1 or 1
ModValue = -1

# Set the Antibiotics to modify
Ab2Mod = ['VANCOMYCIN', 
          'GENTAMICIN', 
          #'TRIMETHOPRIM_SULFA', 
          #'RIFAMPIN', 
         # 'TETRACYCLINE', 
         # 'NITROFURANTOIN', 
         ]

for w in Ab2Mod: 
    ModCounter = 0
    ColNum = dataT.columns.get_loc(w)
    ix = [(row, col) for row in dataTix for col in range(ColNum,ColNum+1)]
    for row, col in random.sample(ix, int(round(RandPerc*len(ix)))):
        dataT.iat[row, col] = ModValue        
        ModCounter = ModCounter + 1
print('modified organism ',SearchOrg)
print('Antibiotic ', Ab2Mod)
print('ModValue', ModValue)
print('Number of mods', ModCounter, '~', round(RandPerc*100),'%')
print('dataT ', dataT.shape)

# Modify dataT same as above: one hot encoder, pca

dataTpre1hot = dataT.copy()
# One Hot Encoder
print('Before One Hot Encoder', dataT.shape)
dataT = pd.get_dummies(dataT)
# Make numbers float 
dataT.astype(float)
print('After One Hot Encoder', dataT.shape)
# PREDICT IN/OUTliers based on the LOF model fit previously on normal

X = dataT.copy()
MyPred = MyModel.predict(X)

OutNum = (MyPred == -1).sum() 
InNum = (MyPred == 1).sum() 
Total = InNum + OutNum
print('Outliers ', OutNum)
print('Inliers ', InNum)
print('Total ', Total)
# Concat dataTpre1hot with IN/OUTliers
print(dataTpre1hot.shape)
MyPred = MyPred.reshape(25448, 1)
print(MyPred.shape)
OutIn = np.concatenate((dataTpre1hot, MyPred), axis=1)
# Concat data with IN/OUTliers
headers = list(dataTpre1hot.columns.values)#.append('Outlier')
headers.append('Outlier')
OutInPd = pd.DataFrame(OutIn, columns = headers)
print(OutInPd.shape)
#OutInPd
outliers = OutInPd[OutInPd['Outlier'] == -1]
inliers = OutInPd[OutInPd['Outlier'] == 1]
print('outliers ', outliers.shape)
print('inliers ', inliers.shape)

SigmaOrg = outliers.groupby(['Organism']).sum()
SigmaOrg = SigmaOrg.sort_values(by=['Outlier'])
print('OUTLIERS antibiograms ')
print(SigmaOrg)
print('_'*80)
SigmaOrg = inliers.groupby(['Organism']).sum()
SigmaOrg = SigmaOrg.sort_values(by=['Outlier'], ascending=False)
print('INLIERS antibiograms')
print(SigmaOrg)
# Confusion Matrix below depends on if the above relates to:
# Baseline ... then the results are FP and TN
# Known number of artificial anomalies (modifications above) ... then results are TP and FN

# Next 2 lines - ONLY ONCE for the baseline FP and TN - NOT for mods
#FP = outliers.loc[outliers['Organism'] == SearchOrg].shape[0]
#TN = NumOrgs - FP
# With k=20 ... FP = 547 ... TN = 6378
# k=2 (neighbors) results below:

FP= 547
TN = 6378

NumOrgs = dataTpre1hot.loc[dataTpre1hot['Organism'] == SearchOrg].shape[0]
TP = outliers.loc[outliers['Organism'] == SearchOrg].shape[0]
FN = NumOrgs - TP

print('NumOrgs ', NumOrgs)
#print('FP ', FP)
#print('TN ', TN)
#print('TP ', TP)
#print('FN ', FN)

############                    Confusion Matrix 

cm = np.zeros([2, 2], dtype=np.int32)
cm[0,0] = TN
cm[0,1] = FP
cm[1,0] = FN
cm[1,1] = TP

accuracy = (TP+TN)/(TP+TN+FP+FN)
recall = TP/(TP+FN)
precision = TP/(TP+FP)
f1score = 2*recall*precision/(recall+precision)

print ('TN: ', round(TN,0))
print ('FP: ', round(FP,0))
print ('FN: ', round(FN,0))
print ('TP: ', round(TP,0))
print('_'*40)

print ('accuracy ',round(accuracy,4))
print('recall ', round(recall,4))
print('precision ', round(precision,4))
print('F1Score ', round(f1score,4))


plot_confusion_matrix(cm, 
                      normalize    = False,
                      target_names = [0,1],
                      title        = "Confusion Matrix for " + str(SearchOrg)
                     )
# ONE organism and its average antibiogram - OUTLIERS

SearchOrg = 'STAPH AUREUS COAG +'

AllAbs = dataTpre1hot.columns[1:]

# ANTIBIOGRAM summarized for ONE organism - INLIERS
OneOrg = data2.loc[data2['Organism'] == SearchOrg]
#OneOrg = OneOrg.drop(['Outlier'], axis=1)
OneOrgAbSigma = round((OneOrg.iloc[:, 1:].sum(axis=0) / OneOrg.shape[0]), 3)
print('Normalized antibiogram for ',SearchOrg, ' based on ', OneOrg.shape, ' INLIERS samples' )
print(OneOrgAbSigma)

signal = OneOrgAbSigma
pos_signal = signal.copy()
neg_signal = signal.copy()

pos_signal[pos_signal <= 0] = np.nan
neg_signal[neg_signal > 0] = np.nan

# Chart antibiogram for one org
fig, ax = plt.subplots()
y_pos = np.arange(len(AllAbs))

ax.barh(y_pos, pos_signal, color='lightgreen')
ax.barh(y_pos, neg_signal, color='tomato')

ax.set_yticks(y_pos)
ax.set_yticklabels(AllAbs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Red = Resistant  ... Green = Sensitive ')
ax.set_xlim(-1,1)
myTitle = 'Normalized antibiogram for '+str(SearchOrg)+ ' based on '+str(OneOrg.shape)+ ' INLIERS samples'
ax.set_title(myTitle)
plt.show()

###################################################################
# ANTIBIOGRAM summarized for ONE organism
OneOrg = outliers.loc[outliers['Organism'] == SearchOrg]
OneOrg = OneOrg.drop(['Outlier'], axis=1)
OneOrgAbSigma = round((OneOrg.iloc[:, 1:].sum(axis=0) / OneOrg.shape[0]), 3)

print('Normalized summary antibiogram for ',SearchOrg, ' based on ', OneOrg.shape, ' OUTLIERS samples' )
print(OneOrgAbSigma)

signal = OneOrgAbSigma
pos_signal = signal.copy()
neg_signal = signal.copy()

pos_signal[pos_signal <= 0] = np.nan
neg_signal[neg_signal > 0] = np.nan

# Chart antibiogram for one org
fig, ax = plt.subplots()
y_pos = np.arange(len(AllAbs))

ax.barh(y_pos, pos_signal, color='green')
ax.barh(y_pos, neg_signal, color='red')

ax.set_yticks(y_pos)
ax.set_yticklabels(AllAbs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Red = Resistant ... Green = Sensitive ')
ax.set_xlim(-1,1)
myTitle = 'Normalized antibiogram for '+str(SearchOrg)+ ' based on '+str(OneOrg.shape)+ ' OUTLIERS samples'
ax.set_title(myTitle)
plt.show()

################################################################
# ONE organism and its average antibiogram - INLIERS & OUTLIERS 

SearchOrg = 'STAPH AUREUS COAG +'

# ANTIBIOGRAM summarized for ONE organism
OneOrgIn = data2.loc[data2['Organism'] == SearchOrg]
#OneOrgIn = OneOrgIn.drop(['Outlier'], axis=1)
OneOrgAbSigmaIn = round((OneOrgIn.iloc[:, 1:].sum(axis=0) / OneOrgIn.shape[0]), 3)
signalIn = OneOrgAbSigmaIn
pos_signalIn = signalIn.copy()
neg_signalIn = signalIn.copy()

pos_signalIn[pos_signalIn <= 0] = np.nan
neg_signalIn[neg_signalIn > 0] = np.nan

# Chart antibiogram for one org
fig, ax = plt.subplots()
y_posIn = np.arange(len(AllAbs))
y_pos = np.arange(len(AllAbs))

ax.barh(y_pos, pos_signalIn, color='lightgreen')
ax.barh(y_pos+0.2, pos_signal, color='green')
ax.barh(y_pos, neg_signalIn, color='tomato')
ax.barh(y_pos+0.2, neg_signal, color='red')

ax.set_yticks(y_pos)
ax.set_yticklabels(AllAbs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Red = Resistant ... Green = Sensitive ')
ax.set_xlim(-1,1)
myTitle = 'Normalized antibiogram for '+str(SearchOrg)+ ' based on all samples'
ax.set_title(myTitle)
plt.show()
