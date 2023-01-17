from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for direname,_,filename in os.walk('/kaggle/input'):

    for file in filename: 

        print(os.path.join(direname,file))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

# Data Set Characteristics: Multivariate

# Number of Instances: 197

# Area: Life

# Attribute Characteristics: Real

# Number of Attributes: 23

# Citation: Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008),'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease',

# IEEE Transactions on Biomedical Engineering (to appear).

# plot the data

# Exploratory data analysis

df=pd.read_csv('/kaggle/input/parkinsons.data');

df.head()
features = df.loc[:,df.columns != 'status'].values[:,1:]



labels=df.loc[:,'status'].values



# Get the label of each label (0 and 1) in labels

print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#Normalize all the columns to values [0-1] using scaler=MinMaxScaler((-1,1))

from  sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels
# Split the dataset

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
lm = LogisticRegression()

lm.fit(x_train,y_train)   #fit the data

y_pred = lm.predict(x_test)   #predict the data

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test,y_pred)   #check the prediction error
print("Mean squared error: ", mse)

print("r2 : ", r2_score(y_test,y_pred))
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

y_pred_proba = lm.predict_proba(x_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Train/Test split results:')

print(lm.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))

print(lm.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))

print(lm.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95



plt.figure()

plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')

plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)

plt.ylabel('True Positive Rate (recall)', fontsize=14)

plt.title('Receiver operating characteristic (ROC) curve')

plt.legend(loc="lower right")

plt.show()



print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  

      "and a specificity of %.3f" % (1-fpr[idx]) + 

      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))