from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
d=pd.read_csv("../input/lung-cancer-dataset-by-staceyinrobert/survey lung cancer.csv")
d['LUNG_CANCER'] = d['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
d=d.drop("GENDER",axis=1)
d.head()
d.info()
y=d["LUNG_CANCER"]
y.head()
x=d
x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y)
x_test.head()
reg=linear_model.LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)
accuracy_score(y_predict,y_test)
print(accuracy_score(y_predict,y_test)*100,'%')
confusion_matrix(y_predict,y_test)
sns.kdeplot(y_test,cumulative=True, bw=1.5)
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(d, nGraphShown, nGraphPerRow):
    nunique = d.nunique()
    d = d[[col for col in d if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = d.shape
    columnNames = list(d)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = d.iloc[:, i]
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

plotPerColumnDistribution(d, 10, 5)
# Correlation matrix
def plotCorrelationMatrix(d, graphWidth):
    d.dataframeName = 'survey lung cancer.csv'
    filename = d.dataframeName
    d = d.dropna('columns') # drop columns with NaN
    d = d[[col for col in d if d[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if d.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({d.shape[1]}) is less than 2')
        return
    corr = d.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

plotCorrelationMatrix(d,10)
# Scatter and density plots
def plotScatterMatrix(d, plotSize, textSize):
    d = d.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    d = d.dropna('columns')
    d = d[[col for col in d if d[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(d)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    d = d[columnNames]
    ax = pd.plotting.scatter_matrix(d, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = d.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
plotScatterMatrix(d, 20, 10)