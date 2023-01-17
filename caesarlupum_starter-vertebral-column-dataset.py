from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

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

nRowsRead = 1000 # specify 'None' if want to read whole file

# column_2C.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/column_2C.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'column_2C.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 18, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file

# column_3C.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/column_3C.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'column_3C.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 18, 10)
sns.pairplot(df2, hue="class", size=3, diag_kind="kde")
df2['class'] = df2['class'].map({'Normal': 0, 'Hernia': 1, 'Spondylolisthesis': 2})
def euclidian(p1, p2): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + np.square(p1[i]-p2[i])

    dist = np.sqrt(dist)

    return dist;



def manhattan(p1, p2): 

    dist = 0

    for i in range(len(p1)):

        dist = dist + abs(p1[i]-p2[i])

    return dist;

from sklearn.metrics import confusion_matrix , classification_report

from sklearn.model_selection import train_test_split



all_X = df2[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

all_y = df2['class']





df2=df2[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis', 'class']]

train_data,test_data = train_test_split(df2,train_size = 0.8,random_state=2)

X_train = train_data[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

y_train = train_data['class']

X_test = test_data[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

y_test = test_data['class']



def transform2(i):

    if i == 0:

        return 'Normal'

    if i == 1:

        return 'Hernia'

    if i == 2:

        return 'Spondylolisthesis'
def dNN_2(X_train,y_train, X_test,dist='euclidian',q=2):

    pred = []

    if isinstance(X_test, np.ndarray):

        X_test=pd.DataFrame(X_test)

    if isinstance(X_train, np.ndarray):

        X_train=pd.DataFrame(X_train)

    vetMean = df2.reset_index().groupby( [ "class"],as_index=False ).agg({'pelvic_incidence': [np.mean],

                            'pelvic_tilt': [np.mean],

                            'lumbar_lordosis_angle': [np.mean],

                            'sacral_slope': [np.mean],

                            'pelvic_radius': [np.mean],

                            'degree_spondylolisthesis': [np.mean]

                          }, as_index=False )

    

    for i in range(len(X_test)):    

        # Distance calculation for test-point

        novadist = np.zeros(len(y_train))

        novadistc0 = np.zeros(len(y_train))

        novadistc1 = np.zeros(len(y_train))

        novadistc2 = np.zeros(len(y_train))

        

        if dist=='euclidian':

            for l in range(len(y_train)):

                for j1 in range(len(vetMean)):

                    novadistc0[l] = euclidian(vetMean.iloc[0,1:], X_test.iloc[i,:])

                    novadistc1[l] = euclidian(vetMean.iloc[1,1:], X_test.iloc[i,:])

                    novadistc2[l] = euclidian(vetMean.iloc[2,1:], X_test.iloc[i,:])

                    

                    novadist[l] = np.minimum(novadistc2[l]  , np.minimum( novadistc0[l],novadistc1[l]  )  )



            if novadistc0[l] <= novadistc1[l] and  novadistc0[l] <= novadistc2[l]:

                pred.append(0)

            elif novadistc1[l] <= novadistc0[l] and  novadistc1[l] <= novadistc2[l]:

                pred.append(1)

            else:

                novadistc2[l] <= novadistc0[l] and  novadistc2[l] <= novadistc1[l]

                pred.append(2)

        novadist = np.array([novadist, y_train])

    return pred
from sklearn.model_selection import KFold



x = all_X.values

y = all_y.values



scores = []

cv = KFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in cv.split(x):

    X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

    best_svr =  dNN_2(X_train,y_train, X_test)

    cm = confusion_matrix(y_test, best_svr)

    scores.append(cm)        



print('Confusion Matrix ',scores)

print('\n')

a = cm.shape

corrPred = 0

falsePred = 0



for row in range(a[0]):

    for c in range(a[1]):

        if row == c:

            corrPred +=cm[row,c]

        else:

            falsePred += cm[row,c]

print('True Pred: ', corrPred)

print('False Pred', falsePred)  