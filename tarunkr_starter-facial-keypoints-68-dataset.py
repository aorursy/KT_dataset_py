from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
df_keypoints = pd.read_csv('../input/training_frames_keypoints.csv')

df_keypoints.head(3)
file_names = list(df_keypoints['Unnamed: 0'])



keypoints = np.asarray(df_keypoints.iloc[:,1:]).reshape([-1, 68, 2])



keypoints_dict = {f: np.asarray(df_keypoints[df_keypoints["Unnamed: 0"] == f]\

                                .iloc[:,1:])\

                                .reshape([ 68, 2])

                                for i, f in enumerate(file_names)}

keypoints_dict["Luis_Fonsi_21.jpg"].shape
random = np.random.randint(0,1000,[16])

rows = np.ceil(random.shape[0]/5)

cols = 4

size = 6

f = plt.figure(figsize = (cols*size,rows*size))



for i,rand in enumerate(random):

    f.add_subplot(rows,cols,i+1)



    img = cv2.cvtColor(cv2.imread("../input/training/" + df_keypoints.iloc[rand][0]),cv2.COLOR_BGR2RGB)

    points = np.asarray(df_keypoints.iloc[rand,1:]).reshape([-1,2])



    plt.scatter(points[:,0], points[:,1],s=3,c='r')

    plt.imshow(img)

    plt.axis('off')

plt.show()
def load_data(path):

    df_keypoints = pd.read_csv(path)

    file_names = list(df_keypoints['Unnamed: 0'])



    keypoints = np.asarray(df_keypoints.iloc[:,1:]).reshape([-1, 68, 2])



    keypoints_dict = {f: np.asarray(df_keypoints[df_keypoints["Unnamed: 0"] == f]\

                                  .iloc[:,1:])\

                                  .reshape([ 68, 2])

                                  for i, f in enumerate(file_names)}

    np.random.shuffle(file_names)

    return file_names, keypoints_dict
# Load Training data

file_names_train, keypoints_train = load_data('../input/training_frames_keypoints.csv')



# Load Test data

file_names_test, keypoints_test = load_data('../input/test_frames_keypoints.csv')
print("Train Samples:", len(file_names_train), "\nTest Samples: ", len(file_names_test))
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

# test_frames_keypoints.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/test_frames_keypoints.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'test_frames_keypoints.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 24)
plotScatterMatrix(df1, 20, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file

# training_frames_keypoints.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/training_frames_keypoints.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'training_frames_keypoints.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 24)
plotScatterMatrix(df2, 20, 10)