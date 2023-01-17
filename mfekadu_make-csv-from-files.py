# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ww_filenames = os.listdir("../input/all_data/Wake Word")

nww_filenames = os.listdir("../input/all_data/Not Wake Word")



print("here is an example from ww_filenames:")

print("\t", ww_filenames[0], "\n")



print("here is an example from nww_filenames:")

print("\t", nww_filenames[0], "\n")
ww_df = pd.DataFrame(ww_filenames)

nww_df = pd.DataFrame(nww_filenames)



ww_df.columns = ['filenames']

nww_df.columns = ['filenames']
ww_df.head()
nww_df.head()
def parse_gender(row):

    feature_list = row.split("_")

    return feature_list[1]



ww_df['gender'] = ww_df['filenames'].apply(lambda row: parse_gender(row))
ww_df.head()
# isNWW=False example

# >>> "ww_m_iss-hushed_bedroom_q_fekadu_michael_05082019072955_fekadu.wav"



# isNWW=True example

# >>> "notww_rhyme-business_bedroom_q_04272019213505_ewenike.wav"



def parse_gender(row, isNWW=False):

    feature_list = row.split("_")

    return feature_list[1] if isNWW == False else None



def parse_description(row, isNWW=False):

    feature_list = row.split("_")

    return feature_list[2] if isNWW == False else feature_list[1]



def parse_location(row, isNWW=False):

    feature_list = row.split("_")

    return feature_list[3] if isNWW == False else feature_list[2]



def parse_loudness(row, isNWW=False):

    feature_list = row.split("_")

#     print(feature_list)

#     assert feature_list[4] in ['m','q','l'] or feature_list[3] in ['m','q','l'], "OOPS"

    return feature_list[4] if isNWW == False else feature_list[3]



def parse_full_name(row, isNWW=False):

    feature_list = row.split("_")

    if (isNWW):

        return None

    return feature_list[5] + "-" + feature_list[6] 



def parse_timestamp(row, isNWW=False):

    feature_list = row.split("_")

    return feature_list[7] if isNWW == False else feature_list[4]



def parse_nametag(row, isNWW=False):

    feature_list = row.split("_")

    return feature_list[8] if isNWW == False else feature_list[5]



def parse_isWW(row, isNWW=False):

    feature_list = row.split("_")

    return True if feature_list[0] == "ww" else False



ww_df['gender'] = ww_df['filenames'].apply(lambda row: parse_gender(row))

nww_df['gender'] = nww_df['filenames'].apply(lambda row: parse_gender(row, isNWW=True))



ww_df['isWW'] = ww_df['filenames'].apply(lambda row: parse_isWW(row))

ww_df['description'] = ww_df['filenames'].apply(lambda row: parse_description(row))

ww_df['location'] = ww_df['filenames'].apply(lambda row: parse_location(row))

ww_df['loudness'] = ww_df['filenames'].apply(lambda row: parse_loudness(row))

ww_df['full_name'] = ww_df['filenames'].apply(lambda row: parse_full_name(row))

ww_df['timestamp'] = ww_df['filenames'].apply(lambda row: parse_timestamp(row))

ww_df['nametag'] = ww_df['filenames'].apply(lambda row: parse_nametag(row))



nww_df['isWW'] = nww_df['filenames'].apply(lambda row: parse_isWW(row, isNWW=True))

nww_df['description'] = nww_df['filenames'].apply(lambda row: parse_description(row, isNWW=True))

nww_df['location'] = nww_df['filenames'].apply(lambda row: parse_location(row, isNWW=True))

nww_df['loudness'] = nww_df['filenames'].apply(lambda row: parse_loudness(row, isNWW=True))

nww_df['full_name'] = nww_df['filenames'].apply(lambda row: parse_full_name(row, isNWW=True))

nww_df['timestamp'] = nww_df['filenames'].apply(lambda row: parse_timestamp(row, isNWW=True))

nww_df['nametag'] = nww_df['filenames'].apply(lambda row: parse_nametag(row, isNWW=True))
ww_df.head()
nww_df.head()
all_df = pd.concat([ww_df, nww_df], ignore_index=True)

all_df.head()
ww_df.to_csv("wake_word.csv")

nww_df.to_csv("not_wake_word.csv")

all_df.to_csv("all_data.csv")
os.listdir()
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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

df1 = pd.read_csv('all_data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'all_data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
plotPerColumnDistribution(df1, 10, 5)
bad_loudness_data_mask = df1['loudness'] == '197-104'

bad_gender_data_mask = df1['gender'] == 'male'

all_bad_data_mask = bad_gender_data_mask | bad_loudness_data_mask



print(df1[all_bad_data_mask]['filenames'].values)
# good_data is ~bad_data (bitwise not)

good_data_mask = ~all_bad_data_mask

# new DataFrame with only good data

df1 = df1[good_data_mask]

df1
plotPerColumnDistribution(df1, 10, 5)