from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
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

# imdb-movies.csv has 10866 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/imdb-movies.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'imdb-movies.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
# df1.head(5)

# df1.columns

#df1["revenue"]

df1.info()
#Start with an empty dictionary

genreGrouping = {}



#Group by genres

print("Genre classification started...")

for genres,df in df1.groupby(["genres"]):

    #Split genres by '|'

    for g in genres.split(('|')):

        #For each genre, if it is not in dictionary, insert it with an empty list as value

        genreGrouping.setdefault(g,[])

        #Split companies by splitting company on '|'

        #Append it to the genre

        genreGrouping[g] += [c for p in df["production_companies"] for c in str(p).split('|')]

print("Genre classification completed")
print(len(genreGrouping.keys()))


#Remove duplicates from each genres list

listOfAllProductionCompaniedFrmGenre = []

for k,v in genreGrouping.items():

    genreGrouping[k] = v

    listOfAllProductionCompaniedFrmGenre.append(v)

# Creating a vector with production companies in columns and genres as rows with values as its count

# print(listOfAllProductionCompaniedFrmGenre)

# the list all production companies are list of ist flatten the list and then apply set to it

#     the flattening can be avoided if any means are found out while it is getting appended to the list

listOfAllProductionCompaniedFrmGenre = [company for sublist in listOfAllProductionCompaniedFrmGenre for company in sublist]

# print(len(list(set(listOfAllProductionCompaniedFrmGenre))))

listOfCompaines = list(set(listOfAllProductionCompaniedFrmGenre))

# create a dictionary to store the frequency vector

print("Freaquency vector genreation started...")

genreVector = {}

for k,v in genreGrouping.items():    

    companyFeaquencyList = []     

    for uniqueCompany in listOfCompaines:

        companyFeaquencyList.append(v.count(uniqueCompany))

    genreVector[k]=companyFeaquencyList

print("Freaquency vector genreation completed")    

# print(genreVector)

# converting to dataframe

df_genre= pd.DataFrame.from_dict(genreVector)

#     genreVector[k] = 
print(len(df_genre.columns))
df_working= pd.DataFrame(genreVector,index = listOfCompaines)

df_working

# df_working.sort_values(by="Action",ascending= False)["Action"].hist(figsize=(20,4))

df_working.sort_values(by="Action",ascending= False)["Action"]
#Group by genres

# for genres,df in df1.groupby(["genres"]):

listOfCompanies =[]

# print(df1.info())

from collections import Counter



for companies in df1[pd.notnull(df1['production_companies'])]:

    for company in companies.split("|"):

        listOfCompanies.append(company)

print(listOfCompaines)

print("Length of total list:"+str(len(listOfCompanies)))

print("Length of unique list:"+str(len(set(listOfCompaines))))

print(Counter(listOfCompaines).most_common(20))
#Start with an empty dictionary

genreGrouping = {}

from collections import Counter

#Group by genres

for genres,df in df1.groupby(["genres"]):

    #Split genres by '|'

    for g in genres.split(('|')):

        #For each genre, if it is not in dictionary, insert it with an empty list as value

        genreGrouping.setdefault(g,[])

        #Split companies by splitting company on '|'

        #Append it to the genre

        genreGrouping[g] += [c for p in df["production_companies"] for c in str(p).split('|')]



#Count top 10 companies for each genre 

genreComCount20 = {}

for k,v in genreGrouping.items():

    genreComCount20[k] = Counter(v).most_common(20)



import matplotlib.pylab as plt

fig = plt.figure(figsize=(50,50))

# fig.set_size_inches(18.5, 5)

# subplots(constrained_layout=True)

for index,genre in enumerate(list(genreGrouping.keys())):

    print(index)

    ax = plt.subplot(7,3,index+1)

    pd.DataFrame(genreComCount20[genre], columns=['company',genre]).set_index('company').plot(kind='bar',ax = ax)

    plt.tight_layout()

plt.show()
genreComCount10['Adventure']
listA = ["a"]

print(listA.count('a'))
df1.groupby(["genre"])["genre"]
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)