from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
df_movies=pd.read_csv("/kaggle/input/movies.csv")
df_movies['year'] = df_movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
df_movies['year'] = df_movies.year.str.extract('(\d\d\d\d)',expand=False)
df_movies['title'] = df_movies.title.str.replace('(\(\d\d\d\d\))', '')
df_movies['title'] = df_movies['title'].apply(lambda x: x.strip())
df_movies.head()
df_movies["genres"]=df_movies["genres"].apply(lambda x:x.split("|"))
df_movies["title"]=df_movies["title"].apply(lambda x:x.replace(x,x.lower()))
df_moviegenre=df_movies.copy()
for index, row in df_movies.iterrows():
    for genre in row['genres']:
        df_moviegenre.at[index, genre] = 1
df_moviegenre= df_moviegenre.fillna(0)
#df_moviegenre.head()
df_ratings=pd.read_csv("/kaggle/input/ratings.csv")
df_ratings=df_ratings.drop("timestamp",1)
#df_ratings.head()
new=[]
print("If movie starts with 'the' enter movie name as '(name,the)'")
print("NUMBER OF MOVIES WATCHED RECENTLY =")
for i in range(int(input())):
    dc={}
    print("MOVIE WATCHED RECENTLY =")
    dc["title"]=input()
    print("GIVEABLE RATING =")
    dc["rating"]=float(input())
    new.append(dc)
usermovies = pd.DataFrame(new)
#usermovies
inputid=df_movies[df_movies["title"].isin(usermovies["title"].tolist())]
usermovies=pd.merge(inputid,usermovies)
#usermovies
usermovies=usermovies.drop("genres",1).drop("year",1)
#usermovies
movies = df_moviegenre[df_moviegenre['movieId'].isin(usermovies['movieId'].tolist())]
movies=movies.drop("year",1).drop("title",1).drop("genres",1).drop("movieId",1)
movies=movies.reset_index(drop=True)
movies=movies.transpose().dot(usermovies["rating"])
genre = df_moviegenre.set_index(df_moviegenre['movieId'])
genre = genre.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
df_movierecommend = ((genre*movies).sum(axis=1))/(movies.sum())
df_movierecommend=df_movierecommend.sort_values(ascending=False)
df_movies.loc[df_movies['movieId'].isin(df_movierecommend.head(20).keys())]
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
# links.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/links.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'links.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 9, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file
# movies.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/movies.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'movies.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
nRowsRead = 1000 # specify 'None' if want to read whole file
# ratings.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/ratings.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'ratings.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')
df3.head(5)
plotPerColumnDistribution(df3, 10, 5)
plotCorrelationMatrix(df3, 8)
plotScatterMatrix(df3, 12, 10)