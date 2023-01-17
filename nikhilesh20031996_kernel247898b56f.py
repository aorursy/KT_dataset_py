import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import plotly.figure_factory as ff
data = pd.read_csv("../input/Songs_Movie_Rating.csv" , sep=";" , encoding ='latin1')
x = data["EncodedGenre"]

y = data["User-Rating"]
data.plot(x='EncodedGenre', y='User-Rating', style='o')  

plt.title('Genre vs Rating')  

plt.xlabel('EncodedGenre')  

plt.ylabel('Genre')  

plt.show()
#since the data is heterosedastic i.e. one value of x have multiple value of y. In this case the "Bollywood" Genre have multiple ratings.

#hence we can't aplly linear regression.
groupedData = data.groupby("Genre")
plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(groupedData.get_group("BollywoodDance")["User-Rating"])
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
newData = data[['Song-Name', 'Singer/Artists', 'Genre', 'Album/Movie', 'User-Rating']]

plotPerColumnDistribution(newData, 10, 5)