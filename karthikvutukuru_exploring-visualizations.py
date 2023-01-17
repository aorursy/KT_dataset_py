# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ssdf = pd.read_csv('/kaggle/input/google-trends-data/20150626_SameSexMarriage.csv')

ssdf.head()
ssdf.info()
ssdf= ssdf.rename(columns={ "Search interest in same sex marriage by city on 6/26" : "City",

                     "Unnamed: 1": "State",

                     "Unnamed: 2": "Country",

                     "Unnamed: 3": "Search Interest",

                     "Unnamed: 4" : "Time"})
ssdf = ssdf[2:]

ssdf.info()
ssdf['Search Interest'] = ssdf['Search Interest'].astype(float)

ssdf['State'] =  ssdf['State'].astype(str)

ssdf['Country'] =  ssdf['Country'].astype(str)

ssdf['City'] =  ssdf['City'].astype(str)
ssdf.info()
from wordcloud import WordCloud, ImageColorGenerator



def gen_wordcloud(df, col):

    text = " ".join(str(each) for each in df[col])

    wordcloud = WordCloud(colormap='Set3', background_color='black').generate(text)

    plt.figure(figsize=(20,20))

    # Display the generated image:

    plt.imshow(wordcloud, interpolation='Bilinear')

    plt.axis("off")

    plt.figure(1,figsize=(12, 12))

    plt.show()
gen_wordcloud(ssdf, "State")
gen_wordcloud(ssdf, "City")
ssdf[["City", "Search Interest"]].groupby("City").agg('mean').sort_values("Search Interest", ascending=False)
ssdf[["State", "Search Interest"]].groupby("State").agg('mean').sort_values("Search Interest", ascending=False)
state_mask = ssdf['State'] == 'Maine'

interest_mask_gt5 = ssdf['Search Interest'] > 5.0

interest_mask_le10 = ssdf['Search Interest'] < 10.0



ssdf[state_mask & interest_mask_gt5 & interest_mask_le10]
searchInterest_by_State = ssdf[['State','Search Interest']].groupby('State', as_index=False).sum()

searchInterest_by_State
searchInterest_by_State.plot(x="State", y="Search Interest")
import seaborn as sns

sns.barplot(x="Search Interest", y="State", data=searchInterest_by_State, color="b")
import matplotlib.pyplot as plt

g = sns.countplot(x="State", data=ssdf, color='b')

ax = g.axes

ax.set_xticklabels(ssdf["State"], rotation=90)

ax.set_ylabel("Search Interest")

plt.show()
mean_interest_by_states = ssdf[['State', 'Search Interest']].groupby('State').agg('mean')

mean_interest_by_states.reset_index(inplace=True)

g= sns.barplot(x="State", y="Search Interest", data=mean_interest_by_states)

ax = g.axes

ax.set_xticklabels(ssdf["State"], rotation=90)

ax.set_ylabel("Search Interest")

plt.show()
pct_of_state_interest= ssdf[['State', 'Search Interest']].groupby('State').agg('sum')/ssdf['Search Interest'].sum()

pct_of_state_interest.reset_index(inplace=True)

pct_of_state_interest
g= sns.scatterplot(x="State", y="Search Interest", data=pct_of_state_interest, size="Search Interest")

ax = g.axes

ax.set_xticklabels(ssdf["State"], rotation=90)

ax.set_ylabel("Search Interest")

plt.show()
import folium

map = folium.Map(location=[48, -102], zoom_start=3)

map
city_states = ssdf[['State', 'City']]

city_states
fifa_data = pd.read_csv("/kaggle/input/google-trends-data/20150528_FIFA.csv", encoding='latin-1')

fifa_data
fifa_data.rename(columns={"country":"Country",

                         "Rank of interest in Fifa, May 27-28, 2015": "Rank"}, inplace=True)

fifa_data
rank= fifa_data["Rank"] <= 25

fifa_data[rank].style.hide_index()
country_filter = fifa_data["Country"].str.contains("u")

fifa_data[country_filter]
curry_james = pd.read_csv("/kaggle/input/google-trends-data/20150528_CurryVsJames.csv", header=1)

curry_james.head()
curry_james["Week"] = pd.to_datetime(curry_james["Week"]).sort_values()

curry_james.info()
curry_james.head()
# Change Index to Week - datetime

curry_james = curry_james.set_index("Week")

curry_james
# Print for a month March

curry_james.loc['2015-03']
# Print for months of April and May

curry_james.loc['2015-04':'2015-05']
curry_james.plot()
cumsumcj = curry_james.cumsum(axis=0)

cumsumcj
def get_cumsum(x):

    y = []

    k = 0

    for i, j in enumerate(x):

        y.append(j + k)

        k = k+ j

    return y



%time get_cumsum(curry_james["LeBron James"])

%time get_cumsum(curry_james["Stephen Curry"])

%time curry_james.cumsum(axis=0)
curry_james.max(axis=1)
cumsumcj.plot()
ukdf = pd.read_csv('/kaggle/input/google-trends-data/20150430_UKDebate.csv')

ukdf.dataframeName='20150430_UKDebate.csv'

nRow, nCol = ukdf.shape

print("Shape of the dataframe", ukdf.shape)
ukdf.head()
ukdf[:1000].nunique()
ukdf[:5]
for col in ukdf.columns:

    if ukdf[col].dtype == 'object':

        print(col)
ukdf.iloc[:,0].dtype

ukdf.iloc[:,6].dtype
def plotPerColumnDistribution(df, nGraphshown, nGraphsRow, maxsum=100):

    nuniq = df.nunique()

    df = df[[col for col in df if nuniq[col] > 1 and nuniq[col] < maxsum]]

    nRow, nCol = df.shape

    colNames = list(df)

    nGRow = (nCol + nGraphsRow - 1) / nGraphsRow

    plt.figure(num = None, figsize=( 6 * nGraphsRow, 8 * nGRow), dpi=80, facecolor='w', edgecolor='k')

    for i in range(min(nCol, nGraphshown)):

        plt.subplot(nGRow, nGraphsRow, i+1)

        colDf = df.iloc[:, i]

        if colDf.dtype != 'O':

            colDf.hist()

        else:

            valueCounts = colDf.value_counts()

            valueCounts.plot.bar()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{colNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
ukdf_sample = ukdf[:1000]

plotPerColumnDistribution(ukdf_sample, 10, 5)
def plotCorrelationMatrix(df, graphWidth):

    fileName = df.dataframeName

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:

        print('Only 1 column, not possible to plot')

        return

    corr = df.corr()

    plt.figure(figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum=1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {fileName}', fontsize=15)

    plt.show()

    
plotCorrelationMatrix(ukdf, 10)
# Scatter Plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df[[col for col in df if df[col].dtype == 'float64']]

    df = df.dropna('columns')

    colNames = list(df)

    if len(colNames) > 10:

        colNames = colNames[:10]

    df = df[colNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

        

    

plotScatterMatrix(ukdf, 9, 10)

gen_wordcloud(ukdf, "city")