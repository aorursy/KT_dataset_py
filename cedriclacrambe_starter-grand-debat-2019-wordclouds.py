from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud  

from stop_words import get_stop_words

import nltk

import spacy.lang
import spacy.lang.fr

stopwords_fr_set=set(nltk.corpus.stopwords.words('french'))

stopwords_fr_set.update(get_stop_words('fr'))

stopwords_fr_set.update(spacy.lang.fr.stop_words.STOP_WORDS)

stopwords_fr_set.update(["c'est","j'ai","n'est","n'ait","ca","ça","sais","jamais","chose","ex"])

stopwords_fr_set

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

def plot_wordcloud(s,title=""):

    cloud = WordCloud(

                   width=1500,

                   height=1500,

                   min_font_size=0.18,

                   background_color='black',

                   colormap='plasma',

                    stopwords =stopwords_fr_set

                   ).generate(s)

    plt.figure(figsize=(25,20))

    plt.title(title)

    plt.imshow(cloud)

    plt.axis('off')

    

def plot_df_debat_wordcloud(df,lmax=2*1024**2,title=""):



    df=df.sample(frac=1).iloc[:,11:]

    s="\n".join(t for t in df.fillna("").values.flatten() if len(t)>6)

    

    plot_wordcloud(s[:lmax],title=title)
df_democatie=pd.read_csv("../input/DEMOCRATIE_ET_CITOYENNETE.csv.xz",low_memory=False)

df_democatie.info()

plot_df_debat_wordcloud(df_democatie,title="Démocatie")

df_democatie.head()
df_fiscalite=pd.read_csv("../input/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.csv.xz",low_memory=False)

df_fiscalite.info()

plot_df_debat_wordcloud(df_fiscalite,title="Fiscalité")

df_fiscalite.head()
df_ecologie=pd.read_csv("../input/LA_TRANSITION_ECOLOGIQUE.csv.xz",low_memory=False)

df_ecologie.info()

plot_df_debat_wordcloud(df_ecologie,title="Ecologie")

df_ecologie.head()
df_etat=pd.read_csv("../input/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv.xz",low_memory=False)

df_etat.info()

plot_df_debat_wordcloud(df_etat,title="ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS")

df_etat.head()
import dask.dataframe as dd

df_vrai_debat=dd.read_csv('../input/le vrai debat/le vrai debat/*.csv',include_path_column=True,dtype=object).compute()



plot_wordcloud(df_vrai_debat.sample(frac=1).contribution_versions_bodyText.str.cat(sep="\n\n")[:4*1024**2]  ,title="vrai débat")

df_vrai_debat.info()

df_vrai_debat.head()