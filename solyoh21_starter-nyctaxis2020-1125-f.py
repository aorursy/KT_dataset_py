from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

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

# sample_submission.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/sample_submission.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'sample_submission.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
nRowsRead = 1000 # specify 'None' if want to read whole file

# test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/test.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'test.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 15, 10)


{

   "schemaVersion": 2,

   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",

   "config": {

      "mediaType": "application/vnd.docker.container.image.v1+json",

      "size": 31231,

      "digest": "sha256:79f52292b1d0c079b84bc79d002947be409a09b15a1320355a4de834f57b2ee8"

   },

   "layers": [

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 45339314,

         "digest": "sha256:c5e155d5a1d130a7f8a3e24cee0d9e1349bff13f90ec6a941478e558fde53c14"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 95104141,

         "digest": "sha256:86534c0d13b7196a49d52a65548f524b744d48ccaf89454659637bee4811d312"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1571501372,

         "digest": "sha256:5764e90b1fae3f6050c1b56958da5e94c0d0c2a5211955f579958fcbe6a679fd"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1083072,

         "digest": "sha256:ba67f7304613606a1d577e2fc5b1e6bb14b764bcc8d07021779173bcc6a8d4b6"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 526,

         "digest": "sha256:19abed793cf0a9952e1a08188dbe2627ed25836757d0e0e3150d5c8328562b4e"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 458,

         "digest": "sha256:df204f1f292ae58e4c4141a950fad3aa190d87ed9cc3d364ca6aa1e7e0b73e45"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 13119161,

         "digest": "sha256:1f7809135d9076fb9ed8ee186e93e3352c861489e0e80804f79b2b5634b456dd"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 555884253,

         "digest": "sha256:03a365d6218dbe33f5b17d305f5e25e412f7b83b38394c5818bde053a542f11b"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 102870915,

         "digest": "sha256:00e3d0b7af78551716541d2076836df5594948d5d98f04f382158ef26eb7c907"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 95925388,

         "digest": "sha256:59782fefadba835c1e83cecdd73dc8e81121eae05ba58d3628a44a1c607feb6e"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 142481172,

         "digest": "sha256:f81b01cf2c3f02e153a71704cc5ffe6102757fb7c2fcafc107a64581b0f6dc10"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1128076783,

         "digest": "sha256:f08bbb5c2bce948f0d12eea15c88aad45cdd5b804b71bee5a2cfdbf53c7ec254"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 444800302,

         "digest": "sha256:b831800c60a36c21033cb6e85f0bd3a5f5c9d96b2fa2797d0e8d4c50598180b8"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 157365696,

         "digest": "sha256:6d354ec67fa4ccf30460efadef27d48edf9599348cbab789c388f1d3a7fee232"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 63237273,

         "digest": "sha256:464f9b4eca5cdf36756cf0bef8c76b23344d0e975667becb743e8d6b9019c3cd"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 427820542,

         "digest": "sha256:6c1f6bcbc63b982a86dc94301c2996505cec571938c60a434b3de196498c7b89"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 44581800,

         "digest": "sha256:c0a8110c6fede3cf54fa00a1b4e2fcb136d00b3cf490f658ec6d596e313c986e"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 127637178,

         "digest": "sha256:c25df885c8dea40780f5b479bb6c7be924763399a21fa46c204d5bfac45056bd"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 956429221,

         "digest": "sha256:7c1d98590e22f78a1b820f89b6ce245321437639957264e628b4abf4862e1223"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 586276809,

         "digest": "sha256:aab720d802b7d006b679ac10df4a259b3556812dea4dfc52d6111db47fc41e62"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 21717560,

         "digest": "sha256:5ee4a4cda8613a3fb172a827143aadacb98128479a22a280495604f989bf4483"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 93512644,

         "digest": "sha256:c4699852e987bc3fe9adde2544ffa690ad52ebec229c20f7e4153b015ac238ff"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 19141,

         "digest": "sha256:8d93692c8dcecacb8aca746a868f53d0b0cf1207e08ced8ffb2134bb01c1f871"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 84125618,

         "digest": "sha256:57c74d175611802a57531be97d19f586dc9cd810a5490eab04fd40b648312ead"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 3261,

         "digest": "sha256:1ac7a265bf03308e06e9cad7e77d12b22ca8bc6b7791d46398d95977e0042574"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 2162,

         "digest": "sha256:1b4a5be69a4439f3de72877b7d408400e3aa0b4c37e9c70c4490b480bce682c0"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 1270,

         "digest": "sha256:648046d6f6c28a42a39c9e68a9da90ccdabbd1ecfd0be77941114df4eb2406a4"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 644,

         "digest": "sha256:19a794f6956d460edfe74d5562d44366a7cf8bd46d83f408f1bf3c46e7282464"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 2052,

         "digest": "sha256:880f92e310c2e03c28c5db85b342069b1a56cd13de7998ae52f699829774f075"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 875,

         "digest": "sha256:cad389727d6cd1696ed7e91b70eedd4c86fd30babb648e7be6cc1639582b0928"

      },

      {

         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",

         "size": 214,

         "digest": "sha256:c873da9a657a590abeae80bd3c0d0d87a6bfdfaf1d3873a0f210760a4050d6db"

      }

   ]

}
