!pip install requests-futures

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

from requests_futures.sessions import FuturesSession

from joblib import Memory

import cv2

import imageio
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

# imagenet_bbox.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/imagenet_bbox.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'imagenet_bbox.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 12, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file

# imagenet_dd_person_merge.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/imagenet_dd_person_merge.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'imagenet_dd_person_merge.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
nRowsRead = 1000 # specify 'None' if want to read whole file

# imagenet_dd_person_url.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df3 = pd.read_csv('/kaggle/input/imagenet_dd_person_url.csv', delimiter=',')# nrows = nRowsRead)

df3.dataframeName = 'imagenet_dd_person_url.csv'

nRow, nCol = df3.shape

print(f'There are {nRow} rows and {nCol} columns')
df3.head(5)
session = FuturesSession()





df3_1=df3.sample(frac=1)
# df3_1["req"]=df3_1.url.apply(lambda x:session.get(x,timeout =2) )

# df3_1
def detcode(f):

    if f.exception() is not None:

        return 0

    else :

        return f.result().status_code

# df3_1["status"]=df3_1.req.apply(detcode)

# df3_1
# (df3_1.status==200).describe()
plotPerColumnDistribution(df3, 10, 5)


df_wnid=pd.read_csv("../input/imagenet_person_wnid.csv")[["wnid","lemmas"]]#.set_index("wnid")

df_wnid




df4=pd.read_csv("../input/imagenet_person_mappings.csv")

df4=df4.merge(pd.read_csv("../input/imagenet_bbox.csv"),on=["id"],how="left")

df4["wnid"]=df4.wnid_x

df4.xmin.fillna(0,inplace=True)

df4.ymin.fillna(0,inplace=True)

df4.xmax.fillna(1,inplace=True)

df4.ymax.fillna(1,inplace=True)

df4=df4.merge(df_wnid)

del df4["wnid_x"]

del df4["wnid_y"]

df4



!mkdir /tmp/imgcache



memory =Memory(location="/tmp/imgcache", backend='local',mmap_mode="r+",verbose=0)
@memory.cache()

def imgget(url):

    img= imageio.imread(t.url)

    return img

    

def getimg(df=df4,marging=0):

    for t in df.sample(frac=1).itertuples():

        try:

            img= imageio.imread(t.url)

        except:

            continue

        w,h,_=img.shape

        

        xmin=int(t.xmin*w-w*marging)

        xmax=np.ceil(t.xmax*w+w*marging)

        ymin=int(t.ymin*h-h*marging)

        ymax=np.ceil(t.ymax*h+h*marging)



        if xmin<0:

            xmin=0

        if ymin<0:

            ymin=0

        if xmax>w:

            xmax=w

        if ymax>h:

            ymax=h

        xmin,xmax,ymin,ymax=np.array((xmin,xmax,ymin,ymax),dtype=np.int)

    #     img1=img[xmin:xmax,ymin:ymax]



    #     img3=cv2.rectangle(img.copy(),color=(255,0,0),pt1=( xmin,ymin),pt2=(xmax,ymax),thickness=5)

    #     plt.imshow(img3)

        yield img,xmin,xmax,ymin,ymax,t.wnid

    
fig=plt.figure(figsize=(20,20))

for n,(img,xmin,xmax,ymin,ymax,wnid) in enumerate(getimg(df4),start=1):

    plt.subplot(3,3,n)

    img1=cv2.rectangle(img.copy(),color=(255,0,0),pt1=( xmin,ymin),pt2=(xmax,ymax),thickness=1)

    plt.imshow(img1)

    txt=",".join(df_wnid[df_wnid.wnid==wnid].lemmas)

    if txt!='':

        txt=wnid+": "+txt 

    else:

        txt=wnid

    plt.title(txt)

    plt.axis("off")

    if n>8:

        break

fig=plt.figure(figsize=(20,20))

for n,(img,xmin,xmax,ymin,ymax,wnid) in enumerate(getimg(df4,marging=0.5),start=1):

    plt.subplot(3,3,n)

    img1=cv2.rectangle(img.copy(),color=(255,0,0),pt1=( xmin,ymin),pt2=(xmax,ymax),thickness=5)

    plt.imshow(img1)

    txt=",".join(df_wnid[df_wnid.wnid==wnid].lemmas)

    if txt!='':

        txt=wnid+": "+txt 

    else:

        txt=wnid

    plt.title(txt)

    plt.axis("off")

    if n>8:

        break



df_wnid=df4[["wnid"]].groupby("wnid").size()

df_wnid.name="num"

df_wnid.sort_values(inplace=True,ascending=False,)

df_wnid=df_wnid.reset_index()

df_wnid.describe()



df_wnid=df_wnid.merge(pd.read_csv('../input/imagenet_person_wnid.csv'))[['wnid', 'num','wnid_offset', 'synset', 'lemmas', 'definition']].reset_index()

df_wnid.rename(columns={'index': 'class'}, inplace=True)

df_wnid.to_csv("classes.csv",index=False)

df_wnid.head(30)

df4_1=df4.merge(df_wnid[["class" ,	"wnid" 	]]).sample(frac=1).reset_index(drop=True)

del df4_1["lemmas"]





df4_1


lval=np.ceil(len(df4_1)*0.2)

lval=int(lval)

lval
df4_1["split"]="train"

df4_1.loc[:lval,"split"]="val"

df4_1.to_csv("dataset.csv",index=False)

df4_1