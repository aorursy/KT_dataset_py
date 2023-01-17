#importing libraries

import pkg_resources

import cmapPy

pkg_resources.get_distribution("cmapPy").version
#importing libraries

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns
#parsing the gct data using cmappy

from cmapPy.pandasGEXpress.parse import parse

data_row = parse('PAAD.gct',).data_df                     #expressional data

data_col = parse('PAAD.gct',col_meta_only=True).T         #informational data
#setting index and visualize the top 5 data row of expressional data

data_row=data_row.set_index(data_row.axes[0]) 

data_row.head(5)
#visualize the top 5 data row of informational data

data_col.head(5)
#shape of expressional data

data_col.shape
#checking null value and counting 

data_row.isnull().sum().sum()
#dropping null values

data_row = data_row.dropna()
#checking null value and counting after dropping null values

data_row.isnull().sum().sum()
#Slicing two informational columns

data_col=data_col.loc[['histological_type','histological_type_other']]
#visuslising sliced informational column

data_col.head()
#checking null value and counting row wise

for i in range(len(data_col.index)) :

    print("Nan in row ", i , " : " ,  data_col.iloc[i].isnull().sum())
#taking most occurence value for imputing in place of null

data_col.loc['histological_type'].mode()
#filling null values by most occurence value 

data_col.loc['histological_type'].fillna('pancreas-adenocarcinoma ductal type',inplace=True)

data_col.loc['histological_type_other'].fillna('pancreas-adenocarcinoma ductal type',inplace=True)
#checking null value and counting 

data_col.isnull().sum().sum()
#visuslising sliced informational column

data_col.head()
#checking unique values in data

data_col.iloc[1,].unique()
dataset = data_row.transpose()
#visuslising expressional column

dataset.head()
#Fitting the PCA algorithm with our Data

pca = PCA().fit(dataset)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.grid()

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Plotting for comonents')

plt.show()
#applying pca to dataset

n_components=125

pca = PCA(n_components)

df = pca.fit_transform(dataset)
pca.explained_variance_ratio_.cumsum()
df.shape
#placing column name and generating dataframe

df1=pd.DataFrame(df,columns=['PCA'+str(i) for i in range(1,n_components+1)],index=None)
#visualize the top 5 row

df1.head(5)
df1.describe()
#scatter plot pca1 and pca2

plt.figure(figsize=(8,5))

plt.scatter(df1['PCA1'],df1['PCA2'])

plt.xlabel('PCA1')

plt.ylabel('PCA2')

plt.title('scatter plot pca1 and pca2')

plt.grid()

plt.show()
df2=df1.copy()
#appending expressinal label to data

df2['label1']= list(data_col.iloc[0])

df2['label2']= list(data_col.iloc[1])
df2.head(5)
plt.figure(figsize=(8,5))

ax=sns.scatterplot(df2['PCA1'],df2['PCA2'],hue=df2['label2'])

plt.xlabel('PCA1')

plt.ylabel('PCA2')

plt.title('scatter plot pca1 and pca2')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid()

plt.show()
g =sns.FacetGrid(df2,hue='label2',height=9)

g.map(plt.scatter,'PCA1', 'PCA2').add_legend()

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df2['label2'])
def myplot(score,coeff,labels,y):

    xs = score[:,0]

    ys = score[:,1]

    n = coeff.shape[0]

    scalex = 1.0/(xs.max() - xs.min())

    scaley = 1.0/(ys.max() - ys.min())

    from matplotlib.pyplot import figure

    figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    plt.scatter(xs * scalex,ys * scaley, c = y)

    for i in range(n):

        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)

        if labels is None:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')

        else:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

    

    plt.xlim(-1,1)

    plt.ylim(-1,1)

    plt.xlabel("PC{}".format(1))

    plt.ylabel("PC{}".format(2))

    plt.grid()
myplot(df[:,0:2],np.transpose(pca.components_[0:2, :]),df2['label2'].values,y)

plt.show()
#reading  Type 1 IFN signature

ifn = pd.read_csv('type1_IFN.txt',header=None)
ifn[0].values
#concat informational column and expressional column by condition

new_data =pd.concat([data_row.loc[ifn[0]],data_col.drop('histological_type_other')])
#drop NaN value

hx=new_data.dropna()
hx.head(5)
hx.tail(5)
#transposing 

X=hx.T
#slicing the values

X.iloc[:,1:].values
#taking groupby on histological_type

x1=X.groupby('histological_type').sum()/X.groupby('histological_type').count()
x1.head(5)
#setting up the dataframe

x2=pd.DataFrame(x1,index=None, columns=None).T
#visualize the top 5 row

x2.head()
#plotting bar plot of each inferons present in histological type

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 24}



plt.rc('font', **font)



ax=x2.plot.bar(figsize = (40, 20))

# for p in ax.patches:

#     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.xlabel('Type 1 IFN signature')

plt.ylabel('score on each histological type')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

plt.tight_layout()

plt.show()
#plotting bar plot of histological type and their correspnding inferons score

ax=x2.T.plot.bar(figsize = (40, 20),rot=0)

# for p in ax.patches:

#     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.xlabel('Histological type')

plt.ylabel('Score on each histological type')

plt.tight_layout()

plt.show()