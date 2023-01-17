import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
%matplotlib inline
dataset=pd.read_csv("../input/dataset_treino.csv")
dataset.head()

#dataset.drop(['id'],axis=1,inplace=True)
dataset.isnull().sum()
count=0
for i in dataset.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total de registros com valores faltantes é', count)
print('Representando',round((count/len(dataset.index))*100), '% de todo o conjunto de dados.')
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribuição",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
draw_histograms(dataset,dataset.columns,6,3)
dataset.classe.value_counts()
sn.countplot(x='classe',data=dataset)
sn.pairplot(data=dataset)
dataset.describe()
from collections import Counter
# Outlier detection 
def detect_outliers(df,n,features):
    outlier_indices = []
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col],25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index       
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v >= n )
    return multiple_outliers   

# detect outliers 
Outliers_to_drop = detect_outliers(dataset,2,["num_gestacoes","glicose","pressao_sanguinea","pressao_sanguinea", "grossura_pele", "insulina", "bmi", "indice_historico", "idade"])

# Show the outliers
print("Outliers count: ", len(Outliers_to_drop))
dataset.loc[Outliers_to_drop]
import seaborn as sns
sns.set_style("darkgrid")

fig, axes = plt.subplots(2,4, figsize = (14,8), sharex=False, sharey=False)
axes = axes.ravel()
cols = ["num_gestacoes","glicose","pressao_sanguinea", "grossura_pele", "insulina", "bmi", "indice_historico", "idade"]
for i in range(len(cols)):
    sns.boxplot(y=cols[i],data=dataset, ax=axes[i])
plt.tight_layout()