!pip install seaborn --upgrade
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
sns.set_style('whitegrid')
import scipy.stats as stats
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/lish-moa/train_features.csv')
train_targets=pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_nonscored=pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
train.head()
train['cp_time']=train['cp_time'].astype(str)
train_targets.head()
#treatment features
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(x=train['cp_type'])
plt.subplot(1,3,2)
sns.countplot(x=train['cp_time'])
plt.subplot(1,3,3)
sns.countplot(x=train['cp_dose'])
np.random.seed(22)
cell_choice=np.random.choice(100,3)
gene_choice=np.random.choice(772,3)

plt.figure(figsize=(15,10))
for i in range(3):
    plt.subplot(2,3,i+1)
    sns.histplot(data=train, x='c-'+str(cell_choice[i]), color='orange', kde=True)
    plt.title('Distribution of '+'c-'+str(cell_choice[i]))
for i in range(3):
    plt.subplot(2,3,i+4)
    sns.histplot(data=train, x='g-'+str(gene_choice[i]), color='blue', kde=True)
    plt.title('Distribution of '+'g-'+str(gene_choice[i]))
    

#additional cell viability plots. 
cell_choice=np.random.choice(100,5)
plt.figure(figsize=(15,10))
for i in range(5):
    plt.subplot(2,3,i+1)
    sns.histplot(data=train[train['c-'+str(cell_choice[i])]<-2], x='c-'+str(cell_choice[i]), color='orange', bins=50)
    plt.title('Distribution of '+'c-'+str(cell_choice[i]))
cell_skews,cell_kurt,gene_skews,gene_kurt=[],[],[],[]
for i in range(100):
    cell_skews.append(stats.skew(train['c-'+str(i)]))
    cell_kurt.append(stats.kurtosis(train['c-'+str(i)]))
for i in range(772):
    gene_skews.append(stats.skew(train['g-'+str(i)]))
    gene_kurt.append(stats.kurtosis(train['g-'+str(i)]))
plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.histplot(x=cell_skews, color='orange', kde=True)
plt.title('Distibution of Skew for Cell Viability Features')
plt.subplot(2,2,2)
sns.histplot(x=gene_skews, color='blue', kde=True)
plt.title('Distibution of Skew for Gene Expression Features')
plt.subplot(2,2,3)
sns.histplot(x=cell_kurt, color='orange', kde=True)
plt.title('Distibution of Kurtosis for Cell Viability Features')
plt.subplot(2,2,4)
sns.histplot(x=gene_kurt, color='blue', kde=True)
plt.title('Distibution of Kurtosis for Gene Expression Features')
cell_std, gene_std=[],[]
for i in range(100):
    cell_std.append(train['c-'+str(i)].std())
for i in range(772):
    gene_std.append(train['g-'+str(i)].std())
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
sns.histplot(x=cell_std, color='orange', kde=True)
plt.title('Distribution of Std. Dev. of Cell Viability Features')
plt.subplot(1,2,2)
sns.histplot(x=gene_std, color='blue', kde=True)
plt.title('Distribution of Std. Dev. of Gene Expression Features')
plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.histplot(data=train, x='c-'+str(cell_skews.index(max(cell_skews))), color='orange')
plt.title('Most Skewed Cell Viability Feature')
plt.subplot(2,2,2)
sns.histplot(data=train, x='c-'+str(cell_kurt.index(max(cell_kurt))), color='orange')
plt.title('Highest Kurtosis Cell Viability Feature')
plt.subplot(2,2,3)
sns.histplot(data=train, x='g-'+str(gene_skews.index(max(gene_skews))), color='blue')
plt.title('Most Skewed Gene Expression Feature')
plt.subplot(2,2,4)
sns.histplot(data=train, x='g-'+str(gene_kurt.index(max(gene_kurt))),color='blue') 
plt.title('Highest Kurtosis Gene Expression Feature')

train_weird=train.copy()
train_weird['sum_cell']=sum([abs(train_weird['c-'+str(i)]) for i in range(100)])
train_weird['sum_gene']=sum([abs(train_weird['g-'+str(i)]) for i in range(772)])
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
cell_max=train.loc[train_weird['sum_cell']==train_weird['sum_cell'].max()]
title=cell_max.values[0][0]
cell_max=cell_max[[col for col in train_weird if 'c-' in col]]
plt.plot(cell_max.values.reshape(-1,1))
plt.title('sig-id: '+ title + ' - large cell values')
plt.subplot(2,2,2)
cell_min=train.loc[train_weird['sum_cell']==train_weird['sum_cell'].min()] #-this is still not capturing small abs values individually - will try again after the code
title=cell_min.values[0][0]
cell_min=cell_min[[col for col in train_weird if 'c-' in col]]
plt.plot(cell_min.values.reshape(-1,1))
plt.title('sig-id: '+ title + ' - small cell values')
plt.subplot(2,2,3)
gene_max=train.loc[train_weird['sum_gene']==train_weird['sum_gene'].max()]
title=gene_max.values[0][0]
gene_max=gene_max[[col for col in train_weird if 'g-' in col]]
plt.plot(gene_max.values.reshape(-1,1))
plt.title('sig-id: '+ title + ' - large gene values')
plt.subplot(2,2,4)
gene_min=train.loc[train_weird['sum_gene']==train_weird['sum_gene'].min()]
title=gene_min.values[0][0]
gene_min=gene_min[[col for col in train_weird if 'g-' in col]]
plt.plot(gene_min.values.reshape(-1,1))
plt.title('sig-id: '+ title + ' - small gene values')
train_corr=train[[col for col in train if ('c-' in col or 'g-' in col)]]
corr_mat=train_corr.corr()
fig=plt.figure(figsize=(50,50))
sns.heatmap(corr_mat, vmax=1, center=0, cmap='bwr', annot=False)
for var in ['cp_type', 'cp_time','cp_dose']:
    cell_choice=np.random.choice(100,3)
    gene_choice=np.random.choice(772,3)
    plt.figure(figsize=(30,10))
    for i in range(3): 
        plt.subplot(2,3,i+1)
        sns.kdeplot(data=train,x='c-'+str(cell_choice[i]), hue=var, fill=True, palette='Set1')
    for i in range(3):
        plt.subplot(2,3,i+4)
        sns.kdeplot(data=train, x='g-'+str(gene_choice[i]), hue=var, fill=True, palette='Set2')
train_targets.head()
#number of activations:
sns.countplot(x=train_targets.sum(axis=1,numeric_only=True))
plt.title('Number of Activations')
#add the number and percentage of total
#number of activations by treatment type
plt.figure(figsize=(10,5))
sns.countplot(x=train_targets.sum(axis=1, numeric_only=True), hue=train['cp_type'])
plt.title('Number of Activations by Treatment Type')
plt.figure(figsize=(15,5))
sns.countplot(x=train_targets.sum(axis=1, numeric_only=True), hue=train['cp_time'])
plt.title('Number of Activations by Treatment Duration')
plt.figure(figsize=(10,5))
sns.countplot(x=train_targets.sum(axis=1, numeric_only=True), hue=train['cp_dose'])
plt.title('Number of Activations by Dose Level')
counts={}
for col in train_targets.columns[1:]:
    counts[col]=train_targets[col].sum()
counts=pd.DataFrame.from_dict(counts, orient='index', columns=['Activation Count'])
plt.figure(figsize=(15,5))
sns.barplot(x=counts.index, y=counts['Activation Count'])
plt.xticks([])
plt.title('Activation Frequencies')

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
counts.sort_values(by='Activation Count',ascending=False, inplace=True)
sns.barplot(x=counts['Activation Count'][0:5], y=counts.index[0:5], color='orange')
plt.yticks(rotation=45)
plt.title('Most Frequent Activations')
plt.subplot(1,2,2)
sns.barplot(x=counts['Activation Count'][-5:], y=counts.index[-5:], color='blue')
plt.yticks(rotation=45)
plt.title('Least Frequent Activations')
plt.subplots_adjust(wspace=0.4)
#antagonist, inhibitor, activator, other
inhibitors=[col for col in train_targets.columns if 'inhibitor' in col]
antagonists=[col for col in train_targets.columns if 'antagonist' in col]
activators=[col for col in train_targets.columns if 'activator' in col]
others=[col for col in train_targets.columns if ((col not in inhibitors+antagonists+activators) and col!='sig_id')]
print('Number of inhibitors: ', len(inhibitors))
print('Number of antagonists: ',len(antagonists))
print('Number of activators: ', len(activators))
print('Number of others: ', len(others))
#corr matrix of activators. 
targ_corr=train_targets.drop('sig_id', axis=1)
corr=targ_corr.corr()
fig=plt.figure(figsize=(15,15))
sns.heatmap(corr, vmax=1, center=0, cmap='bwr', annot=False)
s = corr.unstack()
so = s.sort_values(kind="quicksort",ascending=False)
print(so[(so>0.5) & (so<1)])
print('High cell: ', int(train_targets.loc[train['sig_id']=='id_c75812bed'].sum(axis=1,numeric_only=True)))

print('Low cell: ', int(train_targets.loc[train['sig_id']=='id_3c78210f8'].sum(axis=1,numeric_only=True)))

print('High gene: ',int(train_targets.loc[train['sig_id']=='id_b31edc707'].sum(axis=1,numeric_only=True)))

print('Low gene: ',int(train_targets.loc[train['sig_id']=='id_3ee669c95'].sum(axis=1,numeric_only=True)))


from sklearn.decomposition import PCA
pca=PCA()
pca.fit(train.iloc[:,776:])
pca_cell=pca.transform(train.iloc[:,776:])

sns.lineplot(x=np.arange(1,train.iloc[:,776:].shape[1]+1), y=pca.explained_variance_ratio_*100, color='blue', marker="o")
plt.axis([0,9,0,100])
plt.ylabel('Explained Variance (%)')
plt.xlabel('Components')
plt.title('Scree Plot (Explained Variance) - Cell Viability')
cell_pca_df=pd.DataFrame(data=pca_cell, columns=['pc'+str(i) for i in range(100)])
plt.figure(figsize=(21,5))
plt.subplot(1,3,1)
sns.scatterplot(data=cell_pca_df, x='pc0', y='pc1', hue=train['cp_type'],style=train['cp_type'], palette='Set1', markers=["o","^"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cell Viability PCA by Treatment Type')
plt.subplot(1,3,2)
sns.scatterplot(data=cell_pca_df, x='pc0', y='pc1', hue=train['cp_time'], style=train['cp_time'], palette='Set1', markers=["o","^","D"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cell Viability PCA by Treatment Duration')
plt.subplot(1,3,3)
sns.scatterplot(data=cell_pca_df, x='pc0', y='pc1', hue=train['cp_dose'], style=train['cp_dose'], palette='Set1', markers=["o","^"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cell Viability PCA by Dose Level')


act_counts=train_targets.sum(axis=1,numeric_only=True)
def group_counts(count): #function to group activation counts as frequencies are low for higher counts
    if count>=3:
        return '3+'
    elif count==2:
        return '2'
    elif count==1:
        return '1'
    else: 
        return '0'
act_counts_grouped=act_counts.apply(group_counts)
plt.figure(figsize=(10,5))
sns.scatterplot(data=cell_pca_df, x='pc0', y='pc1', hue=act_counts_grouped, style=act_counts_grouped,palette='Set1', markers=["o","^","D","X"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cell Viability PCA by Activation Count')
pca=PCA()
pca.fit(train.iloc[:,4:776])
pca_gene=pca.transform(train.iloc[:,4:776])

sns.lineplot(x=np.arange(1,train.iloc[:,4:776].shape[1]+1), y=pca.explained_variance_ratio_*100, color='blue', marker="o")
plt.axis([0,9,0,100])
plt.ylabel('Explained Variance (%)')
plt.xlabel('Components')
plt.title('Scree Plot (Explained Variance) - Gene Expression')
gene_pca_df=pd.DataFrame(data=pca_gene, columns=['pc'+str(i) for i in range(772)])
plt.figure(figsize=(21,5))
plt.subplot(1,3,1)
sns.scatterplot(data=gene_pca_df, x='pc0', y='pc1', hue=train['cp_type'],style=train['cp_type'], palette='Set2', markers=["o","^"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gene Expression PCA by Treatment Type')
plt.subplot(1,3,2)
sns.scatterplot(data=gene_pca_df, x='pc0', y='pc1', hue=train['cp_time'], style=train['cp_time'], palette='Set2', markers=["o","^","D"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gene Expression PCA by Treatment Duration')
plt.subplot(1,3,3)
sns.scatterplot(data=gene_pca_df, x='pc0', y='pc1', hue=train['cp_dose'], style=train['cp_dose'], palette='Set2', markers=["o","^"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gene Expression PCA by Dose Level')

plt.figure(figsize=(10,5))
sns.scatterplot(data=gene_pca_df, x='pc0', y='pc1', hue=act_counts_grouped, style=act_counts_grouped,palette='Set2', markers=["o","^","D","X"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gene Expression PCA by Activation Count')
#number of activations:
sns.countplot(x=train_nonscored.sum(axis=1,numeric_only=True))
plt.title('Number of Activations')
#add the number and percentage of total
counts={}
for col in train_nonscored.columns[1:]:
    counts[col]=train_nonscored[col].sum()
counts=pd.DataFrame.from_dict(counts, orient='index', columns=['Activation Count'])
plt.figure(figsize=(15,5))
sns.barplot(x=counts.index, y=counts['Activation Count'])
plt.xticks([])
plt.title('Activation Frequencies')

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
counts.sort_values(by='Activation Count',ascending=False, inplace=True)
sns.barplot(x=counts['Activation Count'][0:5], y=counts.index[0:5], color='orange')
plt.yticks(rotation=45)
plt.title('Most Frequent Activations')
plt.subplot(1,2,2)
sns.barplot(x=counts['Activation Count'][-5:], y=counts.index[-5:], color='blue')
plt.yticks(rotation=45)
plt.title('Least Frequent Activations')
plt.subplots_adjust(wspace=0.4)
print('Number of non-activators: ', counts[counts['Activation Count']==0].shape[0])
#correlations scored vs. non-scored
all_act=train_targets.join(train_nonscored.drop('sig_id',axis=1)).drop('sig_id',axis=1)
all_corr=all_act.corr()
s = all_corr.unstack()
so = s.sort_values(kind="quicksort",ascending=False)
targ_vs_nonscored= so[[(so.index[i][0] in train_targets.columns) & (so.index[i][1] in train_nonscored.columns) for i in range(so.shape[0])]]
print(targ_vs_nonscored[(targ_vs_nonscored>0.5) & (targ_vs_nonscored<=1)])
print(targ_vs_nonscored[(targ_vs_nonscored<-0.5) & (targ_vs_nonscored>=-1)])
#pca on nonscored
nonscored_counts=train_nonscored.sum(axis=1,numeric_only=True)

nonscored_counts_grouped=nonscored_counts.apply(group_counts)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.scatterplot(data=cell_pca_df, x='pc0', y='pc1', hue=nonscored_counts_grouped, style=nonscored_counts_grouped,palette='Set1', markers=["o","^","D","X"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cell Viability PCA by Activation Count (Non-Scored)')
plt.subplot(2,2,2)
sns.scatterplot(data=cell_pca_df, x='pc0', y='pc1', hue=nonscored_counts_grouped[nonscored_counts_grouped!='0'], style=nonscored_counts_grouped[nonscored_counts_grouped!='0'],palette='Set1', markers=["^","D","X"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cell Viability PCA by Activation Count>0 (Non-Scored)')

plt.subplot(2,2,3)
sns.scatterplot(data=gene_pca_df, x='pc0', y='pc1', hue=nonscored_counts_grouped, style=nonscored_counts_grouped,palette='Set2', markers=["o","^","D","X"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gene Expression PCA by Activation Count (Non-Scored)')

plt.subplot(2,2,4)
sns.scatterplot(data=gene_pca_df, x='pc0', y='pc1', hue=nonscored_counts_grouped[nonscored_counts_grouped!='0'], style=nonscored_counts_grouped[nonscored_counts_grouped!='0'],palette='Set2', markers=["^","D","X"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gene Expression PCA by Activation Count>0 (Non-Scored)')