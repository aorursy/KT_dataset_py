import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import numpy as np

import pandas_profiling
train_data= pd.read_csv('../input/lish-moa/train_features.csv')

test_data = pd.read_csv('../input/lish-moa/test_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_non_scored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
print("Train Data: ")

print("Shape:"+str(train_data.shape))

train_data.head(3)
print("Test Data\nShape:"+str(test_data.shape))

test_data.head(3)
print("Targets Scored:\nShape:"+str(train_targets_scored.shape))

train_targets_scored.head(3)
print("Targets Non scored:\nShape:"+str(train_targets_non_scored.shape))

train_targets_non_scored.head(3)
train_gs = train_data.iloc[:,train_data.columns.map(lambda x: x[0:2])=='g-']

train_cs = train_data.iloc[:,train_data.columns.map(lambda x: x[0:2])=='c-']

print("Gene expression data Number of columns: "+str(train_gs.shape[1]))

print("cell viability data Number of columns: "+ str(train_cs.shape[1]))
print("Mean:"+str(pd.concat([train_gs,train_cs],axis=1).values.mean()))

print("Std:"+str(pd.concat([train_gs,train_cs],axis=1).values.std()))

plt.figure(figsize=(5,5))

sns.distplot(pd.concat([train_gs,train_cs],axis=1).values)

plt.title('combined gene expression and cell viability')

plt.figure(figsize=(12,12))

plt.subplot(2,2,1)

sns.distplot(train_gs['g-0'],color='pink')

plt.title('g-0')

plt.subplot(2,2,2)

sns.distplot(train_gs['g-100'],color='pink')

plt.title('g-100')

plt.subplot(2,2,3)

sns.distplot(train_cs['c-1'],color='pink')

plt.title('c-0')

plt.subplot(2,2,4)

sns.distplot(train_cs['c-80'],color='pink')

plt.title('c-80')
print("Gene expression data statistics: ")

print("  Mean: "+str(train_gs.values.mean()))

print("  Std: "+str(train_gs.values.std()))

print("  Max: "+str(train_gs.values.max()))

print("  Min: "+str(train_gs.values.min()))

print('\nCell viability data statistics: ')

print("  Mean: "+str(train_cs.values.mean()))

print("  Std: "+str(train_cs.values.std()))

print("  Max: "+str(train_cs.values.max()))

print("  Min: "+str(train_cs.values.min()))
plt.figure(figsize=(5,12))

plt.subplot(3,1,1)

splot = sns.countplot(train_data["cp_type"])

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.1f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

plt.title('cp_type')

plt.subplot(3,1,2)

sns.countplot(train_data['cp_time'],hue=train_data['cp_type'])

plt.title('cp_time vs cp_type')

plt.subplot(3,1,3)

sns.countplot(train_data['cp_dose'],hue=train_data['cp_type'])

plt.title('cp_dose vs cp_type')

plt.tight_layout()
print("Number of scored targets: "+str(train_targets_scored.shape[1]))
out = dict()

arr=train_targets_scored.drop('sig_id',axis=1).values==1

for a in range(len(arr)):

    o=np.sum(arr[a])

    if o not in out.keys():

        out[o]=1

    else:

        out[o]+=1

length = 23814

plt.figure(figsize=(7,7))

splot = sns.barplot(x=list(out.keys()),y=list(out.values()))

for p in splot.patches:

    splot.annotate(format(p.get_height()*100/length, '.1f')+'%', 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

plt.xlabel('Number of MoAs in sample')

plt.ylabel('Count')

plt.title('Percentage of samples with MoA counts')
cor = train_targets_scored.drop('sig_id',axis=1).corr()
cor = train_targets_scored.drop('sig_id',axis=1).corr()

plt.figure(figsize=(10,10))

sns.heatmap(cor)
df = pd.DataFrame(columns=['drug_a','drug_b','corr'])

for j in range(len(cor)):

    for i in range(len(cor)):

        if cor.iloc[i,j]>=0.7 and cor.iloc[i,j]!=1.0:

            df = pd.concat([df,pd.DataFrame({'drug_a':[cor.columns[j]],'drug_b':[cor.columns[i]],'corr':[cor.iloc[i,j]]})],axis=0)

df
print("Number of non scored targets: "+str(train_targets_non_scored.shape[1]))


out = dict()

arr=pd.concat([train_targets_scored.drop('sig_id',axis=1),train_targets_non_scored.drop('sig_id',axis=1)],axis=1).values==1

for a in range(len(arr)):

    o=np.sum(arr[a])

    if o not in out.keys():

        out[o]=1

    else:

        out[o]+=1

length = 23814

plt.figure(figsize=(7,7))

splot = sns.barplot(x=list(out.keys()),y=list(out.values()))

for p in splot.patches:

    splot.annotate(format(p.get_height()*100/length, '.1f')+'%', 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

plt.xlabel('Number of MoAs in sample')

plt.ylabel('Count')

plt.title('Percentage of samples with MoA counts')
genes = [col for col in train_data if col.startswith('g-')]

cells = [col for col in train_data if col.startswith('c-')]
plt.figure(figsize=(8,8))

sns.heatmap(train_data.loc[:,genes].corr(),cmap='viridis')
plt.figure(figsize=(8,8))

sns.heatmap(train_data.loc[:,cells].corr(),cmap='viridis')
plt.plot(train_cs.iloc[1,:])

plt.title('cell viability data for second sample')
cor = train_gs.corr()
df = pd.DataFrame(columns=['gene_a','gene_b','corr'])

for j in range(len(cor)):

    for i in range(len(cor)):

        if cor.iloc[i,j]<=-0.8 and cor.iloc[i,j]!=1.0:

            df = pd.concat([df,pd.DataFrame({'gene_a':[cor.columns[j]],'gene_b':[cor.columns[i]],'corr':[cor.iloc[i,j]]})],axis=0)

df
cp_1 = train_data[train_data['cp_type']=='trt_cp']

cp_2 = train_data[train_data['cp_type']!='trt_cp']

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(cp_1['g-0'],color='orange',hist=False)

sns.distplot(cp_2['g-0'],color='cyan',hist=False)

plt.title('g-0')

plt.subplot(2,2,2)

sns.distplot(cp_1['g-100'],color='orange',hist=False)

sns.distplot(cp_2['g-100'],color='cyan',hist=False)

plt.title('g-100')

plt.subplot(2,2,3)

sns.distplot(cp_1['g-500'],color='orange',hist=False)

sns.distplot(cp_2['g-500'],color='cyan',hist=False)

plt.title('g-500')

plt.subplot(2,2,4)

sns.distplot(cp_1['g-600'],color='orange',hist=False)

sns.distplot(cp_2['g-600'],color='cyan',hist=False)

plt.title('g-600')
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(cp_1['c-1'],color='orange',hist=False)

sns.distplot(cp_2['c-1'],color='cyan',hist=False)

plt.title('c-1')

plt.subplot(2,2,2)

sns.distplot(cp_1['c-20'],color='orange',hist=False)

sns.distplot(cp_2['c-20'],color='cyan',hist=False)

plt.title('c-20')

plt.subplot(2,2,3)

sns.distplot(cp_1['c-40'],color='orange',hist=False)

sns.distplot(cp_2['c-40'],color='cyan',hist=False)

plt.title('c-40')

plt.subplot(2,2,4)

sns.distplot(cp_1['c-50'],color='orange',hist=False)

sns.distplot(cp_2['c-50'],color='cyan',hist=False)

plt.title('c-50')
cp_1 = train_data[train_data['cp_time']==24]

cp_2 = train_data[train_data['cp_type']!=48]

cp_3 = train_data[train_data['cp_type']!=72]

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(cp_1['g-0'],color='orange',hist=False)

sns.distplot(cp_2['g-0'],color='cyan',hist=False)

sns.distplot(cp_3['g-0'],color='blue',hist=False)

plt.title('g-0')

plt.subplot(2,2,2)

sns.distplot(cp_1['g-100'],color='orange',hist=False)

sns.distplot(cp_2['g-100'],color='cyan',hist=False)

sns.distplot(cp_3['g-100'],color='blue',hist=False)

plt.title('g-100')

plt.subplot(2,2,3)

sns.distplot(cp_1['g-500'],color='orange',hist=False)

sns.distplot(cp_2['g-500'],color='cyan',hist=False)

sns.distplot(cp_3['g-500'],color='blue',hist=False)

plt.title('g-500')

plt.subplot(2,2,4)

sns.distplot(cp_1['g-600'],color='orange',hist=False)

sns.distplot(cp_2['g-600'],color='cyan',hist=False)

sns.distplot(cp_3['g-600'],color='blue',hist=False)

plt.title('g-600')


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(cp_1['c-1'],color='orange',hist=False)

sns.distplot(cp_2['c-1'],color='cyan',hist=False)

sns.distplot(cp_3['c-1'],color='blue',hist=False)

plt.title('c-1')

plt.subplot(2,2,2)

sns.distplot(cp_1['c-20'],color='orange',hist=False)

sns.distplot(cp_2['c-20'],color='cyan',hist=False)

sns.distplot(cp_3['c-20'],color='blue',hist=False)

plt.title('c-20')

plt.subplot(2,2,3)

sns.distplot(cp_1['c-40'],color='orange',hist=False)

sns.distplot(cp_2['c-40'],color='cyan',hist=False)

sns.distplot(cp_3['c-40'],color='blue',hist=False)

plt.title('c-40')

plt.subplot(2,2,4)

sns.distplot(cp_1['c-50'],color='orange',hist=False)

sns.distplot(cp_2['c-50'],color='cyan',hist=False)

sns.distplot(cp_3['c-50'],color='blue',hist=False)

plt.title('c-50')
cp_1 = train_data[train_data['cp_dose']=='D0']

cp_2 = train_data[train_data['cp_dose']!='D1']

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(cp_1['g-0'],color='orange',hist=False)

sns.distplot(cp_2['g-0'],color='cyan',hist=False)

plt.title('g-0')

plt.subplot(2,2,2)

sns.distplot(cp_1['g-100'],color='orange',hist=False)

sns.distplot(cp_2['g-100'],color='cyan',hist=False)

plt.title('g-100')

plt.subplot(2,2,3)

sns.distplot(cp_1['g-500'],color='orange',hist=False)

sns.distplot(cp_2['g-500'],color='cyan',hist=False)

plt.title('g-500')

plt.subplot(2,2,4)

sns.distplot(cp_1['g-600'],color='orange',hist=False)

sns.distplot(cp_2['g-600'],color='cyan',hist=False)

plt.title('g-600')
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(cp_1['c-1'],color='orange',hist=False)

sns.distplot(cp_2['c-1'],color='cyan',hist=False)

plt.title('c-1')

plt.subplot(2,2,2)

sns.distplot(cp_1['c-20'],color='orange',hist=False)

sns.distplot(cp_2['c-20'],color='cyan',hist=False)

plt.title('c-20')

plt.subplot(2,2,3)

sns.distplot(cp_1['c-40'],color='orange',hist=False)

sns.distplot(cp_2['c-40'],color='cyan',hist=False)

plt.title('c-40')

plt.subplot(2,2,4)

sns.distplot(cp_1['c-50'],color='orange',hist=False)

sns.distplot(cp_2['c-50'],color='cyan',hist=False)

plt.title('c-50')