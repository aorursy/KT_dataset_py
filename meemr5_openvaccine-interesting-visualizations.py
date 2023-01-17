import pandas as pd

import os

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')



SEED = 2020



def seed_everything(SEED):

    np.random.seed(SEED)

#     tf.random.set_seed(SEED)

    os.environ['PYTHONHASHSEED'] = str(SEED)



seed_everything(SEED)
PATH = '/kaggle/input/stanford-covid-vaccine'

os.listdir(PATH)
train = pd.read_json(os.path.join(PATH,'train.json'),lines=True).drop('index',axis=1)

test = pd.read_json(os.path.join(PATH,'test.json'),lines=True).drop('index',axis=1)

sub = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))
print("Train data shape: ",train.shape)

print("Test data shape: ",test.shape)

print("Sample submission shape: ",sub.shape)
train.head()
test.head()
train.columns
fig, ax = plt.subplots(1,2,figsize=(20,5))



train['signal_to_noise'].plot.kde(ax=ax[0])

ax[0].set_title('Signal/Noise')



sns.countplot(data=train,y='SN_filter',ax=ax[1])

ax[1].set_title('SN_filter')



plt.show()
plt.figure(figsize=(20,2))

sns.boxplot(data=train,x='signal_to_noise')



plt.show()
print("Number of samples with -ev signal/noise values: ",train[train['signal_to_noise']<0].shape[0])



Q1 = np.percentile(train['signal_to_noise'],q=25)

Q3 = np.percentile(train['signal_to_noise'],q=75)

IQR = Q3 - Q1

print("Number of samples with too high signal/noise values", train[train['signal_to_noise']>Q3+1.5*IQR].shape[0])
train.seq_length.value_counts()
test.seq_length.value_counts()
avg_reactivity = np.array(list(map(np.array,train.reactivity))).mean(axis=0)

avg_deg_50C = np.array(list(map(np.array,train.deg_50C))).mean(axis=0)

avg_deg_pH10 = np.array(list(map(np.array,train.deg_pH10))).mean(axis=0)



avg_deg_Mg_50C = np.array(list(map(np.array,train.deg_Mg_50C))).mean(axis=0)

avg_deg_Mg_pH10 = np.array(list(map(np.array,train.deg_Mg_pH10))).mean(axis=0)
fig, ax = plt.subplots(1,3,figsize=(20,4))



# Distribution of Reactivity Averaged over position

sns.kdeplot(avg_reactivity,ax=ax[0])

ax[0].set_title('Distribution of Reactivity Averaged over position',size=15)



# Distribution of deg_50C Averaged over position

sns.kdeplot(avg_deg_50C,ax=ax[1])

ax[1].set_title('Distribution of deg_50C Averaged over position',size=15)



# Distribution of deg_pH10 Averaged over position

sns.kdeplot(avg_deg_pH10,ax=ax[2])

ax[2].set_title('Distribution of deg_pH10 Averaged over position',size=15)





plt.show()
plt.figure(figsize=(20,8))



sns.lineplot(x=range(68),y=avg_reactivity,label='reactivity')

sns.lineplot(x=range(68),y=avg_deg_50C,label='deg_50C')

sns.lineplot(x=range(68),y=avg_deg_pH10,label='deg_ph10')



plt.xlabel('Positions')

plt.xticks(range(0,68,2))

plt.ylabel('Values')

plt.title('Average Target Values (w/o Mg) V/S Positions')



plt.show()
plt.figure(figsize=(20,8))

sns.regplot(x=avg_deg_50C,y=avg_deg_pH10)



plt.title('Average deg_50C V/S deg_pH10')

plt.ylabel('deg_50C')

plt.xlabel('deg_pH10')



plt.show()
print("Correlation Coeff between avg_deg_50C & avg_deg_pH10: ",np.corrcoef(avg_deg_50C,avg_deg_pH10)[0][1])
fig, ax = plt.subplots(1,2,figsize=(20,4))



# Distribution of deg_50C Averaged over position

sns.kdeplot(avg_deg_Mg_50C,ax=ax[0])

ax[0].set_title('Distribution of deg_Mg_50C Averaged over position',size=15)





# Distribution of deg_pH10 Averaged over position

sns.kdeplot(avg_deg_Mg_pH10,ax=ax[1])

ax[1].set_title('Distribution of deg_Mg_pH10 Averaged over position',size=15)



plt.show()
plt.figure(figsize=(20,8))



sns.lineplot(x=range(68),y=avg_deg_Mg_50C,label='deg_Mg_50C')

sns.lineplot(x=range(68),y=avg_deg_Mg_pH10,label='deg_Mg_pH10')



plt.xlabel('Positions')

plt.xticks(range(0,68,2))

plt.ylabel('Values')

plt.title('Average Target Values (w Mg) V/S Positions')



plt.show()
print("Correlation Coeff between avg_deg_Mg_50C & avg_deg_Mg_pH10: ",np.corrcoef(avg_deg_Mg_50C,avg_deg_Mg_pH10)[0][1])
pos = np.random.choice(68)



fig, ax = plt.subplots(1,3,figsize=(20,4))



# Distribution of Reactivity at Random position

sns.kdeplot(np.array(list(map(np.array,train.reactivity)))[:,pos],ax=ax[0])

ax[0].set_title(f'Distribution of Reactivity at position-{pos}',size=15)



# Distribution of deg_50C at Random position

sns.kdeplot(np.array(list(map(np.array,train.deg_50C)))[:,pos],ax=ax[1])

ax[1].set_title(f'Distribution of deg_50C at position-{pos}',size=15)



# Distribution of deg_pH10 at Random position

sns.kdeplot(np.array(list(map(np.array,train.deg_pH10)))[:,pos],ax=ax[2])

ax[2].set_title(f'Distribution of deg_pH10 at position-{pos}',size=15)



plt.show()
fig, ax = plt.subplots(1,2,figsize=(20,4))



# Distribution of deg_50C at Random position

sns.kdeplot(np.array(list(map(np.array,train.deg_Mg_50C)))[:,pos],ax=ax[0])

ax[0].set_title(f'Distribution of deg_Mg_50C at position-{pos}',size=15)





# Distribution of deg_pH10 at Random position

sns.kdeplot(np.array(list(map(np.array,train.deg_Mg_pH10)))[:,pos],ax=ax[1])

ax[1].set_title(f'Distribution of deg_Mg_pH10 at position-{pos}',size=15)



plt.show()
y = ['reactivity_error','deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C','deg_error_50C']

x = [np.array(list(map(np.array,train[col]))).mean(axis=0) for col in y]



plt.figure(figsize=(20,5))



sns.boxplot(y=y,x=x)



plt.xlabel('Error values')

plt.title('Average Errors in Calculation of Targets')



plt.show()
plt.figure(figsize=(20,8))



for i in range(len(y)):

    sns.lineplot(x=range(68),y=x[i],label=y[i])

    

plt.xlabel('Positions')

plt.xticks(range(0,68,2))

plt.ylabel('Error')

plt.title('Error V/S Position')



plt.show()
pos = np.random.choice(68)



y = ['reactivity_error','deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C','deg_error_50C']

x = [np.array(list(map(np.array,train[col])))[:,pos] for col in y]



plt.figure(figsize=(20,5))

plt.title(f'Error Distribution at position - {pos}')

plt.xlabel('Error')



sns.boxplot(y=y,x=x)



plt.show()
y = ['reactivity_error','deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C','deg_error_50C']

x = [np.array(list(map(np.array,train[train.SN_filter==1][col])))[:,pos] for col in y]



plt.figure(figsize=(20,5))

plt.title(f'(Filtered) Error Distribution at position - {pos}')

plt.xlabel('Error')



sns.boxplot(y=y,x=x)



plt.show()
!pip3 install -q forgi[all]

!conda install -y -c bioconda viennarna
import forgi.graph.bulge_graph as fgb

import forgi.visual.mplotlib as fvm
def plot_sample(sample):

    

    """Source: https://www.kaggle.com/erelin6613/openvaccine-rna-visualization"""

    

    struct = sample['structure']

    seq = sample['sequence']

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{struct}\n{seq}')[0]

    

    plt.figure(figsize=(20,8))

    fvm.plot_rna(bg)

    plt.title(f"RNA Structure (id: {sample.id})")

    plt.show()
sample = train.iloc[np.random.choice(train.shape[0])]

plot_sample(sample)
print("Predicted Loop type: ",sample['predicted_loop_type'])
from collections import defaultdict



reactivity = defaultdict(lambda: [])

deg_Mg_50C = defaultdict(lambda: [])

deg_Mg_pH10 = defaultdict(lambda: [])



for i in range(len(sample['reactivity'])):

    reactivity[sample['structure'][i]].append(float(sample['reactivity'][i]))

    deg_Mg_50C[sample['structure'][i]].append(float(sample['deg_Mg_50C'][i]))

    deg_Mg_pH10[sample['structure'][i]].append(float(sample['deg_Mg_pH10'][i]))



plt.figure(figsize=(20,5))

for key in reactivity.keys():

    sns.kdeplot(data=reactivity[key],label=key)



plt.title('Structure wise Distribution of Reactivity of a Random Sample')

plt.show()
plt.figure(figsize=(20,5))

for key in reactivity.keys():

    sns.kdeplot(data=deg_Mg_50C[key],label=key)



plt.title('Structure wise Distribution of deg_Mg_50C of a Random Sample')

plt.show()
plt.figure(figsize=(20,5))

for key in reactivity.keys():

    sns.kdeplot(data=deg_Mg_pH10[key],label=key)



plt.title('Structure wise Distribution of deg_Mg_pH10 of a Random Sample')

plt.show()
reactivityDict = defaultdict(lambda: [])



for index in range(train.shape[0]):

    

    sample = train.iloc[index]



    structure = sample['structure']

    sequence = sample['sequence']

    reactivity = sample['reactivity']



    q = []



    for i,s in enumerate(structure[:len(reactivity)]):

        if s=='.':

            reactivityDict[sequence[i]].append(reactivity[i])

        elif s=='(':

            q.append(i)

        elif s==')':

            j = q.pop(0)

            key = "-".join(sorted([sequence[i],sequence[j]]))

            reactivityDict[key].append(reactivity[i])

            reactivityDict[key].append(reactivity[j])
fig, ax = plt.subplots(len(reactivityDict.keys()),1,figsize=(20,2*len(reactivityDict.keys())),sharex=True)



for i, key in enumerate(reactivityDict.keys()):

    sns.boxplot(x=reactivityDict[key],ax=ax[i])

    ax[i].set_ylabel(key)



plt.xlabel('Reactivity')

plt.show()