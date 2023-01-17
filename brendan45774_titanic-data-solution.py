%%capture 

!pip install pandas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path

import matplotlib.pyplot as plt

from fastai import *

from fastai.tabular import *

import torch

import missingno as msno

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = Path('/kaggle/input/titanic')

trpath = path/'train.csv'

cvpath = path/'test.csv'



df_train_raw = pd.read_csv(trpath)

df_test_raw = pd.read_csv(cvpath)



df_train = df_train_raw.copy(deep = True)

df_test  = df_test_raw.copy(deep = True)



data_cleaner = [df_train_raw, df_test_raw] #to clean both simultaneously
df_train.head(n=10)
varnames = list(df_train.columns)

for name in varnames:

    print(name+": ",type(df_train.loc[1,name]))
df_train.isnull().sum(axis=0)
msno.matrix(df_train)
msno.bar(df_test)
plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14)



plt.figure()

fig = df_train.groupby('Survived')['Age'].plot.hist(histtype= 'bar', alpha = 0.8)

plt.legend(('Died','Survived'), fontsize = 12)

plt.xlabel('Age', fontsize = 18)

plt.show()
df_train.corr(method='pearson')['Age'].abs()
plt.figure()

fig = df_train.groupby('Survived')['Parch'].plot.hist(histtype= 'bar',alpha = 0.8)

plt.legend(('Died','Survived'),)

plt.xlabel('Parch')

plt.show()
df_train['Family onboard'] = df_train['Parch'] + df_train['SibSp']

plt.rcParams['figure.figsize'] = [20, 8]

plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14)



fig, axes = plt.subplots(nrows=1, ncols=3)

df_train.groupby(['Parch'])['Survived'].value_counts(normalize=True).unstack().plot.bar(ax=axes[1],width = 0.85)

df_train.groupby(['SibSp'])['Survived'].value_counts(normalize=True).unstack().plot.bar(ax=axes[2],width = 0.85)

df_train.groupby(['Family onboard'])['Survived'].value_counts(normalize=True).unstack().plot.bar(ax=axes[0],width = 0.85)



axes[0].set_xlabel('Family onboard',fontsize = 18)

axes[1].set_xlabel('parents / children aboard',fontsize = 18)

axes[2].set_xlabel(' siblings / spouses aboard',fontsize = 18)



for i in range(3):

    axes[i].legend(('Died','Survived'),fontsize = 12, loc = 'upper left')



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=0)



plt.suptitle('Survival rates over Number of relatives onboard',fontsize =22)

plt.show()
plt.rcParams['figure.figsize'] = [6, 5]

plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14) 



plt.figure()

fig = df_train.groupby(['Sex'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.9)

plt.legend(('Died','Survived'),fontsize = 12, loc = 'upper left')

plt.xlabel('Gender',fontsize =18)

plt.xticks(rotation=0)



plt.suptitle('Survival rates over Gender',fontsize =22)

plt.show()
plt.figure()

fig = df_train.groupby('Survived')['Fare'].plot.hist(histtype= 'bar', alpha = 0.8)

plt.legend(('Died','Survived'))

plt.xlabel('Fare')

plt.show()



plt.rcParams['figure.figsize'] = [10, 5]

plt.rc('xtick', labelsize=12) 

plt.rc('ytick', labelsize=12) 
df_train['Title'] = df_train['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

varnames = list(df_train.columns)

for name in varnames:

    print(name+": ",type(df_train.loc[1,name]))

    

print(list(df_train['Title'].unique()))    

df_test['Title'] = df_test['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

df_test['Title'].unique()
def new_titles(df):

    new_titles = dict()

    assert 'Title' in df.columns

    for key in df['Title'].unique():

        females = ['Mrs','Miss','Ms','Mlle','Mme','Dona']

        males = ['Mr','Don']

        notable = ['Jonkheer','the Countess','Lady','Sir','Major','Col','Capt','Dr','Rev','Notable']

        titles = [females,males,notable,'Master']

        newtitles = ['Mrs','Mr','Notable','Master']

        idx = [key in sublist for sublist in titles]

        idx = np.where(idx)[0] 

        new_titles[key] = newtitles[idx[0]]

    return new_titles





new_titles_dict = new_titles(df_train)

df_train['Title'] = df_train['Title'].replace(new_titles_dict)
plt.rcParams['figure.figsize'] = [12, 5]

plt.rc('xtick', labelsize=12) 

plt.rc('ytick', labelsize=12) 



plt.figure()

fig = df_train.groupby(['Title'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.9)

plt.legend(('Died','Survived'),fontsize = 12, loc = 'upper left')

plt.xlabel('Title',fontsize =16)

plt.xticks(rotation=0)





plt.suptitle('Survival rates over Title',fontsize =20)

plt.show()
df_train['Cabin'][df_train['Cabin'].isnull()]='Missing'

df_train['Cabin'] = df_train['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]
plt.rcParams['figure.figsize'] = [12, 5]

plt.figure()

fig = df_train.groupby(['Cabin'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.9)

plt.legend(('Died','Survived'),fontsize = 12, loc = 'upper left')

plt.xlabel('Cabin Deck',fontsize =12)

plt.suptitle('Survival rates over Cabin Deck',fontsize =18)

plt.xticks(rotation=0)

plt.show()
plt.rcParams['figure.figsize'] = [10, 5]

plt.figure()

fig = df_train.groupby(['Embarked'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.7)

plt.legend(('Died','Survived'),fontsize = 12, loc = 'upper left')

plt.xlabel('Embarking Port',fontsize =18)

plt.suptitle('Survival rates over embarking port',fontsize =22)

plt.xticks(rotation=0)

plt.show()
def df_fill(datasets, mode):

    assert mode =='median' or mode =='sampling'

    datasets_cp =[]

    np.random.seed(2)

    varnames = ['Age','Fare']

    for d in datasets:

        df = d.copy(deep = True)

        for var in varnames:

            idx = df[var].isnull()

            if idx.sum()>0:

                if mode =='median':

                    medians = df.groupby('Pclass')[var].median()

                    for i,v in enumerate(idx):

                        if v:

                            df[var][i] = medians[df['Pclass'][i]]

                else:

                    g = df[idx==False].groupby('Pclass')[var]

                    for i,v in enumerate(idx):

                        if v:

                            df[var][i] = np.random.choice((g.get_group(df['Pclass'][i])).values.flatten())

    #Embarked                 

        idx = df['Embarked'].isnull()

        g = df[idx==False].groupby('Pclass')['Embarked']

        for i,v in enumerate(idx):

            if v:

                df['Embarked'][i] = np.random.choice((g.get_group(df['Pclass'][i])).values.flatten())                   

    #Cabin

        df['Cabin'][df['Cabin'].isnull()]='Missing'

        df['Cabin'] = df['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]

        datasets_cp.append(df)

    return datasets_cp



data_clean = df_fill(data_cleaner,'median')
def prepare_data(datasets):

        datasets_cp = []

        for d in datasets:

            df = d.copy(deep = True)

            df['Family onboard'] = df['Parch'] + df['SibSp']

            df['Title'] = df['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

            new_titles_dict = new_titles(df)

            df['Title'] = df['Title'].replace(new_titles_dict)

            df.drop(columns = ['PassengerId','Name','Ticket'],axis = 1, inplace = True)

            datasets_cp.append(df)

        return datasets_cp

        
train,test =prepare_data(df_fill(data_cleaner,mode = 'sampling'))  

print("Training data")

print(train.isnull().sum())

print("Test data")

print(test.isnull().sum())
def corr_matrix(x,y, quant = None):

    x_quants = x.quantile(quant) if quant else x.quantile([0, 0.25, 0.5, 0.75, 1])

    out = np.zeros((x_quants.shape[0]-1,int(y.unique().max()+1)))

    for i in range(x.shape[0]):

        comp = x[i]<=x_quants

        idx = int(next((j for j,compv in enumerate(comp) if compv),None))

        out[idx-1,int(y[i])]+=1

    return out.T,x_quants



def plot_corr_matrix(x,quants,fig, ax, **kwargs):

    assert x.shape[1] == quants.shape[0]-1

    cmap = kwargs['cmap'] if kwargs['cmap'] else 'Blues'

    ax.set_xlabel(kwargs['xlabel'])

    ax.set_ylabel(kwargs['ylabel'])

    ticks = np.arange(quants.shape[0])

    ax.set_xticks(ticks)

    ax.set_xticklabels(list(quants))

    if 'xlabel' and 'ylabel' in kwargs.keys():

        ax.title.set_text(f"{kwargs['xlabel']} vs {kwargs['ylabel']}")

    p = ax.pcolor(x,cmap = cmap)

    fig.colorbar(p,ax = ax)

    return fig,ax

    

    

def gen_corr_matrix(*args,quant = None,cmap = 'YlOrBr'):

    totalvars = len(args)

    assert totalvars>1

    

    out   = dict()

    out_q = dict()

    fig,axs = plt.subplots(1, totalvars-1, squeeze=False)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    fig.figsize=(800, 800) 

    fig.suptitle("Correlation Matrix") if totalvars<3 else fig.suptitle("Correlation Matrices")

    for i in range(totalvars-1):

        out[i],out_q[i] = corr_matrix(args[0],args[i+1],quant)

        plot_corr_matrix(out[i], out_q[i],

                         fig,

                         axs[0,i],

                         cmap = cmap ,

                         xlabel = args[0].name,

                         ylabel = args[i+1].name)

    plt.show()
gen_corr_matrix(train['Age'],train['Parch'],train['SibSp'],train['Family onboard'])
def scatterplot(x,y):

    fig,ax = plt.subplots()

    ax.scatter(x,y)

    ax.set_xlabel(x.name)

    ax.set_ylabel(y.name)

    ax.grid(True)



    coef = np.polyfit(x,y,1)

    poly1d_fn = np.poly1d(coef) 

    plt.plot(x,y, 'ro', x, poly1d_fn(x), '--k')

    plt.show()

    

scatterplot(train['Age'],train['Fare'])
gen_corr_matrix(df_train['Fare'],df_train['Survived'])
cont_names = ['Fare','Age']

cat_names = ['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Family onboard']

procs = [Categorify]

dep_var = 'Survived'



data_test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)





data = (TabularList.from_df(train, path='/kaggle/working', cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(0.2)

                           .label_from_df(cols = dep_var)

                           .add_test(data_test, label=0)

                           .databunch()

       )
learn = tabular_learner(data, 

                        layers=[1000,500, 200,50, 15],

                        metrics=accuracy,

                        emb_drop=0.1

                       )

torch.device('cuda')

learn.fit_one_cycle(5, 2.5e-2)
learn.export('stage1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(2e-4))
# learn.model

learn.recorder.plot_losses()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(5e-2))
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':labels})
submission.to_csv('submission-fastai.csv', index=False)