%%capture 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path #flexible path files

import matplotlib.pyplot as plt #plotting

from fastai import *  

from fastai.tabular import *

import torch #Pytorch

import missingno as msno #library for missing values visualization

import warnings #ignoring warnings

warnings.filterwarnings('ignore')



%matplotlib inline
# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.import os

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
df_train.info()
varnames = list(df_train.columns)

for name in varnames:

    print(name+": ",type(df_train.loc[1,name]))
print("Training Set")

print(df_train.isnull().sum(axis=0))

print("Test Set")

print(df_test.isnull().sum(axis=0))
msno.matrix(df_train)
msno.bar(df_test)
print('Overall survival quota:')

df_train['Survived'].value_counts(normalize = True)
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [10, 10]

plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14)



plt.figure()

fig = df_train.groupby('Survived')['Age'].plot.hist(histtype= 'bar', alpha = 0.7)

plt.legend(('Died','Survived'), fontsize = 13)

plt.xlabel('Age', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.suptitle('Histogram of the ages of survivors and decased ones',fontsize =22)

plt.show()
df_train['Family onboard'] = df_train['Parch'] + df_train['SibSp']

plt.rcParams['figure.figsize'] = [20, 7]

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

    axes[i].legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

axes[0].set_ylabel('Survival rate',fontsize = 18)



for ax in fig.axes:

    plt.sca(ax)

    plt.xticks(rotation=0)



plt.suptitle('Survival rates over Number of relatives onboard',fontsize =22)

plt.show()
plt.rcParams['figure.figsize'] = [7, 5]

plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14) 



plt.figure()

fig = df_train.groupby(['Sex'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.5)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Gender',fontsize =18)

plt.xticks(rotation=0)

plt.ylabel('Survival rate',fontsize = 18)





plt.suptitle('Survival rates over Gender',fontsize =22)

plt.show()
plt.rcParams['figure.figsize'] = [8, 5]

plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14) 



plt.figure()

fig = df_train.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.5)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Ticket Class',fontsize =18)

plt.ylabel('Survival rate',fontsize = 18)

plt.suptitle('Survival rate over Ticket class', fontsize = 22)

plt.xticks(rotation=0)

plt.show()

df_train['Title'] = df_train['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

varnames = list(df_train.columns)

    

print("Training set: " ,list(df_train['Title'].unique()))    

df_test['Title'] = df_test['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

print("Test set: " ,list(df_test['Title'].unique()))    

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

plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14) 



plt.figure()

fig = df_train.groupby(['Title'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.7)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Title',fontsize =16)

plt.xticks(rotation=0)





plt.suptitle('Survival rates over Title',fontsize =22)

plt.show()
df_train['Cabin'][df_train['Cabin'].isnull()]='Missing'

df_train['Cabin'] = df_train['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]
plt.rcParams['figure.figsize'] = [12, 5]

plt.figure()

fig = df_train.groupby(['Cabin'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.9)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Cabin Deck',fontsize =18)

plt.suptitle('Survival rates over Cabin Deck',fontsize =22)

plt.xticks(rotation=0)

plt.show()
plt.rcParams['figure.figsize'] = [10, 5]

plt.figure()

fig = df_train.groupby(['Embarked'])['Survived'].value_counts(normalize=True).unstack().plot.bar(width = 0.7)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Embarking Port',fontsize =18)

plt.suptitle('Survival rates over embarking port',fontsize =22)

plt.xticks(rotation=0)

plt.show()
df_train.groupby(['Embarked'])['Pclass'].value_counts(normalize=True).unstack()
df_train.corr(method='pearson')['Age'].abs()
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
cont_names = ['Fare','Age','Pclass','SibSp','Parch','Family onboard']

cat_names = ['Sex','Cabin','Embarked']

procs = [Categorify,Normalize]

dep_var = 'Survived'



data_test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)



data = (TabularList.from_df(train, path='/kaggle/working', cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(0.2)

                           .label_from_df(cols = dep_var)

                           .add_test(data_test, label=0)

                           .databunch()

       )
learn = tabular_learner(data, 

                        layers=[500,200,100],

                        metrics=accuracy,

                        emb_drop=0.1,

                       )



learn.model
torch.device('cuda')

learn.fit_one_cycle(2, 2.5e-2)

learn.save('stage1')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(4e-1))

learn.save('stage2')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-2))

learn.save('stage3')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(5e-3))

learn.save('stage4')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(9e-4))

learn.save('stage5')
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(5e-5))

learn.save('stage6')
learn.recorder.plot_losses()
# learn.load('stage6')

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':labels})
submission.to_csv('submission-fastai.csv', index=False)