# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import seaborn as sns

plt.style.use("fivethirtyeight")

%matplotlib inline

plt.rcParams['figure.figsize']=10,6

import warnings

warnings.filterwarnings("ignore")



import textwrap
from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL
train_df = pd.read_csv("../input/lish-moa/train_features.csv")

test_df = pd.read_csv("../input/lish-moa/test_features.csv")



train_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
print("Shape of Training Data...",train_df.shape)

print("Shape of Testing Data...",test_df.shape)

print("Shape of Trainscored Data...",train_scored.shape)

print("Shape of Trainnonscored Data...",train_nonscored.shape)
#######################################################################

## Helper Function##

#######################################################################





    

def plott(f1):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15,5))

    #1 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((1,3),(0,0))

    plt.hist(cp24[f1], bins=3, color='deepskyblue',alpha=0.5)

    plt.title(f'Treatment duration 24h: {f1}',weight='bold', fontsize=14)

    #first row sec col

    ax1 = plt.subplot2grid((1,3),(0,1))

    plt.hist(cp48[f1], bins=3, color='lightgreen',alpha=0.5)

    plt.title(f'Treatment duration 48h: {f1}',weight='bold', fontsize=14)

    #first row 3rd column

    ax1 = plt.subplot2grid((1,3),(0,2))

    plt.hist(cp72[f1], bins=3, color='gold',alpha=0.5)

    plt.title(f'Treatment duration 72h: {f1}',weight='bold', fontsize=14)

    plt.show()



def plotf(f1, f2, f3, f4):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')



    fig= plt.figure(figsize=(15,10))

    #2 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((2,2),(0,0))

    sns.distplot(train_df[f1], color='crimson')

    plt.title(f1,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #first row sec col

    ax1 = plt.subplot2grid((2,2), (0, 1))

    sns.distplot(train_df[f2], color='gainsboro')

    plt.title(f2,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #Second row first column

    ax1 = plt.subplot2grid((2,2), (1, 0))

    sns.distplot(train_df[f3], color='deepskyblue')

    plt.title(f3,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #second row second column

    ax1 = plt.subplot2grid((2,2), (1, 1))

    sns.distplot(train_df[f4], color='black')

    plt.title(f4,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')



    return plt.show()



def ploth(data, w=15, h=9):

    plt.figure(figsize=(w,h))

    sns.heatmap(data.corr(), cmap='hot')

    plt.title('Correlation between targets', fontsize=18, weight='bold')

    return plt.show()



# corrs function: Show dataframe of high correlation between features

def corrs(data, col1='Gene 1', col2='Gene 2',rows=5,thresh=0.8, pos=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53]):

        #Correlation between genes

        corre= data.corr()

         #Unstack the dataframe

        s = corre.unstack()

        so = s.sort_values(kind="quicksort", ascending=False)

        #Create new dataframe

        so2= pd.DataFrame(so).reset_index()

        so2= so2.rename(columns={0: 'correlation', 'level_0':col1, 'level_1': col2})

        #Filter out the coef 1 correlation between the same drugs

        so2= so2[so2['correlation'] != 1]

        #Drop pair duplicates

        so2= so2.reset_index()

        pos = pos

        so3= so2.drop(so2.index[pos])

        so3= so2.drop('index', axis=1)

        #Show the first 10 high correlations

        cm = sns.light_palette("Red", as_cmap=True)

        s = so3.head(rows).style.background_gradient(cmap=cm)

        print(f"{len(so2[so2['correlation']>thresh])/2} {col1} pairs have +{thresh} correlation.")

        return s
train_df.head()
sns.countplot(x=train_df['cp_type'],data=train_df)

plt.xlabel("cp_type",weight='bold',fontsize = 14)

plt.ylabel("Count",weight='bold',fontsize = 14)

plt.title("Count of cp_type",weight='bold',fontsize = 18)

plt.show()
def bar_chart(df,parameter, figsize=(10,6)):

    target_counts = df[parameter].value_counts()

    target_perc = target_counts.div(target_counts.sum(), axis=0)

    plt.figure(figsize=figsize)

    ax = sns.barplot(x=target_counts.index.values, y=target_counts.values, order=target_counts.index)

#     plt.xticks(rotation=90)

    plt.xlabel(f'{parameter}', weight ='bold',fontsize=16)

    plt.ylabel('# of occurances', weight ='bold',fontsize=16)

    plt.title("Count of "+f'{parameter}', weight ='bold',fontsize=20)



    rects = ax.patches

    labels = np.round(target_perc.values*100, 2)

    for rect, label in zip(rects, labels):

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2, height + 5, f'{label}%', ha='center', va='bottom')

    

    try:

        labels =target_counts.index.tolist()



    #     labels.sort()

        labels=[textwrap.fill(text,12) for text in labels]

        pos = np.arange(len(labels)) 

        plt.xticks(pos, labels,fontsize=12)

    except:

        pass

bar_chart(train_df,"cp_time")
bar_chart(train_df,"cp_dose")
def distribution1(feature, color):

    plt.figure(figsize=(15,7))

    plt.subplot(121)

    sns.distplot(train_df[feature],color=color)

    plt.subplot(122)

    sns.violinplot(train_df[feature])

    print("{}Max value of {} is: {} {:.2f} \n{}Min value of {} is: {} {:.2f}\n{}Mean of {} is: {}{:.2f}\n{}Standard Deviation of {} is:{}{:.2f}"\

      .format(y_,feature,r_,train_df[feature].max(),g_,feature,r_,train_df[feature].min(),b_,feature,r_,train_df[feature].mean(),m_,feature,r_,train_df[feature].std()))
distribution1("g-1","blue")
train_df['g_mean'] = train_df[[x for x in train_df.columns if x.startswith("g-")]].mean(axis=1)

test_df['g_mean'] = test_df[[x for x in test_df.columns if x.startswith("g-")]].mean(axis=1)



distribution1("g_mean","yellow")
#Distribution of single cell viability

distribution1("c-0","green")
#Distribution of mean of cell viability¶

train_df['c_mean'] = train_df[[x for x in train_df.columns if x.startswith("c-")]].mean(axis=1)

test_df['c_mean'] = test_df[[x for x in test_df.columns if x.startswith("c-")]].mean(axis=1)



distribution1('c_mean','orange')
#Distribution of g_mean based on cp_type,cp_time, cp_dose

def distribution2(feature):

    plt.figure(figsize=(15,14))

    plt.subplot(231)

    for i in train_df.cp_type.unique():

        sns.distplot(train_df[train_df['cp_type']==i][feature],label=i)

    plt.title(f"{feature} based on cp_type")

    plt.legend()



    plt.subplot(232)

    for i in train_df.cp_time.unique():

        sns.distplot(train_df[train_df['cp_time']==i][feature],label=i)

    plt.title(f" {feature}  based on cp_time")

    plt.legend()

    

    plt.subplot(233)

    for i in train_df.cp_dose.unique():

        sns.distplot(train_df[train_df['cp_dose']==i][feature],label=i)

    plt.title(f" {feature} based on cp_dose ")

    

    plt.subplot(234)

    sns.violinplot(data=train_df,y=feature,x='cp_type')

    plt.title(f"{feature} based on cp_type")

    plt.legend()



    plt.subplot(235)

    sns.violinplot(data=train_df,y=feature,x='cp_time')

    plt.title(f" {feature}  based on cp_time")

    plt.legend()

    

    plt.subplot(236)

    sns.violinplot(data=train_df,y=feature,x='cp_dose')

    plt.title(f" {feature} based on cp_dose ")

    plt.legend()
distribution2('g_mean')
#Distribution of c_mean based on cp_type,cp_time, cp_dose

distribution2('c_mean')
#Gene exp of 4 random samples

g_cols = [x for x in train_df.columns if x.startswith("g-")]

c_cols = [x for x in train_df.columns if x.startswith("c-")]

def plot1(features):

    rnd = np.random.randint(0,train_df.shape[0]-16)

    plt.figure(figsize=(10,7))

    

    for i in range(4):

        data = train_df.loc[rnd+i,features]

        mean = np.mean(data.values)

        plt.subplot(2,2,i+1)

        sns.scatterplot(data=data.values,marker=">") 

        plt.tick_params(

        axis='x',          

        which='both',      

        bottom=False,    

        top=False,        

        labelbottom=False)

        sns.lineplot(x=list(range(len(data))), y = [mean]*len(data),color='r',linewidth=2)

        

    plt.show()

plot1(g_cols)
#cell validity of 4 random sample

plot1(c_cols)
#Heat map of n random gene exp columns¶

def heat(n):

    plt.figure(figsize=(13,13))

    rnd = np.random.randint(0,len(g_cols)-n)

    data = train_df[g_cols]

    data = data.iloc[:,rnd:rnd+n]

    sns.heatmap(data.corr())

    plt.show()
heat(30)
#Count of top 50 targets

df = train_scored.iloc[:,1:].sum(axis=0).sort_values(ascending=True)[-50:]
df = df.sort_values(ascending=False)
df
plt.figure(figsize=(8,13))

sns.barplot(y=df.index.tolist(),x=df)

plt.show()
# count of lowest 50 target

df = train_scored.iloc[:,1:].sum(axis=0).sort_values(ascending=True)[:50]

df = df.sort_values(ascending= False)
plt.figure(figsize=(8,13))

sns.barplot(y=df.index.tolist(),x=df)

plt.show()
data = train_df.merge(train_scored,on='sig_id')

top_50 = train_scored.drop("sig_id",axis=1).columns[train_scored.iloc[:,1:].sum(axis=0)>=89]

bottom_50 = train_scored.drop("sig_id",axis=1).columns[train_scored.iloc[:,1:].sum(axis=0)<=19]

data_top_50 = data[data[top_50].any(axis=1)][g_cols]

data_bottom_50  = data[data[bottom_50].any(axis=1)][g_cols]
#Plotting of mean of gene exp for top 50

plt.figure(dpi=70)

sns.distplot(data_top_50.mean(axis=1),color='violet')

plt.show()
#random 4 gene exp from top 50

def plot2(df):

    rnd = np.random.randint(0,df.shape[0]-5)

    plt.figure(figsize=(10,7))

    

    for i in range(4):

        data = df.iloc[rnd+i,:]

        mean = np.mean(data.values)

        plt.subplot(2,2,i+1)

        sns.scatterplot(data=data.values,marker=">") 

        plt.tick_params(

        axis='x',          

        which='both',      

        bottom=False,    

        top=False,        

        labelbottom=False)

        sns.lineplot(x=list(range(len(data))), y = [mean]*len(data),color='r',linewidth=2)

        

    plt.show()
plot2(data_top_50)
#Plot of mean of gene exp for top 50

plt.figure(dpi=80)

sns.distplot(data_bottom_50.mean(axis=1),color='blue')

plt.show()
#random 4 gene exp from bottom 50

plot2(data_bottom_50)
#Top 50 sample with highest count of target

df = train_scored.iloc[:,1:].sum(axis=1).sort_values(ascending=True)[-50:]
plt.figure(figsize=(8,13))

sns.barplot(y=df.index.tolist(),x=df)

plt.show()
fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(9, 9))

sns.distplot(train_df['c-0'] ,color="b", kde_kws={"shade": True}, ax=axs[0][0] )

sns.distplot(train_df['c-1'] ,color="r", kde_kws={"shade": True}, ax=axs[0][1] )

sns.distplot(train_df['c-2'], color="g", kde_kws={"shade": True}, ax=axs[1][0] )

sns.distplot(train_df['c-3'] ,color="y", kde_kws={"shade": True}, ax=axs[1][1] )

plt.show()
DO_EDA = True 
if DO_EDA:

    # sample_cols = ['g-0']

    # Create a sampled dataframe and use hue to denote different histograms?



    sns.set_context('poster')



    ax = sns.distplot(train_df['g-0'])

    ax2 = sns.distplot(train_df['g-100'])

    ax3 = sns.distplot(train_df['g-200'])

    ax4 = sns.distplot(train_df['g-300'])

    ax5 = sns.distplot(train_df['g-400'])

    ax6 = sns.distplot(train_df['g-500'])

    ax7 = sns.distplot(train_df['g-600'])

    ax8 = sns.distplot(train_df['g-700'])

    ax9 = sns.distplot(train_df['g-750'])

    ax10 = sns.distplot(train_df['g-150'])





    ax.set(title = "Regulation of 10 Random Genes",

          xlabel = "Upregulation or Downregulation",

          ylabel = "Percent of Sample")



    plt.annotate("Gene Deeply Downregulated", xy = (-9.9, .01), xytext = (-7.8, 0.21),

                 size = 16,

                 arrowprops = {'facecolor':'grey', 'width':3})



    plt.annotate("Somewhat Downregulated", xy = (-5, 0.05), xytext = (-7.8, 0.11), size = 16,

                arrowprops = {'facecolor':'grey', 'width':3},

                backgroundcolor = 'white')



    plt.annotate("Genes Upregulated.  Slight Right Skew", xy = (2.5, 0.06), xytext = (2.5, 0.11), size = 16,

                arrowprops = {'facecolor':'grey', 'width':3},

                backgroundcolor = 'white')



    plt.legend()

    plt.show()
if DO_EDA:

    sns.set_context('poster')



    ax = sns.distplot(train_df['c-0'])

    ax2 = sns.distplot(train_df['c-10'])

    ax3 = sns.distplot(train_df['c-20'])

    ax4 = sns.distplot(train_df['c-30'])

    ax5 = sns.distplot(train_df['c-40'])

    ax6 = sns.distplot(train_df['c-50'])

    ax7 = sns.distplot(train_df['c-60'])

    ax8 = sns.distplot(train_df['c-70'])

    ax9 = sns.distplot(train_df['c-80'])

    ax10 = sns.distplot(train_df['c-90'])



    ax.set(title = "Viability of 10 Random Cell Samples",

          xlabel = "Increased or decreased viability",

          ylabel = "Percent of Sample")



    plt.annotate("Drug effective at killing cells / Error?", xy = (-9.9, .08), xytext = (-7.8, 0.21),

                 size = 16,

                 arrowprops = {'facecolor':'grey', 'width':3})



    plt.annotate("More cells are killed in general", xy = (-4, 0.02), xytext = (-7.8, 0.11), size = 16,

                arrowprops = {'facecolor':'grey', 'width':3},

                backgroundcolor = 'white')



    plt.annotate("Cell viability enhanced less often", xy = (1.5, 0.06), xytext = (2.5, 0.11), size = 16,

                arrowprops = {'facecolor':'grey', 'width':3},

                backgroundcolor = 'white')



    plt.legend()

    plt.show()
plt.style.use('seaborn')

# sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.countplot(x='cp_type', data=train_df, palette='rainbow', alpha=0.75)

plt.title('Train: Control and treated samples', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(x='cp_dose', data=train_df, palette='Purples', alpha=0.75)

plt.title('Train: Treatment Doses: Low and High',weight='bold', fontsize=18)

plt.show()
plt.figure(figsize=(15,5))

sns.distplot( train_df['cp_time'], color='red', bins=5)

plt.title("Train: Treatment duration ", fontsize=15, weight='bold')

plt.show()
plotf('g-10','g-100','g-200','g-400')
train_df.head()
from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from category_encoders import CountEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss



from sklearn.multioutput import MultiOutputClassifier



import os
SEED = 42

NFOLDS = 5

DATA_DIR = "/kaggle/input/lish-moa/"

np.random.seed(SEED)
train = pd.read_csv(DATA_DIR+"train_features.csv")

targets = pd.read_csv(DATA_DIR + "train_targets_scored.csv")



test = pd.read_csv(DATA_DIR+"test_features.csv")

sub = pd.read_csv(DATA_DIR+"sample_submission.csv")



# drop id col

X = train.iloc[:,1:].to_numpy()

X_test = test.iloc[:,1:].to_numpy()

y = targets.iloc[:,1:].to_numpy()
classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))



clf = Pipeline([('encode', CountEncoder(cols=[0, 2])),

                ('classify', classifier)

               ])
params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



_ = clf.set_params(**params)
oof_preds  = np.zeros(y.shape)

test_preds = np.zeros((test.shape[0],y.shape[1]))

oof_losses = []

kf = KFold(n_splits=NFOLDS)



for fn,(trn_idx,val_idx) in enumerate(kf.split(X,y)):

    print("Starting fold: ",fn)

    X_train,X_val = X[trn_idx],X[val_idx]

    y_train,y_val = y[trn_idx],y[val_idx]

    

    # drop where cp_type == ctl_vehicle(baseline)

    ctl_mask = X_train[:,0]=="ctl_vehicle"

    X_train = X_train[~ctl_mask,:]

    y_train = y_train[~ctl_mask]

    

    clf.fit(X_train,y_train)

    val_preds = clf.predict_proba(X_val) # list of preds per class

    

    val_preds = np.array(val_preds)[:,:,1].T  # take the positive class

    

    oof_preds[val_idx] = val_preds

    

    loss = log_loss(np.ravel(y_val),np.ravel(val_preds))

    oof_losses.append(loss)

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T # take the positive class

    test_preds +=preds/NFOLDS

    

print(oof_losses)

print("Mean OOF loss across folds",np.mean(oof_losses))

print("STD OOF loss across folds",np.std(oof_losses))
# set control train preds to 0h

control_mask = train["cp_type"] == "ctl_vehicle"

oof_preds[control_mask] = 0



print("OOF log loss: ",log_loss(np.ravel(y),np.ravel(oof_preds)))
# Analysis of OOF preds



# set control test preds to 0

control_mask = test["cp_type"]=='ctl_vehicle'



test_preds[control_mask] = 0
# Create the submission file



sub.iloc[:,1:] = test_preds

sub.to_csv("submission.csv",index=False)
# Still working in progress!!