import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import LabelEncoder
data_s = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv',

                     index_col='Page')#.rename(columns=pd.to_datetime)

data_s.info()
def extractPageFeatures(df, key1='Page',index=False):

    """

    Input df: pandas DataFrame/Series

    key: string, column name

    index: boolean False: default, return numerical index

            True: returns Page as index

    ==============

    returns pandas DataFrame X with feature columns

    

    Example Use:

    

    s = pd.Series(['2NE1_zh.wikipedia.org_all-access_spider',

    'AKB48_en.wikipedia.org_all-access_spider',

    'Angelababy_zh.wikipedia.org_all-access_mobile'])

    

    Xp = extractPageFeatures(s)

    print(Xp)

        Name        Language    Access      Agent

    --  ----------  ----------  ----------  -------

     0  2NE1        zh          all-access  spider

     1  AKB48       en          all-access  spider

     2  Angelababy  zh          all-access  mobile

    

    """

    fnames = ['Name','Language','Access','Agent']

    fnamedict = dict(zip(range(len(fnames)),fnames))

    if type(df) == pd.DataFrame:

        ser = df[key1]

    else:

        ser = df

    X = ser.str.extract(

    '(.+)_(\w{2})\.wiki.+_(.+)_(.+)',expand=True).rename(columns=fnamedict)

    if index:

        X['Page']=ser.values

        X.set_index('Page',inplace=True)

    return X

# help(extractPageFeatures)
tmp = extractPageFeatures(data_s.index,index=True)

cat_f_names = tmp.columns.tolist()

data_s = data_s.join(tmp)

tmp=None

data_s.iloc[:5,-5:]
lookup_c_i = dict(zip(cat_f_names,[LabelEncoder()]*4))

lookup_c_i
data_s[['{}_en'.format(n) for n in cat_f_names]] = data_s[cat_f_names].apply(

    lambda col: lookup_c_i[col.name].fit_transform(col.astype(str)))

data_s.info()
data_s.head()
#Read language dictionary

lang_dict = pd.read_csv('../input/wikipedia-language-iso639/lang.csv',

                        index_col=0).iloc[:,0].to_dict()

print(list(lang_dict.items())[:5])
out_col = '2015-07-01'

enc_names = ['{}_en'.format(n) for n in cat_f_names]

data = data_s[cat_f_names+[out_col]+enc_names].copy()

data.Language= data.Language.map(lang_dict)

data.dropna(subset=[out_col],inplace=True)

X = data[enc_names]

y = data[out_col]

# y.head()

# data.head()

X.head()
# from sklearn.linear_model import LogisticRegression

# from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.model_selection import GridSearchCV

# md = GradientBoostingRegressor(n_jobs=-1)

# grid_pram = {'C':[.01]}

# grid = GridSearchCV(md,param_grid=grid_pram,scoring='roc_auc',verbose=10,n_jobs=-1)

# grid.fit(X,y.values.flatten())
data[out_col].isnull().value_counts()
sns.set(style="whitegrid")

with sns.plotting_context('notebook',font_scale=2):

    g = sns.factorplot(x='Access', 

                                y=out_col, 

                                hue='Language', 

                                data=data, 

                                palette="colorblind",

                                kind='box',

                                size = 10,

                                aspect = 1.7,

                                legend_out=True,showfliers=False)



    

    g.ax.set_ylim([0,3e3]);

    #move legened outside the plot

#     plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
sns.set(style="ticks")

with sns.plotting_context('notebook',font_scale=2):

    g = sns.factorplot(x='Access', 

                                y=out_col, 

                                hue='Language', 

                                data=data, 

                                palette="colorblind",

                                kind='bar',

                                size = 10,

                                aspect = 1.7,

                                legend_out=True,capsize=.05)



    

#     g.ax.set_ylim([0,3e3]);
sns.set(style="ticks")

with sns.plotting_context('notebook',font_scale=2):

    g = sns.factorplot(x='Language', 

                                y=out_col, 

                                hue='Access', 

                                data=data, 

                                palette="Paired",

                                kind='bar',

                                size=10,aspect=1.6,lw=0,capsize=.08)



    

#     g.ax.set_ylim([0,3e3]);


with sns.plotting_context('notebook',font_scale=1.5):

    fig,axs = plt.subplots(3,1,figsize=(16,16))

    for ax,col  in zip(axs,cat_f_names[1:]):

        sns.countplot(data=data,x=col,orient='h',ax=ax);



g = sns.FacetGrid(data,col='Language',row='Access');

g.map(sns.boxplot,'Language',out_col,showfliers=False);
# help(sns.countplot)