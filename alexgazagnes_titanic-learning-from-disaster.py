import os, sys, logging, random, time
from math import ceil
import itertools as it
from collections import OrderedDict, Iterable

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

print(os.listdir("../input"))
PATH = "../input/"
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)
l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info

# import warnings
# warnings.filterwarnings('ignore')
PROJECT     = "Kaggle-Titanic_Machine_Learning_From_Disaster"
DATA        = "data"
SUBMISSIONS = "submissions"
TRAIN_FILE  = "train.csv"
TEST_FILE   = "test.csv"
N_1         = 20
N_2         = 10
N_3         = 30
N_4         = 50 
TEST_SIZE   = 25
CV          = 5
%matplotlib inline
# get_ipython().magic('matplotlib inline')
sns.set()
################################################################################

# ----------------------------------------------------------
# 00-first_dataset_tour.py
# ----------------------------------------------------------

################################################################################



# Find here a first study of the dataset, in which we try to understand and
# give meaning to the dataset.

# We are not trying to solve our problem but to be focused on visualization,
# clenaning and all feature engineering improvements.

# At first we will just study the corelations, the links, the quality and the
# meaning of our dataset. External research and more general considerations may 
# be included in this work
# our first function designed to init a dataframe form a file

def init_df(path, file, precast=False) : 

    # init dataframe
    df = pd.read_csv(path+file, index_col=0)

    # if train df
    if len(df.columns)  == 11 : 
        df.columns  = pd.Index( [   "target", "pclass","name", "sex", "age",
                                    "sibsp","parch","ticket","fare","cabin",
                                    "embarked"], dtype="object")
    # if test df 
    elif len(df.columns )  == 10 : 
        df.columns  = pd.Index( [   "pclass","name", "sex", "age",
                                    "sibsp","parch","ticket","fare","cabin",
                                    "embarked"], dtype="object")
    else : 
        raise ValueError("invalid numb of columns")

    # if needed, change sex and embarled feature in int dtype
    if precast : 
        sex_dict        = {"male":1, "female":0}
        embarked_dict   = {"S":2, "C":1, "Q":0}

        df["sex"]        = df.sex.map(sex_dict)
        df["embarked"]   = df.embarked.apply(lambda x : x if x not in ["S", "C", "Q"] else embarked_dict[x] )

    return df

####

# train and test
train_df = init_df(PATH, TRAIN_FILE)
test_df = init_df(PATH, TEST_FILE)

# for nas : concat train and test df
both_df = train_df.copy().append(test_df)

# we will work on train to have target feature
df = train_df.copy()

# it could be good to keep a copy of our original df
DF = df.copy()

df.head()
# let's have a first tour of our dataframe with some old print functions
def study_global_df(df) :     
    print("data frame dimension :       ")
    print(df.ndim)
    print("\n\ndata frame shape :       ")
    print(df.shape)
    print("\n\ndata frame types :      ")
    print(df.dtypes)
    print("\n\ndata frame index :       ") 
    print(df.index)
    print("\n\ndata frame columns :     ")
    print(df.columns)
    print("\n\ndata frame info :     ")
    print(df.info())


    
####

study_global_df(df)
df.describe()
df.head()
# visualisation
def visualize_global(df) : 
    df.hist(grid=True,bins=50, figsize=(10,10))
    # sns.pairplot(df)

    
####

visualize_global(df)
# in order to have sex and embarked 
def precast_df(df) : 
        # DO NOT FORGET TO MAKE A COPY :):)
        _df = df.copy()
        sex_dict        = {"male":1, "female":0}
        embarked_dict   = {"S":2, "C":1, "Q":0}

        _df["sex"]        = _df.sex.map(sex_dict)
        _df["embarked"]   = _df.embarked.apply(lambda x : x if x not in ["S", "C", "Q"] else embarked_dict[x] )

        return _df        
    
####

_df = precast_df(df)
visualize_global(_df)
df.head()
# pclass, sex, and embarqued are definitively categorcial features, so let's work on this
def visualize_global_cat(df) : 
    for feat in ["embarked", "sex", "pclass"] : 
        sns.factorplot(feat,'target', data=df,size=4,aspect=3)
    
    # other way : 
    # fig, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
    # for ax, feat in zip((ax1,ax2, ax3), ["embarked", "sex", "pclass"]) :
    #     data = df[[feat, "target"]].groupby([feat],as_index=False).mean()
    #     sns.barplot(x=feat, y='target', data=data, size=4,aspect=3, ax=ax)

####

visualize_global_cat(df)
# can we learn something with continuous features ?
def visualize_global_continuous(df) : 
    for feat in ["age", "fare", "sibsp", "parch" ] : 
        data = pd.concat([pd.cut(df[feat], 11, labels=range(11)), df["target"]], axis=1)
        # data.columns = [feat, "target"]
        sns.factorplot(feat, "target", data=data, size=4,aspect=3)

####

visualize_global_continuous(df)
# let's go deeper and visualize our categorical features in depth
def visualize_depth_cat_1(df) : 
    for feat in ["embarked", "sex", "pclass"] : 
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        sns.countplot(x=feat, data=df, ax=axs[0])
        sns.countplot(x='target', hue=feat, data=df, order=[1,0], ax=axs[1])
        data = df[[feat, "target"]].groupby([feat],as_index=False).mean()
        sns.barplot(x=feat, y='target', data=data,ax=axs[2])

####

visualize_depth_cat_1(df)
def visualize_depth_cat_2(df) :
    for feat in ["sex", "pclass", "embarked"] : 
        sns.factorplot("target", col=feat, col_wrap=4,
                    data=df, kind="count", size=3.5, aspect=.8)
        
####

visualize_depth_cat_2(df)
def visualize_depth_continuous_2(df) : 
    for feat in  ["fare", "age", "sibsp", "parch"] : 
        facet = sns.FacetGrid(df, hue="target",aspect=4)
        facet.map(sns.kdeplot,feat,shade= True)
        facet.set(xlim=(0, df[feat].max()))
        facet.add_legend()

####

visualize_depth_continuous_2(df)      
def just_another_fancy_graph_1(df) : 
    fig, axs = plt.subplots(1,3, figsize = (20,5))
    feats = ["embarked", "sex", "pclass"]

    for feat, ax in zip(feats, axs) : 
        for f in df[feat].unique() : 
            age = df[~df["age"].isnull()]
            age = age[age[feat] == f]
            sns.distplot(age["age"], bins=50, ax=ax)
            plt.legend(df[feat].unique(),loc='best')
            plt.title(feat)

####

just_another_fancy_graph_1(df)
def just_another_fancy_graph_2(df) :
    g = sns.FacetGrid(df, col="sex", row="target", margin_titles=True)
    g.map(plt.hist, "age",color="purple");

####

just_another_fancy_graph_2(df)
def just_another_fancy_graph_3(df) :
    fig, ax = plt.subplots(1,1, figsize=(20,5))
    sns.boxplot(x="embarked", y="age", hue="pclass", data=df, ax=ax);

####

just_another_fancy_graph_3(df)
# finally let's see the correlation matrix

def graph_corr_matrix(df) : 
    
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap="coolwarm", annot=True, fmt='.3g', ax=ax)
    plt.title("correlation matrix")
    
####

graph_corr_matrix(df)
def study_nas(df): 
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False).round(3)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

####

study_nas(both_df)
df.head()
# 2 nas for embarked and 1 for fare, easy...

def fill_embarked(df) :
    _df = df.copy()
    _df.loc[_df.embarked.isna(), : ]
    try : 
        _embarked = _df.embarked.value_counts().sort_values(ascending=False).index[0]
        _df["embarked"] = _df.embarked.fillna(_embarked)
    except: 
        pass
    return _df

def fill_fare(df) : 
    _df = df.copy()
    _pclass =  int(_df.loc[_df.fare.isna(),"pclass"].values)
    try : 
        val = _df.loc[_df.pclass == _pclass, "fare"].median()
        _df["fare"] = _df.fare.fillna(val)
    except : 
        pass
    return _df

####

_both_df = fill_embarked(fill_fare(both_df))
study_nas(_both_df)
# but for age it is a much more complex problem! 
# we 20% of nas and know due to obvious logic and due to our data visualization that this a 
# very important feature
# first thing to do could be a statistical fillin strategy based of pclass, sex and embarked

def fill_age_easy(df) : 
    _df = df.copy()
    
    idxs = _df.loc[_df["age"].isna(), : ].index

    # using 'for loop' with df is (strongly)  recommanded
    # this 'for  loop' is just here to increase code readability 
    for i in idxs : 
        pers = _df.loc[i, :]

        mask_1 = _df["pclass"]    == pers["pclass"]
        mask_2 = _df["embarked"]  == pers["embarked"]
        mask_3 = _df["sex"]       == pers["sex"]

        mask = mask_1 & mask_2 & mask_3
        sub_df = _df.loc[mask, :]

        if len(sub_df) > 100 : 
            age_mean = sub_df.age.mean()
            age_std = sub_df.age.std()
        
        else : 
            mask = mask_1 & mask_3
            sub_df = _df.loc[mask, :]

            if len(sub_df) > 100 : 
                age_mean = sub_df.age.mean()
                age_std = sub_df.age.std()

            else : 
                mask = mask_1 
                sub_df = _df.loc[mask, :]
                age_mean = sub_df.age.mean()
                age_std = sub_df.age.std()

        # define a random age based on the specific norma distr of our filtered samples
        age = np.random.randint(age_mean - age_std, age_mean + age_std)
        _df.loc[i, "age"] = int(age)

    return _df
# a much more complex way to fill age - but maybe much more
# correct - could to use a ML algo to predict good values of age
# we will not implement this right now but it could be something like this 

def fill_age_hard(df) : 

    _df = df.copy()

    # separate target and other features 
    features_list = ["target", "cabin", "name", "ticket", "embarked"]
    droped_features = df.loc[:,features_list ]
    _df = df.drop(features_list, axis=1)

    # sep train and test df
    _train_df = _df.loc[df["age"].isna(), :]
    _test_df  = _df.loc[~df["age"].isna(), :]
    print(_train_df.shape, _test_df.shape)

    # split X_train, X_test...
    X, y = _train_df.drop("age", axis=1), _train_df.age
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    print([i.shape for i in (X_train, X_test, y_train, y_test) ])

    model_list = [LogisticRegression]
    for m in model_list : 
        grid = GridSearchCV(m(), {}, cv=5)
        grid.fit(X_train, y_train)
        acc = accuracy_score(grid.predict(X_test), y_test)
        print(acc)

    pred_ages = grid.predict(_test_df).astype(np.uint32)
    _test_df["age"] =  pred_ages


    # merge df with droped features 
    for f in droped_features.columns : 
        _df[f] = droped_features[f] 

    return _df

# finally we can define our fill_nas function, keeping in mind that our fill_age could be strongly improved
# and that we should have our train df and our test df in args

def fill_nas(df, df2=None,  
                 embarked_meth=fill_embarked, 
                 fare_meth=fill_fare,
                 age_meth=fill_age_easy) :

    # merge 2 df if needed
    if isinstance(df2, Iterable) : 
        idx_1 = df.index
        idx_2 = df2.index
        _df = df.copy().append(df2)
    else : 
        _df = df.copy()
        
    #  fill embarked, age, and fare
    _df = embarked_meth(_df)
    _df = fare_meth(_df)
    _df = age_meth(_df)
    
    # if needed re-split train_df and test_df
    if isinstance(df2, Iterable) : 
        df_1 = _df.loc[idx_1, :]
        df_2 = _df.loc[idx_2, :]
        return df_1, df_2
    else :
        return _df
####

# please not the 2 examples are the same
_train_df, _test_df =  fill_nas(train_df, test_df)
_both_df = fill_nas(both_df)
# you just need to re-split both_df in train_df and test_df

df = _train_df.copy()
df.head()
study_nas(df)
# how many non unique values for each feature?
def count_unique(df) : 
    data = pd.DataFrame([len(df[feat].unique()) for feat in df.columns], columns=["unique values"], index=df.columns)
    return data.sort_values(by="unique values", ascending=False)

####

count_unique(df)
# let's have a look to unique - non continuous- values

def study_unique(df) : 

    col = [i for i in df.columns if i not in ("age", "ticket", "name", "cabin", "fare")]
    ser = pd.DataFrame( 
            [ (df[i].unique() if len(df[i].unique())<20 else "too many values", 
              df[i].dtype) for i in col], index=col, columns=["unique", "dtype"])
    return ser


####

study_unique(df)
def detect_outliers(df) : 

    # see global fare info
    print(df.describe().fare)

    print(df.fare.sort_values(ascending=True).iloc[:15])

    # plot info
    fig, axs = plt.subplots(2,1, figsize=(20, 5))
    df.fare.hist(bins=100, ax=axs[0])
    _df = df.loc[df.fare <=50.0, :]
    _df.fare.hist(bins=20, ax=axs[1])

    # regul fare >250
    print(df.fare.sort_values(ascending=False).iloc[:10])


####

detect_outliers(both_df)
# handle outliers
def manage_fare_outliers(df, small=True, high=False) : 
    _df = df.copy()

    if small : 
        # regul fare == 0
        idxs = _df.loc[df["fare"]<1.0 , : ].index
        
        for i in idxs : 
            pers = _df.loc[i, :]
            _pclass = int(pers.pclass)
            sub_df = _df.loc[_df.pclass == _pclass, :]
            fare_mean = sub_df.fare.mean()
            fare_std = sub_df.fare.std()

            _fare = np.random.randint(fare_mean - fare_std, fare_mean + fare_std)
            _df.loc[i, "fare"] = int(_fare)

    if high : 
        _df.loc[_df.fare >260, "fare"] = 260

    return _df

####

_df = manage_fare_outliers(df)
_df.loc[_df.fare == 0, :]
detect_outliers(_df)
df.head()
# we now are going to group every people with the same ticket
# we can consider that there are together, as family or as goup of friends
# this is a very important information because if we are able to "regroup" families / freinds we could enhance 
# significatively our level of information

def group_ticket(df, threshold=2, reg_fare=False) : 
    
    _df = df.copy() 

    # handle wierd tickets 
    _df.loc[df.ticket == "LINE", "ticket"] = 'LINE -1'
      
    _df["group_id"] = np.nan
    _df["group_count"] = np.nan

    # sep nb and letters
    _df["ticket_nb"]  = _df.ticket.apply(lambda i : int(i) if " " not in i else int(i[i.rfind(" ")+1 :]))
    _df["ticket_let"] = _df.ticket.apply(lambda i : "Nan" if " " not in i else i[ : i.rfind(" ")])
    # len(df["_ticket_nb"]) == len(df["_ticket_nb"].unique()) 
    # ticket nb non unique!!! snif snif grrr grrrrr
    
    # we need to simplify this feature
    print(_df.ticket_let.unique().sort())
    _df["ticket_let"] = _df.ticket_let.apply(lambda i : i.replace(".", ""))

    ticket_dict = { 'A/4'          : "A4",
                    'A/5'          : "A5",
                    'A/S'          : "AS",
                    'A4'           : "A4",
                    'A5'           : "A5",
                    'C'            : "C",
                    'CA'           : "CA",
                    'CA/SOTON'     : "CA",
                    'FC'           : "FC",
                    'FCC'          : "FCC",
                    'Fa'           : "FA",
                    'LINE'         : "LINE",
                    'Nan'          : np.nan,
                    'P/PP'         : "PP",
                    'PC'           : "PC",
                    'PP'           : "PP",
                    'SC'           : "SC",
                    'SC/A4'        : "A4",
                    'SC/AH'        : "AH",
                    'SC/AH Basle'  : "Basle",
                    'SC/PARIS'     : "PARIS",
                    'SC/Paris'     : "PARIS",
                    'SCO/W'        : "SCO",
                    'SO/C'         : "SOC",
                    'SO/PP'        : "PP",
                    'SOC'          : "SOC",
                    'SOP'          : "SOP",
                    'SOTON/O2'     : "STON",
                    'SOTON/OQ'     : "STON",
                    'SP'           : "SP",
                    'STON/O 2'     : "STON",
                    'STON/O2'      : "STON",
                    'SW/PP'        : "PP",
                    'W/C'          : "WC",
                    'WE/P'         : "WEP",
                    'WEP'          : "WEP"}

    _df["ticket_let"] = _df.ticket_let.map(ticket_dict)

    # how many tickets per pers
    group_count = _df.ticket.value_counts().to_dict(ticket_dict)
    _group_count = [(i,j) for i,j in group_count.items() if j>=threshold]
    _group_count.sort(key=lambda i : i[1], reverse=True)

    # add group id and group count
    for i, tup in enumerate(_group_count) : 
        t, c = tup
        idxs = _df.loc[df.ticket == t, :].index
        _df.loc[idxs,"group_id"] = i
        _df.loc[idxs,"group_count"] = c

    # regularize fare if needed
    if reg_fare : 
        idxs = _df.loc[~_df.group_count.isna(), :].index        
        _df.loc[idxs, "fare"] =  _df.loc[idxs, "fare"] / _df.loc[idxs, "group_count"] 

    # fill na group_count = 1
    _df["group_count"] = _df.group_count.fillna(1)
    
    _df = _df.drop("ticket", axis=1)
    
    return _df

####

_df = group_ticket(df)
_df.ticket_let.value_counts(dropna=False)
print(DF.ticket.head())
_df.head()
_df.loc[_df.group_count.isna(), : ]
sub_df = df.ticket.value_counts().sort_values(ascending=False)
pd.DataFrame(sub_df[sub_df>2], columns=["ticket"])
def vizualize_group_count_1(df) : 
    fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))
    feat = "group_count"
    sns.countplot(x=feat, data=df, ax=axis1)
    sns.countplot(x='target', hue=feat, data=df, order=[1,0], ax=axis2)
    data = df[[feat, "target"]].groupby(df[feat],as_index=False).mean()
    sns.barplot(x=feat, y='target', order=sorted(df.group_count.unique()), data=data,ax=axis3)
    
####
print(_df.columns)
vizualize_group_count_1(_df) 

def vizualize_group_count_2(df): 
    feat = "group_count"
    facet = sns.FacetGrid(df, hue="target",aspect=4)
    facet.map(sns.kdeplot,feat,shade= True)
    facet.set(xlim=(0, df[feat].max()))
    facet.add_legend()
    
####

vizualize_group_count_2(_df)
def vizualize_group_count_3(df): 
    feat= "group_count"
    g = sns.factorplot("target", col=feat, col_wrap=4,
                        data=df[df[feat].notnull()],
                        kind="count", size=2.5, aspect=.8)
    
####

vizualize_group_count_3(_df)
# what about names? 
# obiouvsly we can separate first name, last name and name title

def sep_names(df) :
    _df = df.copy()
    
    # create a tmp feat
    _df["name_len"] = _df.name.apply(lambda i : len(i))
    _df["_name"]    = _df.name.apply(lambda i : i.split(", "))
    _df["_name"]    = _df["_name"].apply(lambda i : [    i[0],
                                                    i[1][: i[1].find(".")], 
                                                    i[1][i[1].find(" "):]   ]   )

    # split last, first, title
    _df["name_last"]     = _df["_name"].apply(lambda i : i[0])
    _df["title"]         = _df["_name"].apply(lambda i : i[1])
    _df["name_first"]    = _df["_name"].apply(lambda i : i[2])

    # countness ?
    idx = _df.loc[_df.title == "the Countess", :].index
    _df.loc[idx, "title"] = "Countess"
    _df.loc[idx, "name_first"] = "Lucy Noel Martha Dyer-Edwards"

    # sep spouce/second name : 
    _df["name_second"]   = _df.name_first.apply(lambda i : i[i.find("(")+1 : ] if "(" in i else np.nan)
    _df["name_first"]    = _df.name_first.apply(lambda i : i[: i.find(" (")] if "(" in i else i)

    # #################################   
    # df["name_last_count"] = np.nan
    ###################################   
    
    # clean :
    items = ["name_first", "name_second", "name_last"]
    def clean(i) : 
        try     : return i.replace(")", "").replace('""', "").replace('"', "").replace("'", '')
        except  : return i

    for item in items : 
        _df[item] = _df[item].apply(lambda i : clean(i))
        # more pandastic 
        # _df[item] = _df[item].map(clean)
        
    _df = _df.drop(["name", "_name"], axis= 1)

    return _df

####

_df = sep_names(df)
_df.head()
# 20th most poular names
def most_popular_names(df) : 
    _df = sep_names(df)
    ser = _df.name_last.value_counts().sort_values(ascending=False)[:20]
    return pd.DataFrame({"20th most poular names":ser})

####

most_popular_names(both_df)
def visualize_name_len_1(df) : 
    _df = sep_names(df)
    feat = "name_len"
    facet = sns.FacetGrid(_df, hue="target",aspect=4)
    facet.map(sns.kdeplot,feat,shade= True)
    facet.set(xlim=(0, _df[feat].max()))
    facet.add_legend()
    
####

visualize_name_len_1(df)
def visualize_name_count(df) : 
    _df = sep_names(df)
    feat = "name_count"
    facet = sns.FacetGrid(_df, hue="target",aspect=4)
    facet.map(sns.kdeplot,feat,shade= True)
    facet.set(xlim=(0, _df[feat].max()))
    facet.add_legend()
    
####
#_df = group_ticket(df)
# visualize_name_count(_df)
# titles
def study_titles(df) : 
    _df = sep_names(df)
    return pd.DataFrame({"title_freq": _df.title.value_counts(normalize=True).round(2), "title_nb": _df.title.value_counts(),})

####

study_titles(df)
# second names?
def study_second_names(df) : 
    _df = sep_names(df)
    fem = pd.Series(_df.loc[_df.sex == "female", "name_second"].isna().value_counts(), name="female")
    mal = pd.Series(_df.loc[_df.sex == "male", "name_second"].isna().value_counts(), name="male")
    print("name_second count by sex")
    return pd.DataFrame(dict(male=mal, female=fem))

####

study_second_names(df)
# with name, let's build a fancy name, easy to read
def fancy_names(df) :  
    _df = df.copy()
    names = list()
    for i in df.index :
        second = "" if _df.loc[i, "name_second"] is np.nan else str(_df.loc[i, "name_second"]) 
        txt =  _df.loc[i, "title" ] + " "+ _df.loc[i, "name_first"] + " " + second + " " + _df.loc[i, "name_last"]
        txt = txt.replace("  ", " ").replace('"', "").replace("'", "")
        names.append(txt)
    _df["fancy_name"] = names
    return _df

####

_df = fancy_names(sep_names(df))
pd.DataFrame(dict(fancy_name= _df.sort_values(by="name_len", ascending=False).fancy_name.iloc[:20]))
# let's regroup titles in a smaller list
def group_title(df) : 

    _df = df.copy()
    
    # we do not touch to Mr, Mrs, Miss, Master, Rev, Dr
    group_dict_1 = {    "Don"       : "Nobl",  
                        "Mme"       : "Mrs",
                        "Ms"        : "Miss",
                        "Major"     : "Mil",
                        "Lady"      : "Nobl",
                        "Sir"       : "Nobl",
                        "Mlle"      : "Miss",
                        "Col"       : "Mil",
                        "Capt"      : "Mil",
                        "Dona"      : "Nobl",
                        "Countess"  : "Nobl",
                        "Jonkheer"  : "Nobl",  }
    
    group_dict_2 = {"Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "Countess":   "Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

    group_dict = group_dict_2
    _df["title"] = _df.title.apply(lambda i : group_dict[i] if i in group_dict.keys() else i )

    return _df  

####

_df = group_title(sep_names(df))
_df.title.value_counts()
def visualize_titles_1(df):
    _df = group_title(sep_names(df))
    fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))
    feat = "title"

    sns.countplot(x=feat, data=_df, ax=axis1)
    sns.countplot(x='target', hue=feat, data=_df, order=[1,0], ax=axis2)
    data = _df[[feat, "target"]].groupby([feat],as_index=False).mean()
    sns.barplot(x=feat, y='target', data=data,ax=axis3)

####

visualize_titles_1(df)
def visualize_titles_2(df):
    feat = "title"
    _df = group_title(sep_names(df))
    g = sns.factorplot("target", col=feat, col_wrap=4,
                    data=_df[_df[feat].notnull()],
                    kind="count", size=2.5, aspect=.8)
####

visualize_titles_2(df)
# we now have a tricky topic 
# what about people who are together, a family for ex, but with various tickets nb/id?
# we need to tackle this problem, let's do this...

def group_name(df, threshold=2) : 

    try     : 
        _df = sep_names(group_ticket(df))
    except : 
        _df = df.copy()

    # add company
    _df["company"] = _df.parch + _df.sibsp

    # add a specific mask
    name_count = _df.name_last.value_counts().to_dict()
    _df["name_count"] = _df.name_last.apply(lambda i : name_count[i])
    mask = (_df.name_count >1) & ((_df.parch +_df.sibsp) >= threshold)
    sub_df = _df.loc[mask, :].copy()
    sub_df.sort_values(by=["name_count", "name_last", "pclass", "embarked", "age", "group_id"], ascending=False, inplace=True)
    names_list = sub_df.name_last.unique()

    # identify wierd / fake family
    manual_check = list()
    for i, n in enumerate(names_list) : 

        fam = _df.loc[_df.name_last == n , :]
        idxs = fam.index
        nb = len(fam)

        is_wierd = True if (   (len(fam.embarked.unique()) != 1) 
                            or (len(fam.pclass.unique())   != 1)
                            or (len(fam.group_id.unique())   != 1) 
                            or (len(fam.company.unique())  != 1) 
                            or (((fam.company.sum() + nb) / nb) != nb )) else False
        if is_wierd :  manual_check.append(idxs)

    # uncomment if you want to print it 
    # for idxs in manual_check : 
    #     sub_df = _df.loc[idxs, :]
    #     sub_df = sub_df.sort_values(by=["name_last", "group_id", "company", "pclass", "embarked", "age", "sex", ], ascending=False)
    #     print(sub_df.loc[:, [  "name_last", "name_first", "name_second", 
    #                         "pclass", "embarked", "age","sex",
    #                         "cabin_nb", "cabin_let", "parch", "sibsp", "company", "group_id"]])
    #     input()
    #     print(50 * "\n")

    # manualy correct group id 
    try : 
        _df.loc[  1079, "group_id"] = 40.0
        _df.loc[  1025, "group_id"] = 211.0
        _df.loc[ [39, 334], "group_id"] = 163.0
        _df.loc[  530, "group_id"] = 92.0
        _df.loc[ [705, 624], "group_id"]  = 209.0
        _df.loc[ [393,105], "group_id"]  = 1000.0
        _df.loc[ [353, 533, 1229, 774], "group_id"] = 1001.0
        _df.loc[  176, "group_id"] = 201.0
        _df.loc[  1296, "group_id"] = 153.0
        _df.loc[  1197,"group_id"] = 149.0
        _df.loc[  594,"group_id"] = 214.0
        _df.loc[ [1268, 70],"group_id"] = 1002.0
    except: 
        print("both_df expected, method failed")

    # drop useless features
    _df = _df.drop(["name_count", "name_first", "name_second", "name_last"], axis=1)

    return df

####

_df = group_name(both_df)
_df.head()
# finally we have so separate cabin number and cabin letter

def sep_cabins(df) : 

    _df = df.copy()
    # replace cabin by list of str if various cabs for on pers
    _df["_cabin"]    = _df.cabin.apply(lambda i : [str(i), ])
    _df["_cabin"]    = _df._cabin.apply(lambda i : i[0].split(" "))
    _df["_cabin"]    = _df._cabin.apply(lambda i : sorted(i))

    # count how many cabs by pers
    _df["cabin_count"]  = _df["_cabin"].apply(lambda i : len(i) if i[0] != "nan" else 0)

    # drop various cabs and take dirst cab for every people 
    # the split letter (deck ? ) from numb
    _df["_cabin"]    = _df["_cabin"].apply(lambda i : i[0]) 
    _df["cabin_let"] = _df["_cabin"].apply(lambda i : i[0] if i != "nan" else np.nan) 
    _df["cabin_nb"]  = _df["_cabin"].apply(lambda i : int(i[1:]) if ((i != "nan") and (len(i)>1)) else np.nan) 

    # encode letter (deck) as an int
    cabin_let_list  = ["A", "B", "C", "D", "E", "F", "G", "T"]
    cabin_let_dict  = {j:i for i,j in enumerate(cabin_let_list)}
    _df["cabin_let"] = _df["cabin_let"].apply(lambda i :cabin_let_dict[i] if i is not np.nan else np.nan)

    # drop useless features
    _df = _df.drop(["cabin", "_cabin"], axis=1, inplace=False)

    return _df

_df = sep_cabins(both_df)
_df.head()
def visualize_cabins_1(df) : 
    _df = sep_cabins(df)
    fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))
    feat = "cabin_let"

    sns.countplot(x=feat, data=_df, ax=axis1)
    sns.countplot(x='target', hue=feat, data=_df, order=[1,0], ax=axis2)
    data = _df[[feat, "target"]].groupby([feat],as_index=False).mean()
    sns.barplot(x=feat, y='target', order=sorted(_df[feat].unique()), data=data,ax=axis3)

####

visualize_cabins_1(df)
def visualize_cabins_2(df) : 
    _df = sep_cabins(df)
    feat = "cabin_let"

    facet = sns.FacetGrid(_df, hue="target",aspect=4)
    facet.map(sns.kdeplot,feat,shade= True)
    facet.set(xlim=(0, _df[feat].max()))
    facet.add_legend()
    
####

visualize_cabins_2(df)
# Not for now, but find bellow some very useful features we will be happy to find to upgrade accuracy score to 86+%

def is_alone(df) : 
    return None

def din_not_pay_histicket(df) : 
    return None

def is_the_dominant_of_the_family(df) : 
    return None

def is_child(df):
    return None

def is_child_with_mother_survivor(df) : 
    return None

def is_man_just_with_his_wife(df) : 
    return None

def is_wife_with_men_survivor(df):
    return None

def is_child_with_brother_sister_survivor(df):
    return df
# of course we also can separate age into categories : is_baby (-3), is_little_child (3-7), is_child (7-12), is_pre_ado (12-14), is_ado (15-18), is_young (18-25) is_adult_1 (25-35) 
# is_adult_2 (35-45), is_adult_3 (45-55), is pre_old (55-65), is old (65-75) and is_very_old (75-200)...
# in order to have a good feature engineering strategy, we have to enhance our dataframe with dummy features

def add_noises(df) : 
    df["n_noise"] = np.random.randn(len(df))
    df["u_noise"] = np.random.rand(len(df))

    return df 

####

_df = add_noises(df)
_df.head()
# remember to retype all num?
def retype_all_num(df) : 
    _df = df.copy()
    sex_dict        = {"male":1, "female":0}
    embarked_dict   = {"S":2, "C":1, "Q":0}
    _df["sex"]        = _df.sex.map(sex_dict)
    _df["embarked"]   = _df.embarked.apply(lambda x : x if x not in ["S", "C", "Q"] else embarked_dict[x] )
    return _df

# it will be also helpfull to retype all our features into categorical values

def retype_all_cat_1(df) :  
    
    # add name_count, group_count, name_len, cabin_count
    # is_alone, is_with_survivor, is_dominant_male
    
    # is_baby (-3), is_little_child (3 years - 7) is child (7-12), is_preado(12 - 14) is_ado(15-18), is_young(18-25) is_adult1(25-35) is_adult1(35-45) is_adult1(45-55)
    #  is pre_old(55-65) is old(65-75) very old(75-200)
    
    _df = df.copy()
    
    # equivalent to hot encoding / dummy encoding
    features_list = ("cabin_nb", "cabin_let", "group_id", "group_count")
    features_list = [("as_"+i, i) for i in features_list]
    
    for new, old in features_list : 
        _df[new] = 1
        idxs = _df.loc[_df[old].isna(), :].index
        df.loc[idxs, new] = 0

    for new, old in features_list : 
        cat          = pd.Categorical(df[new].unique())
        df[new]      = df[new].astype(cat)

    # fill nas
    df["cabin_nb"]      = df["cabin_nb"].fillna(-1)
    df["cabin_let"]     = df["cabin_let"].fillna(-1)
    df["group_id"]      = df["group_id"].fillna(1234)
    df["group_count"]   = df["group_count"].fillna(0)

    # numerize str features
    group_dict = {j:i for i, j in enumerate(df.title.unique())} 
    df["title"] = df.title.apply(lambda i : group_dict[i] if i in group_dict.keys() else i )                       
    
    sex_dict = {j:i for i, j in enumerate(df.sex.unique())} 
    df["sex"] = df.sex.apply(lambda i : sex_dict[i] if i in sex_dict.keys() else i )                       

    embarked_dict = {j:i for i, j in enumerate(df.embarked.unique())}
    df["embarked"] = df.embarked.apply(lambda i : embarked_dict[i] if i in embarked_dict.keys() else i )                       

    # columns
    columns = df.columns

    # categories
    # bool_cat          = pd.Categorical([0,1])
    pclass_cat          = pd.Categorical(df.pclass.unique(), ordered=True) 
    sex_cat             = pd.Categorical(df.sex.unique())
    embarked_cat        = pd.Categorical(df.embarked.unique())
    group_count_cat     = pd.Categorical(df.group_count.unique(), ordered=True)
    title_cat           = pd.Categorical(df.title.unique())
    # cabin_count_cat     = pd.Categorical(df.cabin_count.unique(), ordered=True)
    cabin_let_cat       = pd.Categorical(df.cabin_let.unique(), ordered=True)
    parch_cat           = pd.Categorical(df.parch.unique(), ordered=True)
    sibsp_cat           = pd.Categorical(df.sibsp.unique(), ordered=True)
    company_cat         = pd.Categorical(df.company.unique(), ordered=True)
    group_id_cat        = pd.Categorical(df.group_id.unique())

    # discrete
    if "pclass" in columns : 
        df['pclass']        = df['pclass'].astype(pclass_cat)
    if "sex" in columns : 
        df['sex']           = df['sex'].astype(sex_cat)
    if "embarked" in columns : 
        df['embarked']      = df['embarked'].astype(embarked_cat)
    if "group_count" in columns :     
        df['group_count']   = df['group_count'].astype(group_count_cat)
    if "title" in columns :     
        df['title']         = df['title'].astype(title_cat)
    # if "cabin_count" in columns :     
    #    df['cabin_count']   = df['cabin_count'].astype(cabin_count_cat)
    if "cabin_let" in columns :    
        df['cabin_let']     = df['cabin_let'].astype(cabin_let_cat)
    if "parch" in columns :  
        df['parch']         = df['parch'].astype(parch_cat)
    if "sibsp" in columns :      
        df['sibsp']         = df['sibsp'].astype(sibsp_cat)
    if "company" in columns :      
        df['company']       = df['company'].astype(company_cat)
    if "group_id" in columns :  
        df['group_id']      = df['group_id'].astype(group_id_cat)

    # continous
    if "age" in columns :  
        df['age']           = pd.cut(df.age,12, labels=range(12))
    if "fare" in columns :  
        df['fare']          = pd.cut(df.fare,12, labels=range(12))
    # if "cabin_nb" in columns :  
    #    df['cabin_nb']      = pd.cut(df.cabin_nb,12, labels=range(12))
    if "u_noise" in columns :  
        df['u_noise']       = pd.cut(df.u_noise,11, labels=range(11))
    if "n_noise" in columns :  
        df['n_noise']       = pd.cut(df.n_noise,11, labels=range(11))

    return df

# _df = retype_all_cat_1(group_name(df))
# _df.dtypes
def retype_all_cat2(df) :
    pass

    #####
    
    # all in one hot dummy var
    
    #####
# let's now draw our first feature engineering startegy
# a very basic one, just for naive models

def num_feature_eng_1(path, train_file, test_file) : 
    
    # init 2 df
    _train_df = init_df(PATH, TRAIN_FILE)
    _test_df = init_df(PATH, TEST_FILE)
    _train_idxs = train_df.index.copy()
    _test_idxs = test_df.index.copy()
    
    # merge them
    _both_df = _train_df.copy().append(_test_df)
    
    # handle them
    _both_df = fill_nas(_both_df)
    _both_df = manage_fare_outliers(_both_df)

    # delete str or nas features
    _both_df = _both_df.drop(["cabin", "name", "ticket"],axis=1)

    # all numerical
    _both_df = retype_all_num(_both_df)
    
    # re split 
    _train_df, _test_df = _both_df.loc[_train_idxs, :], _both_df.loc[_test_idxs, :]
    
    return _train_df, _test_df 

####

# train and test
_train_df, _test_df = num_feature_eng_1(PATH, TRAIN_FILE, TEST_FILE) 
df = _train_df.copy()
df.head()
# samething but with more complex strategy

def num_feature_eng_2(path, train_file, test_file) : 
    
    # init 2 df
    _train_df = init_df(PATH, TRAIN_FILE)
    _test_df = init_df(PATH, TEST_FILE)
    _train_idxs = train_df.index.copy()
    _test_idxs = test_df.index.copy()
    
    # merge them
    _both_df = _train_df.copy().append(_test_df)
    _both_df = fill_nas(_both_df)
    _both_df = manage_fare_outliers(_both_df)
    
    # handle them
    _both_df = group_ticket(_both_df)
    _both_df = sep_names(_both_df)
    _both_df = group_title(_both_df)
    _both_df = group_name(_both_df)
    _both_df = sep_cabins(_both_df)
    
    # drop useless features
    _both_df = _both_df.drop(["group_id", "ticket_nb", "ticket_let", "name_last", "title", "name_first","name_second", "cabin_nb", "cabin_let"], axis=1)

    # all numerical
    _both_df = retype_all_num(_both_df)
    
    # re split 
    _train_df, _test_df = _both_df.loc[_train_idxs, :], _both_df.loc[_test_idxs, :]
    
    return _train_df, _test_df 

####

_train_df, _test_df = num_feature_eng_2(PATH, TRAIN_FILE, TEST_FILE)
df = _train_df.copy()
_train_df.dtypes
# we now are building a much more complex feature engineering strategy

####
# we will join train_df and test_df to apply best feat eng strategy and the we will split train_df and test_df
####

def cat_feature_eng_1(train_df, test_df) : 
    train_index = train_df.index
    test_index  = test_df.index

    df = train_df.copy().append(test_df)

    """
    df = fill_nas(df)
    df = sep_names(df)
    # df = fancy_names(df)
    df = sep_cabins(df)
    df = group_title(df)
    df = group_ticket(df)
    df = group_name(df)
    df = add_noises(df)
    df = retype_all_cat(df)
    """
    
    train_df = df.loc[train_index, :]
    test_df = df.loc[test_index, :]
    
    return train_df, test_df
# _df, _ = cat_feature_eng_1(train_df, test_df)
# _df.head()
def cat_feature_eng_2(df) : 
    """
    dzdzad
    zdazdazd
    azdzaazdza
    dadazdzad  
    """
    return df
################################################################################

# ----------------------------------------------------------
# 01-first_naive_models.py
# ----------------------------------------------------------

################################################################################



# In this second part, we will implement our first logistic regression model.

# We will first implement by hand a naive classifier, then a dummy classifier 
# (who does the same job), and finally a basic logistic regression model.

# Rather than looking at the results of a regression we will implement a 
# function that will test the model x times and that will average the results
# obtained

# split our features from our target

def return_X_y(df) : 
    if "target" in df.columns : 
        X = df.drop("target", axis=1)
        y = df.target
        return X, y  
    else : 
        return df 
    
####

_train_df, _test_df = num_feature_eng_1(PATH, TRAIN_FILE, TEST_FILE) 
df = _train_df.copy()
DF = df.copy()
X,y = return_X_y(df)
print(X.columns)
print(y.name)
print(X.head())
print(y.head())
print(N_1)
print(CV)
print(TEST_SIZE)
# split test and train df/target

def split(X,y=None, size=0.33) :
    if isinstance(y, Iterable) : 
        return train_test_split(X, y, test_size=size)
    else : 
        X,y = return_X_y(X)
        return train_test_split(X, y, test_size=size)

####

X_tr, X_te, y_tr, y_te = tup = split(X,y)
X_train, X_test, y_train, y_test = X_tr, X_te, y_tr, y_te 
for i in tup : print(i.shape)
    

# rather than coding a dummy model from scratch, use sklearn DummyClassifier 

def dummy_model(df, param=None) : 

    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test = split(X,y)

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 

    acc = accuracy_score(y_test, y_pred).round(3)

    return acc, model

####

acc, mod = dummy_model(df)
print(round(acc,4))

ser = pd.Series([dummy_model(df)[0] for i in range(N_2)])
print(round(ser.mean(),4))
print(round(ser.median(),4))

# just for fun, trying to make predictions with a very basic model (no meta 
# params, no features engineering) this one will be our model prediction base
# it is suposed to be better than our DummyClassifier. If not there is a major
# issue...

def basic_model(df, param=None) : 

    X,y     = return_X_y(df)

    X_train, X_test, y_train, y_test = split(X,y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred).round(3)
    
    return acc, model

####

acc, mod = basic_model(df)
print(round(acc,4))

ser = pd.Series([basic_model(df)[0] for i in range(N_2)])
print(round(ser.mean(),4))
print(round(ser.median(),4))

# find here an high level function wich is charged of all basic tasks of a GridSearchCV

def GSCV_basic(model=None,  params=None, df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = DF.copy()
        
    if not model   : model = LogisticRegression()
    if not params  : params = dict() 
    try            : model = model()
    except         : pass
    
    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test = split(X,y)

    grid = GridSearchCV(model, params, 
                        n_jobs=6, 
                        scoring="accuracy",
                        cv=10)

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred).round(3)
    
    return acc, grid


####

acc, mod = GSCV_basic(df=df)
ser = pd.Series([GSCV_basic(df=df)[0] for i in range(N_2)])
print(round(acc,4))
print(round(ser.mean(),4))
print(round(ser.median(),4))
# we now are studying quickly wich model is the best

COLUMNS = [     "LR",       "RC",
                "SVC",      # "Nu",
                "KNN",
                "DT",
                "RF", 
                "Ada", # "Per",      
                "MLP"   ]

MODELS = [      LogisticRegression, RidgeClassifier,
                LinearSVC, # NuSVC,
                KNeighborsClassifier,
                DecisionTreeClassifier, 
                RandomForestClassifier,
                AdaBoostClassifier, # Perceptron, 
                MLPClassifier ]
# as GSCV_basic function, we build a 'meta ML handler' (overkill you said???)

def parse_various_models(  n, df=None, params=None,
                                    models = MODELS, columns= COLUMNS) : 

    if not isinstance(df, pd.DataFrame): 
        df = DF.copy()

    if len(models) != len(columns) : 
        raise ValueError("lens not goods")

    if not params : params = dict()    

    results = [     pd.Series([GSCV_basic(m, df=df)[0] for m in models], 
                        index=columns) for i in range(n)]
    
    results = pd.DataFrame(results, columns=columns)

    return results

####

results = parse_various_models(N_2, df)
# print out raw values
results.iloc[:10, :]
# lets have fancy representation of our results
_results = results.describe().T.sort_values(by="50%", ascending=False)
_results
# graph it 
fig, ax = plt.subplots(1,1, figsize=(20,8))
results.boxplot(ax=ax)
plt.xlabel("models")
plt.ylabel("log_loss score")
plt.title("benchmark various models, without feat eng or meta params")
################################################################################

# ----------------------------------------------------------
# 02-playing_with_LR..py
# ----------------------------------------------------------

################################################################################



# In this third part we will finally start to make real machine learning. We will first parse various feature engineering
# strategies helped by an other magic function

# We will then benchmark the different classification models as well as the impact of the different meta 
# parametres on the relevance of the basic model: number of folds, preprocessing, scoring method, clip
# of the predicted values, etc.

# This work is clearly laborious, but its successful execution depends on our ability to really push 
# our model at best.

# here we have various first considerations about feature engineering 

def merge_age_sex(df, age=14) : 
    _df = df.copy() 
    _df["childness"]  = _df.age.apply(lambda x : (0 if x > 17 else (1 if x > 14 else 2)))  
    _df["status"]     = _df.childness.apply(lambda x : 2 if x > 0 else -1)
    mask = _df["status"] == -1
    _df.loc[mask, "status"] = (_df.loc[mask, "sex"] - 1).abs()
    _df.drop(["age", "sex", "childness"], axis=1, inplace=True)
    return _df


def add_childness(df) : 
    _df = df.copy() 
    _df["childness"] = _df.age.apply(lambda x : (0 if x > 17 else (1 if x > 14 else 2)))
    return _df


def convert_age(df, ages=None, coefs=None) :
    _df = df.copy() 
    if not ages :  ages     = [ 3, 7,14,15,16,17,20,30,40,50,60,70,80,100]
    if not coefs : coefs    = [10,10,10, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0,  0]     
    def converter(x) : 
        for a, c in zip(ages, coefs) : 
            if x <= a :return c
        raise ValueError("error")
    _df["age"] = _df.age.apply( lambda x : converter(x))
    return _df


def regularize_fare(df) : 
    _df = df.copy() 
    # gérer les fare à 0 et les fares à 200+
    _df.loc[_df.fare >200,"fare"] = 220
    for i in range(1,4) : 
        val = _df.loc[_df["pclass"] == i, "fare"].mean()
        _df.loc[(_df.fare == 0.0) & (_df["pclass"] == i), "fare"] = val
    return _df


def add_family(df) : 
    _df = df.copy() 
    try : 
        _df["familly"] = _df.sibsp + _df.parch
    except : 
        ValueError("impossible")
    return _df
# various lambda methods 

nothing             = lambda df : df
drop_age            = lambda df : df.drop(["age"], axis=1)
childness_del_age   = lambda df : drop_age(add_childness(df))
drop_embarked       = lambda df : df.drop(["embarked"], axis=1)
drop_fare           = lambda df : df.drop(["fare"], axis=1)
drop_fare_embarked  = lambda df : drop_fare(drop_embarked(df))
drop_sibsp          = lambda df : df.drop(["sibsp"], axis=1)
drop_parch          = lambda df : df.drop(["parch"], axis=1)
family_del_both     = lambda df : drop_parch(drop_sibsp(add_family(df)))
family_del_sibsp    = lambda df : drop_sibsp(add_family(df))
family_del_parch    = lambda df : drop_parch(add_family(df))
del_sibsp_parch     = lambda df : drop_parch(drop_sibsp(df))
# and here our scenarii to benchmark

METHOD_LIST = []
COLUMN_LIST = []

# about age
# METHOD_LIST = [nothing, convert_age, merge_age_sex, drop_age, add_childness, childness_del_age]
# COLUMN_LIST = ["nothing", "convert_age", "merge_age_sex", "drop_age", "add_childness", "childness_del_age"]

# about embarked
# METHOD_LIST =     [nothing, drop_embarked]
# COLUMN_LIST =     ["nothing", "drop_embarked"]

#  about fare
# METHOD_LIST =   [nothing,   drop_fare,     regularize_fare]
# COLUMN_LIST =   ["nothing", "drop_fare",   "regul_fare",  ]

# # about family 
# METHOD_LIST = [nothing, add_family, drop_sibsp, drop_parch, family_del_both, family_del_sibsp, family_del_parch, del_sibsp_parch]
# COLUMN_LIST = ["nothing", "add_family", 'drop_sibsp', 'drop_parch', 'family_del_both', 'family_del_sibsp', 'family_del_parch', "del_sibsp_parch"]
# we need to have a tool to compare basic model, vs feature engineered model
# our goal is to select the method wich offers us the best accuracy gain
# of course it will be impossible to gain 5 or 10%, but 0.5, or 1% will be fine 

def df_enhance_gain(method, df=None, model=None, params=None) : 
    if not isinstance(df, pd.DataFrame) : 
        df = DF.copy()
    if not model :  model = LogisticRegression()
    if not params : params = dict()
        
    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")

    # init acc
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict(X_te)
    init_acc = accuracy_score(y_te, y_pred)

    # new acc
    _df = df.copy()
    _df = method(_df)
    X_tr, X_te, y_tr, y_te = split(*return_X_y(_df))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict(X_te)
    new_acc = accuracy_score(y_te, y_pred)
    
    return round((new_acc - init_acc) / init_acc,3)
# we now need a tool to handle multiple tests and to give us back a global result dataframe

def benchmark_various_df_enhance(  n, df=None, params=None, model=None,
                                methods = METHOD_LIST, cols = COLUMN_LIST) : 
    if not isinstance(df, pd.DataFrame): 
        df = DF.copy()

    if not model : model =  LogisticRegression()
    if not params : params = dict() 
    if len(methods) != len(cols) : raise ValueError("len do not match")

    results = [ [df_enhance_gain(m, df, model, params) for m in methods] 
                    for i in range(n) ]
    results = pd.DataFrame(results, columns=cols)
    return results

####

# let's try this

METHOD_LIST =   [nothing,   drop_fare,     regularize_fare]
COLUMN_LIST =   ["nothing", "drop_fare",   "regul_fare",  ]

res = benchmark_various_df_enhance(N_1, None, methods=METHOD_LIST, cols=COLUMN_LIST)
# print sorted describe()
res.describe().T.sort_values(by="50%", ascending=False)
res.describe().T.sort_values(by="mean", ascending=False)
# ok it works! even if results are not good, we have a tool to track each feature engineering impact
# in this case we can se that 'drop_fare' method help us to increase gain mean of 1% and median of ... 0.4% 
# Q1 and Q3 are also better with this method, but the std of our results is about 3-4%.
# our accuracy gain is not so good enouth... 

# why the hell is there a 'nothing' method wich do nothing? 
# it is supposed to give us a scale of natural std of our results... 
# boxplot
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.boxplot(ax=ax)
plt.xlabel("fare strategy")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various feat eng regarding fare feature")
# plot mean med, Q1, Q3
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.describe().T.loc[:, ["mean", "50%", "75%", "25%"]].plot(ax=ax)
plt.xlabel("fare strategy")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various feat eng regarding fare feature")
# about embarked
# same method...

METHOD_LIST =     [nothing, drop_embarked]
COLUMN_LIST =     ["nothing", "drop_embarked"]
                   
res = benchmark_various_df_enhance(N_1, None, methods=METHOD_LIST, cols=COLUMN_LIST)
# print sorted describe()
res.describe().T.sort_values(by="50%", ascending=False)
res.describe().T.sort_values(by="mean", ascending=False)
# boxplot
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.boxplot(ax=ax)
plt.xlabel("embarked strategy")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various feat eng regarding emabrked feature")
# plot mean med, Q1, Q3
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.describe().T.loc[:, ["mean", "50%", "75%", "25%"]].plot(ax=ax)
plt.xlabel("embarked strategy")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various feat eng regarding emabrked feature")
# about age
METHOD_LIST = [nothing, convert_age, merge_age_sex, drop_age, add_childness, childness_del_age]
COLUMN_LIST = ["nothing", "convert_age", "merge_age_sex", "drop_age", "add_childness", "childness_del_age"]

res = benchmark_various_df_enhance(N_1, None, methods=METHOD_LIST, cols=COLUMN_LIST)
# print sorted describe()
res.describe().T.sort_values(by="50%", ascending=False)
res.describe().T.sort_values(by="mean", ascending=False)
# boxplot
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.boxplot(ax=ax)
plt.xlabel("age strategy")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various feat eng regarding age feature")
# plot mean med, Q1, Q3
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.describe().T.loc[:, ["mean", "50%", "75%", "25%"]].plot(ax=ax)
plt.xlabel("age strategy")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various feat eng regarding age feature")
# we now will try to ehance our output with thresholding our predictions 

def clipping_results(y_pred, x=0.5) : 
    if not isinstance(y_pred, Iterable) : 
        raise ValueError("y_pred has to be a pd.Series")
    if not(0 <= x <= 1 ) :
        raise ValueError("threshold must be 0.00 --> 0.5")
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.apply(lambda i : 1 if i>=x else 0)
    return y_pred
# compute accuracy gain for one threshold

def clipping_results_gain(k, df, model=None, params=None) :     
    if not isinstance(df, pd.DataFrame) : 
        df = DF.copy()
    if not model  : model = LogisticRegression()
    if not params : params = dict()
    #  info(k)
    X,y = return_X_y(df)
    X_tr, X_te, y_tr, y_te = split(X, y)
    y_test = y_te
    
    grid = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")
    
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]

    init_acc = accuracy_score(y_test, clipping_results(y_pred))

    y_pred = clipping_results(y_pred, k)
    new_acc = accuracy_score(y_test, y_pred)

    return round((new_acc - init_acc) / init_acc,3)
# and benchmark every threshold between 0.0 and 0.5

def benchmark_various_clipping(   n , df=None, params=None, 
                                        model=None, threshold_list=None) :     
    if not isinstance(df, pd.DataFrame) : 
        df = DF.copy()
    if not model :  model = LogisticRegression()
    if not params : params = dict()

    if not threshold_list : 
        threshold_list = np.arange(0.2,0.8, 0.05).round(2)
        # threshold_list = np.arange(0.44,0.66, 0.02).round(2)
        # threshold_list = [round(i/1000, 3) for i in range(10,101)]
        # threshold_list = [round(i/1000, 3) for i in range(10,500, 5)]
    results = [ [clipping_results_gain(k, df, model, params) for k in threshold_list]
                     for _ in range(n)]
    results = pd.DataFrame(results, columns=threshold_list)
    return results

####

res = benchmark_various_clipping(N_1, df)
# print sorted results describe()
res.describe().T.sort_values(by="50%", axis=0, ascending=False)
# boxplot 
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.boxplot(ax=ax)
plt.xlabel("clipping levels")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various clipping strategies")
# plot mean, med, Q1,Q3
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.describe().T.loc[:,  ["mean", "50%", "75%", "25%"]].plot(ax=ax)
plt.xlabel("clipping levels")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various clipping strategies")
# find here a new version of our GSCV function
# but with clipping option

def GSCV_basic(     model, params,  
                    X_train, X_test, y_train, y_test, 
                    clipping=None,
                    n_jobs=6, scoring="accuracy", cv=10): 

    try    : model = model()
    except : pass 

    grid = GridSearchCV(model, params, 
                        n_jobs=n_jobs, 
                        scoring=scoring,
                        cv=cv)

    try : grid.fit(X_train, y_train)
    except Exception as e : print(e)

    if not clipping : 
        try : y_pred = grid.predict(X_test)
        except  Exception as e : print(e)
    else : 
        try :
            y_pred = grid.predict_proba(X_test)[:, 1]
            y_pred = clipping_results(y_pred, clipping)
        except Exception as e: 
            raise e

    try : acc = accuracy_score(y_test, y_pred).round(3)
    except  Exception as e : print(e)

    return acc, grid

# find here 3 dict with meta parametres 

default_params  = {     "penalty":["l2"],
                        "dual":[False],
                        "tol":[0.0001],
                        "C":[1.0],
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["liblinear"],
                        "max_iter":[100],
                        "multi_class":["ovr"],
                        "warm_start":[False],   }

all_params          = { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001, 0.001, 0.1, 1],                   # consider also np.logspace(-6, 2, 9)
                        "C":[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],      # consider also np.logspace(-3, 1, 40)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100, 1000],   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[False, True],   }

all_params2     = { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001, 0.001, 0.01],            # consider also np.logspace(-6, 2, 9)
                        "C":[0.001, 0.01, 0.1, 1, 10],      # consider also np.logspace(-3, 1, 40)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100],                   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[True, False],   }
def _mean(x) : 
    if not isinstance(x, Iterable) : 
        raise ValueError("x must be iter")
    return round(float(sum(x) / len(x)), 3)


def _med(x) : 
    x = sorted(x)
    if not (len(x) % 2) : 
        idx     = len(x) /2
        idx_u   = ceil(idx)
        idx_d   = ceil(idx) - 1
        med = _mean([x[idx_u], x[idx_d]])
    else :
        idx = int(len(x)/2)
        med = x[idx]
    return round(med, 3)


def _mix(x) : 
    mea_x = _mean(x)
    med_x = _med(x)
    return _mean([mea_x, med_x]) 
# in order to be able to parse easly various meta parametres, without wondering if there are
# correct, we need a function which is able to combine a dict of list and one
# other which is able to  select only good parametres

def combine_param_dict(d) : 
    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)
    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]
    return d


def valid_param_dict(model, list_of_dicts, X_train, X_test, y_train, y_test) : 
    good_dicts = list()
    try : _model = model()
    except : _model = model
    for d in list_of_dicts : 
        try : 
            m = GridSearchCV(_model, d, cv=3)
            m.fit( X_train, y_train)
            good_dicts.append(d)
        except : 
            print("params!", end="**")
    return good_dicts

####

d = {"a" : ["a","b","c"], "b": [0,1,2,3,4]}
d = combine_param_dict(d)
d
# and here we have the heart of our meta parametres search
# is job is to parse all feeded params, to combine them,
# to select only good ones, and to compute the average accuracy score

def parse_various_params(   n, model, params, df=None, 
                             meth=None, save=True, feat_ing=None,
                             clipping = None, 
                             n_file=4, name="benchmark_various_params",
                             path="benchmarks/params/",
                             n_jobs=6, scoring="accuracy",cv=5) : 

    if not isinstance(df, pd.DataFrame): 
        df = DF.copy()
    if      meth == None   : meth = _mix
    elif    meth == "mean" : meth = _mean
    elif    meth == "med"  : meth = _med
    elif    meth == "mix"  : meth = _mix
    else                   : raise ValueError("not good method") 

    if not feat_ing : feat_ing = "no feat_ing"
    if not name : name = "benchmark_various_params"
    if not path : path = "benchmarks/params/"
    if not n_file :  n_file =  1

    name = path+name+str(n_file)+".csv"

    if save : 
        txt =   "init file         \n"
        txt +=  "model     : {}      \n".format(model)
        txt +=  "params    : {}      \n".format(params)
        txt +=  "n         : {}      \n".format(n)
        txt +=  "clipping  : {}      \n".format(clipping)
        txt +=  "meth      : {}      \n".format(meth)
        txt +=  "feat_ing  : {}      \n".format(feat_ing)
        txt +=  "\n\n  ********************************************** \n\n"

        with open(name, "w") as f : f.write(txt)

    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test = split(X,y)
    columns = list(params.keys())
    columns.append("acc")
    results = list()

    param_dict = combine_param_dict(params)
    param_dict = valid_param_dict(model, param_dict, X_train, X_test, y_train, y_test)

    for param in param_dict : 
        info("testing param : " + str(param))
        accs = [GSCV_basic(   model, param, 
                                X_train, X_test, y_train, y_test, 
                                clipping=clipping, 
                                n_jobs=n_jobs, scoring=scoring,cv=cv )[0] 
                                for i in range(n)]

        acc = round(meth(accs), 3)
        # grid_param = grid.get_params()
        if save : 
            txt = str(acc) + "," + str(param) + "\n"
            with open(name, "a") as f : f.write(txt)
                
        serie = {i: j[0] for i,j in param.items()}
        serie["acc"] = acc
        results.append(pd.Series(serie))

        info("done")

    results = pd.DataFrame(results, columns =columns )
    results.sort_values(by="acc", ascending=False, inplace=True)
    return results
# let's try a dict params 

small_dict_params    = { "penalty":["l1", "l2"],
                        "dual":[False, True],
                        "tol":[0.0001, 0.001, 0.01],            # consider also np.logspace(-6, 2, 9)
                        "C":[0.001, 0.01, 0.1, 1, 10],      # consider also np.logspace(-3, 1, 40)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100],                   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr","multinomial"],
                        "warm_start":[True],   }
res = parse_various_params( N_1, # should be 10, 20, 30 or 50
                           LogisticRegression, 
                           small_dict_params, 
                           save=False)
# print sorted results describe()
_res = res.sort_values(by="acc", axis=0, ascending=False).iloc[:20, :]
print(len(res))
_res
_res.sort_values(by="acc", axis=0, ascending=False).iloc[20: , :]
# scatter various sub parametres
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(res.tol,res.acc )
plt.xlabel("tol levels")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various accuracy gain for tol levels")
# scatter various sub parametres
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(res.C, res.acc)
plt.xlabel("C levels")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark various accuracy gain for C levels")
# we now are going to check if our best params dict is 'really' can 
# strongly impact our results
# we now are going to check if our best params dict is 'really' can 
# strongly impact our results

best_params_top_5 = list()
for i in range(5) : 
    best_params = _res.drop("acc",axis=1).iloc[i,:].to_dict()
    best_params = {i:[j] for i, j in best_params.items()}
    best_params_top_5.append(best_params)

BEST_PARAMS = dict(     dual            = [False], 
                        penalty         = ["l2"], 
                        tol             = [0.001], 
                        multi_class     = ["multinomial"], 
                        warm_start      = [True], 
                        solver          = ["lbfgs"],
                        C               = [0.1], 
                        fit_intercept   = [True]    )
# compute accuracy gain : with vs without params 

def meta_params_results_gain(model, params, df, re_split=False) : 
    
    try     : model = model()
    except  : pass 

    grid = GridSearchCV(model, dict(), 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")

    grid2 = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")
    
    # init acc
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict(X_te)
    init_acc = accuracy_score(y_te, y_pred)

    # new acc
    if re_split : 
        X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    
    grid2.fit(X_tr, y_tr)   
    y_pred = grid2.predict(X_te)
    new_acc = accuracy_score(y_te, y_pred)

    return round((new_acc - init_acc) / init_acc,3)
# benchmark various params, or just one

def benchmark_various_meta_params(   n, model, params, df=None) : 
    if not isinstance(df, pd.DataFrame) : 
        df = DF.copy()
    if not model : model= LogisticRegression
    if not params : params = BEST_PARAMS
    model = model()
    results = [meta_params_results_gain(model, params, df) for _ in range(n)]
    # results = [[params_results_gain(model, p, df) for _ in range(n)]for p in params]
    results = pd.DataFrame({"params" : results})   
    # results = pd.DataFrame(results, columns=map(str,params))
    return results

####

res = benchmark_various_meta_params(N_1, LogisticRegression, BEST_PARAMS)
# print sorted results describe()
res.describe().T.sort_values(by="mean", ascending=False)
# boxplot 
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.boxplot(ax=ax)
plt.xlabel("best meta parmas")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark best meta params")
# very famous in DL, the pseudo labelling can be a good method to upgrade the accuracy rate
# it works for multi class problems, let's try for our dataset

def pseudo_labelling(df, model=None, params=None) : 
    if not isinstance(df, pd.DataFrame) : 
        df = DF.copy()

    if not model : model = LogisticRegression()
        
    if not params : params = dict()

    X,y = return_X_y(df)
    X_tr, X_te, y_tr, y_te = split(X, y)
    y_test = y_te
    
    grid = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")
    
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict(X_te)

    init_acc = accuracy_score(y_test, y_pred)

    TR = X_tr.copy()
    TR["target"] = y_tr
    TE = X_te.copy()
    TE["target"] = y_pred
    
    new_df = TR.append(TE)
    new_X,new_y = return_X_y(new_df)

    grid.fit(new_X,new_y)

    y_pred = grid.predict(X_te)
    new_acc = accuracy_score(y_test, y_pred)

    return round((new_acc - init_acc) / init_acc,3)

####

res = pd.Series([pseudo_labelling(df) for i in range(N_3)], name="pseudo_labelling")
res.describe()
default_params  =     { "alpha":[1.0],
                        "normalize":[False],
                        "tol":[0.001],
                        "fit_intercept":[True],
                        "class_weight":[None],
                        "solver":["auto"],
                        "max_iter":[None],  }

all_params      =     { "alpha":np.logspace(-5, 3, 9), 
                        "normalize":[False, True],
                        "tol":[1, 0.1, 0.001, 0.0001],
                        "fit_intercept":[True],
                        "class_weight":[None],
                        "solver":['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        "max_iter":[None],  }
        
res = parse_various_params( N_1, # should be 10, 20, 30 or 50 
                           RidgeClassifier, 
                           all_params, 
                           save=False)
_res = res.sort_values(by="acc", axis=0, ascending=False).iloc[:20, :]
print(len(_res))
_res
_res.sort_values(by="acc", axis=0, ascending=False).iloc[20: , :]
# select our best params 
BEST_PARAMS_2 = _res.drop("acc",axis=1).iloc[0,:].to_dict()
BEST_PARAMS_2 = {i:[j] for i, j in BEST_PARAMS_2.items()}
BEST_PARAMS_2

res = benchmark_various_meta_params(N_1, RidgeClassifier, BEST_PARAMS_2)
# print sorted results 
res.describe().T.sort_values(by="mean", ascending=False)
# boxplot 
fig, ax = plt.subplots(1,1, figsize=(20,8))
res.boxplot(ax=ax)
plt.xlabel("best meta parmas")
plt.ylabel("accuracy gain in % (0.01 means 1%)")
plt.title("benchmark best meta params")
# ok we can see that our meta params gain (+1.3%) is very good compared to other meta params gain
_train_df, _test_df = num_feature_eng_1(PATH, TRAIN_FILE, TEST_FILE) 
df = _train_df.copy()
DF = df.copy()
X,y = return_X_y(df)
print(X.columns)
print(y.name)
print(X.head())
print(y.head())
_train_df, _test_df = num_feature_eng_2(PATH, TRAIN_FILE, TEST_FILE) 
df = _train_df.copy()
DF = df.copy()
X,y = return_X_y(df)
print(X.columns)
print(y.name)
print(X.head())
print(y.head())
def feat_strat_gain(feat_eng_1, feat_eng_2, path=None, 
                    train_file=None, test_file=None, model=None, params=None) : 
    if not path : path = PATH
    if not train_file : train_file = TRAIN_FILE
    if not test_file : test_file = TEST_FILE
    if not model :  model = LogisticRegression()
    if not params : params = dict()
        
    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")

    # init acc
    _train_df, _test_df = feat_eng_1(path, train_file, test_file)
    X_tr, X_te, y_tr, y_te = split(*return_X_y(_train_df))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict(X_te)
    init_acc = accuracy_score(y_te, y_pred)

    # new acc
    _train_df, _test_df = feat_eng_2(path, train_file, test_file)
    X_tr, X_te, y_tr, y_te = split(*return_X_y(_train_df))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict(X_te)
    new_acc = accuracy_score(y_te, y_pred)
    
    return round((new_acc - init_acc) / init_acc,3)

####

results = [feat_strat_gain(num_feature_eng_1, num_feature_eng_2) for _ in range(30)]
results = pd.DataFrame({"feat. strat" : results})  
results.head()
results.describe()
results.boxplot()
################################################################################

# ----------------------------------------------------------
# 03-playing_with_RF..py
# ----------------------------------------------------------

################################################################################



# In this 4st part we will play with or Random Forest Classifier and with all feature engineering strategies we can to 
# improve our accuracy score. 



