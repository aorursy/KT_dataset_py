# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.tabular import *

from fastai.callbacks import *

import fastai

from fastai.imports import *

import string 

from fastai.metrics import CMScores

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from sklearn.metrics import roc_auc_score
data_path = Path("/kaggle/input/cat-in-the-dat") #!!!!!!!!!!

path = Path(".") # for saving models
def display_top(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)

        

def get_infrequent_valuesXX(data, column, threshold):

    value_counts = data[column].value_counts()

    return list(value_counts[value_counts < threshold].index)
if torch.cuda.is_available():

    defaults.device = torch.device('cuda') # makes sure the gpu is used

else:

    print("not using the gpu")

    defaults.device = torch.device('cpu') # makes sure the cpu is used
# read in test and train data sets

raw_train_df = pd.read_csv(data_path/"train.csv")

raw_test_df = pd.read_csv(data_path/"test.csv")

sample_sub_df = pd.read_csv(data_path/'sample_submission.csv')



display_top(raw_train_df.tail().transpose())
num_train = len(raw_train_df)

raw_test_df['target'] = -1 #so that both data sets have the same width

# save for later split 



# to simplify recoding - put train and test sets together 

full_df = pd.concat([raw_train_df, raw_test_df], axis=0)
# split ord 5 into two seperate chars 

full_df["ord_5a"]=full_df["ord_5"].str[0]

full_df["ord_5b"]=full_df["ord_5"].str[1]
ordinal_features = ['ord_0','ord_3','ord_4','ord_5','ord_5a','ord_5b']
ord1_enc = OrdinalEncoder(categories=[np.array(['Novice','Contributor','Expert','Master','Grandmaster'])])

full_df.ord_1 = ord1_enc.fit_transform(full_df.ord_1.values.reshape(-1,1))
ord2_enc = OrdinalEncoder(categories=[np.array(['Freezing','Cold','Warm','Hot','Boiling Hot','Lava Hot'])])

full_df.ord_2 = ord2_enc.fit_transform(full_df.ord_2.values.reshape(-1,1))
for feat in ordinal_features:

    enc = OrdinalEncoder()

    full_df[feat] = enc.fit_transform(full_df[feat].values.reshape(-1,1))
hex_df = full_df.loc[:,"nom_5":"nom_9"]
hex_1 = lambda x: int(bin(int(x,16))[2:].zfill(36)[:9],2)

hex_2 = lambda x: int(bin(int(x,16))[2:].zfill(36)[9:18],2)

hex_3 = lambda x: int(bin(int(x,16))[2:].zfill(36)[18:27],2)

hex_4 = lambda x: int(bin(int(x,16))[2:].zfill(36)[27:],2)
new_ord_df = pd.DataFrame()

for col in hex_df:

    new_ord_df['%s_1'%col] = hex_df[col].apply(hex_1)

    new_ord_df['%s_2'%col] = hex_df[col].apply(hex_2)

    new_ord_df['%s_3'%col] = hex_df[col].apply(hex_3)

    new_ord_df['%s_4'%col] = hex_df[col].apply(hex_4)
full_df.drop(hex_df.columns,axis=1,inplace=True)
full_df = pd.concat([full_df,new_ord_df],axis=1)

display_top(full_df.tail().transpose())
country_dict = {'Finland':[61.924110,25.748152,'europe',2], 

                'Russia':[61.524010,105.318756,'asia',4], 

                'Canada':[56.130367,-106.346771,'asia',3], 

                'Costa Rica':[9.748917,-83.753426,'sa',1], 

                'China':[35.861660,104.195396,'asia',6], 

                'India':[20.593683,78.962883,'na',5]}

country_df = pd.DataFrame()

country_df['lat'] = full_df.nom_3.apply(lambda x: country_dict[x][0])

country_df['lon'] = full_df.nom_3.apply(lambda x: country_dict[x][1])

country_df['continent'] = full_df.nom_3.apply(lambda x: country_dict[x][2])



full_df = pd.concat([full_df,country_df],axis=1)
for feat in full_df.columns:

    if full_df[feat].dtype == 'object':

        print('Encoding ',feat)

        le = LabelEncoder()

        full_df[feat] = le.fit_transform(full_df[feat].values.reshape(-1,1))
cyclic_days = pd.DataFrame()

cyclic_days['day_sin'] = np.sin(2 * np.pi * full_df['day']/7)

cyclic_days['day_cos'] = np.cos(2 * np.pi * full_df['day']/7)

cyclic_months = pd.DataFrame()

cyclic_months['month_sin'] = np.sin(2 * np.pi * full_df['month']/12)

cyclic_months['month_cos'] = np.cos(2 * np.pi * full_df['month']/12)

full_df = pd.concat([full_df,cyclic_days,cyclic_months],axis=1)
drop_cols = ['id','bin_0','day','month','nom_3']

full_df.drop(drop_cols,axis=1,inplace=True)





train_df = full_df[:len(raw_train_df)]

test_df = full_df[len(raw_train_df)-1:-1]





test_df.drop('target',axis=1,inplace=True)

display_top(test_df.head().T)

print(test_df.shape)
print(len(train_df), len(test_df))

display_top(test_df.tail().transpose())
#  duplicate the zeros to make the groups about the same size 

one = train_df[lambda df: df.target == 1]

zero = train_df[lambda df: df.target ==0]

train_df = train_df.append(one)
dep_var = 'target'

names = set(test_df.columns)

cat_names = [ 'nom_0', 'nom_1', 'nom_2', 'nom_4',

       'nom_5_1', 'nom_5_2', 'nom_5_3', 'nom_5_4', 'nom_6_1', 'nom_6_2',

       'nom_6_3', 'nom_6_4', 'nom_7_1', 'nom_7_2', 'nom_7_3', 'nom_7_4',

       'nom_8_1', 'nom_8_2', 'nom_8_3', 'nom_8_4', 'nom_9_1', 'nom_9_2',

       'nom_9_3', 'nom_9_4',  'continent']

cont_names = list(names - set(cat_names))



cont_names

# for fastai -  need to mark catagoroes and continuous variables

print("len1", len(test_df))



procs = [Categorify, Normalize]

test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(train_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(.2, seed = 42)

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch(bs=2048))
def convert_to_df(d):

    def cat2num(df):

        cat_columns = df.select_dtypes(['category']).columns

        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

        return df

    

    dtrain = cat2num(d.train_ds.x.inner_df)

    dtrain  = dtrain.drop(['target'], axis=1)

    dvalid = cat2num(d.valid_ds.x.inner_df)

    dvalid = dvalid.drop(['target'], axis=1)

    return dtrain, d.train_ds.y.items, dvalid, d.valid_ds.y.items, cat2num(d.test_ds.x.inner_df)





X_train, y_train, X_valid, y_valid,  test_df = convert_to_df(data)







display_top(X_train.head().T)
class Myacc(CMScores):

    def on_epoch_end(self, last_metrics, **kwargs):

        total = self.cm.sum()

        wrong = total - self.cm.diag().sum()

        print("total", total, "wrong", wrong)

        return add_metrics(last_metrics, wrong)



class FastaiRunner:

    def emb_sz_rule(n_cat:int)->int: return min(50, (n_cat+5) // 2)

    def __init__(self, data, X_train, count):

        self.count = count

        def emb_sz_rule(n_cat:int)->int: return min(50, (n_cat+5) // 2)

        self.embedding_dict = {}

        for column in data.x.cat_names:

            self.embedding_dict[column] = emb_sz_rule(len(X_train[column].unique()))

        pre = Myacc()

        self.learn = tabular_learner(data, layers=[100, 100], metrics=[pre, AUROC()],

                                     ps =.3,wd = .1, emb_drop = .2, emb_szs= self.embedding_dict)

        self.path = Path(".")

        

    def fit(self, _, _1):

        # shoukld be 32 or more 

        self.learn.fit_one_cycle(self.count, callbacks=[SaveModelCallback(self.learn, monitor="myacc", mode = "min", name = "savedModel")])



    def predict_proba(self, _):

        self.learn.load("savedModel")

        preds, _ = self.learn.get_preds(DatasetType.Test)

        return  preds



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



classifiers = [

    FastaiRunner(data, X_train, 32),

    RandomForestClassifier(n_estimators=100, oob_score=True),

    LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500),

    LinearDiscriminantAnalysis(),

    GaussianNB(),

    BernoulliNB(),

    DecisionTreeClassifier(),

    XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', 

                     nrounds = 'min.error.idx', num_class = 3, 

                     maximize = False, eval_metric = 'logloss', eta = .1,

                     max_depth = 14, colsample_bytree = .4, n_jobs=-1)

]





submission_df = sample_sub_df.copy()

submission_df.drop(["target"],axis=1,inplace=True)

columns = []

for clf in classifiers:

    name = clf.__class__.__name__

    columns.append(name)

    print("process", name)

    print(X_train.shape)

    clf.fit(X_train, y_train)

    preds = clf.predict_proba(test_df)[:,1]

    print("lenpreds ", len(preds), " len sub", len(submission_df))

    submission_df[name] = preds

    

    

    

submission_df.head()

output = pd.DataFrame(columns=["id", "target"])

output.id = submission_df.id

output.target = submission_df[columns].mean(axis=1)





# each value of preds is a pair with two probs [p0 p1]
print(output.head())

output.to_csv("submission.csv", index=False)


  

    