import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/lish-moa/train_features.csv")
df.head(10)
df.cp_dose.value_counts()
df.cp_time.value_counts()
df.cp_time = df.cp_time.apply(lambda x : x//24)
df.cp_time.value_counts()
df.rename(columns={'cp_time':'days'}, inplace=True)
df.head()
df.cp_type.value_counts()
g_list=[]
cols=df.columns
for i in cols:
    if 'g-' in i:
        g_list.append(i)
c_list=[]
cols=df.columns
for i in cols:
    if 'c-' in i:
        c_list.append(i)
g_list[-1]
#i=0
def g_sum(x):
#     global i
#     i=i+1
#     print(i)
    return sum(x)
dummy = df.loc[:,"g-0" : "g-771"]
s=dummy.apply(g_sum,axis=1)
s=pd.DataFrame(s)
s.head()
s.shape
dummy = df.loc[:,"c-0" : "c-99"]
s_c=dummy.apply(g_sum,axis=1)
s_c=pd.DataFrame(s_c)
s_c.head()
s_c.shape,df.shape,s.shape
df= pd.concat([df,s_c,s],axis=1)
df.shape
df.head()
df.drop(g_list,axis=1,inplace=True)
df.drop(c_list,axis=1,inplace=True)
df.head()
df.columns=['sig_id', 'cp_type', 'days', 'cp_dose', 'c-type', 'g-type']
df.head()
plt.figure(figsize=(15,9))
sns.countplot(df.cp_dose)
plt.figure(figsize=(15,9))
sns.countplot(df.cp_type,palette="inferno")
plt.figure(figsize=(15,9))
sns.countplot(df.days,palette="rainbow")
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
df.cp_dose=lab.fit_transform(df.cp_dose)
df.cp_type=lab.fit_transform(df.cp_type)
corr= df.corr()
corr.style.background_gradient(cmap='inferno')
df_scored =  pd.read_csv("../input/lish-moa/train_targets_scored.csv")
df_not_scored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
df_scored.head()
df_not_scored.head()
df_scored.shape , df_not_scored.shape
sample_sub = pd.read_csv("../input/lish-moa/sample_submission.csv")
sample_sub.head()
df = pd.read_csv("../input/lish-moa/train_features.csv")
df.head(10)
df.cp_time = df.cp_time.apply(lambda x : x//24)
df.cp_dose=lab.fit_transform(df.cp_dose)
df.cp_type=lab.fit_transform(df.cp_type)
df.head()
df.drop("sig_id",axis=1,inplace=True)
df.head()
df_scored.drop("sig_id",axis=1,inplace =True)
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
model = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))
model.fit(df,df_scored)
import pickle

pkl_path = "./Moa.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(model, f)
with open(pkl_path, 'rb') as f:
    model = pickle.load(f)