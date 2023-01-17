import pandas as pd

import numpy as np

import string



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import ExtraTreesClassifier

import collections



from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler



from sklearn.decomposition import PCA



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
df=pd.read_csv("../input/dataset.csv", sep=",", na_values='?')

Dupes = ['Sponsors', 'Plotsize', 'Account2']

for col in Dupes:

    df[col] = df[col].apply(lambda x: x.lower())

df['Plotsize'] = df['Plotsize'].apply(lambda x: "me" if (x=="m.e.") else x)

df['Motive'] = df['Motive'].apply(lambda x: "q10" if x=="p10" else x)

df['Plotsize'] = df['Plotsize'].apply(lambda x: "1sm" if x=="sm" else "2me" if x=="me" else "3la" if x=="la" else "4xl" if x=="xl" else x)
df_comp = df.copy()

df_comp_mod = df_comp.drop(['id'],1)

target = df_comp['Class']
df_full = pd.DataFrame()

for col in df_comp_mod.columns:

    if(col in ['Account1','History','Motive','InstallmentRate','Tenancy Period','Age','Monthly Period']):

        df_full[col] = df_comp_mod[col].fillna(df_comp_mod[col].mode()[0])

    elif(col in ['InstallmentCredit', 'Yearly Period','Credit1']):

        df_full[col] = df_comp_mod[col].fillna(df_comp_mod[col].mean())

    else:

        df_full[col] = df_comp_mod[col]
le = LabelEncoder()

df_full_oh = df_full.copy()

df_copy = df_full.copy()

for col in df_full.columns:

    if(df_full[col].dtype == np.object):

        le.fit(df_full[col])

        df_full[col] = le.transform(df_full[col])
f, ax = plt.subplots(figsize=(20, 16))

corr = df_full.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot = True);
data = df_full.loc[:, df_full.columns != 'Class'].loc[:174]

targ = df_full['Class'].loc[:174]

model = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, random_state=0)

model.fit(data,targ)

preds = model.predict(data)

dt = dict(zip(model.feature_importances_,np.array(data.columns)))

od = collections.OrderedDict(sorted(dt.items()))

items = []

for key, value in sorted(dt.items()):

    items.append(value)
def one_hot(df,sca = 'std'):

    df_full_oh = df.copy()

    oh_cols = []

    for col in df_full_oh.columns:

        if(df_full_oh[col].dtype == np.object and df_full_oh[col].nunique()>50):

            le.fit(df_full_oh[col])

            df_full_oh[col] = le.transform(df_full_oh[col])

        elif(df_full_oh[col].dtype == np.object):

            oh_cols.append(col)

    df_full_oh = pd.get_dummies(df_full_oh, columns=oh_cols)

    if(sca=='mms'):

        sc = MinMaxScaler()

    else:

        sc = StandardScaler()

    df_full_oh = pd.DataFrame(sc.fit_transform(df_full_oh),columns = df_full_oh.columns)

    return df_full_oh
def reduce_to_three(pred,tar):

    val = pd.DataFrame(pred[:175],columns=['Class'])

    tar = pd.DataFrame(tar.loc[:174],columns=['Class'])

    pred = pd.DataFrame(pred,columns=['Class']) 

    for cls in pred['Class'].unique():

        tst = tar[val['Class']==cls]

        if(len(tst['Class'].value_counts())==0):

            nw_cls = tar['Class'].value_counts().idxmax()

        else:

            nw_cls = tst['Class'].value_counts().idxmax()

        pred['Class'].apply(lambda x: nw_cls if x==cls else x)

    return pred,tar
def best_acc(pred,tar):

    pred,tar = reduce_to_three(pred,tar)  

    best_ac = 0  

    val = pd.DataFrame(pred[:175],columns=['Class'])

    

    combi = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]

    out = pred

    bi=0

    

    for i,comb in enumerate(combi):

        pr_temp = val['Class'].apply(lambda x: comb[0] if x==0 else comb[1] if x==1 else comb[2])

        acc_temp = accuracy_score(pr_temp,tar)

        if(acc_temp>best_ac):

            best_ac = acc_temp

            out = pred['Class'].apply(lambda x: comb[0] if x==0 else comb[1] if x==1 else comb[2])

            

    return best_ac*100,out.as_matrix()
cols_rem = items[:12]



targ = df_full['Class']

df_main = df_copy.copy()

df_main = df_main.loc[:, df_main.columns != 'Class']

df_main.drop(cols_rem,axis=1,inplace=True)

data = one_hot(df_main,'std')



pca1 = PCA(n_components=2)

pca1.fit(data)

T1 = pca1.transform(data)



kmpred = KMeans(n_clusters = 5, random_state = 42).fit_predict(data)

kmacc,kmout = best_acc(kmpred,targ)



plt.title("Predicted KMeans Acc = " + str(kmacc))

plt.scatter(T1[:, 0], T1[:, 1], c=kmout)



plt.show()



print("Model Accuracy: ",kmacc)



if 'Class' in df.columns:

    df.drop(['Class'],axis=1,inplace=True)

df['Class'] = kmout

# df[['id','Class']].loc[175:].to_csv("Submissions/submission.csv",index=False)

sub_df = df[['id','Class']].loc[175:]
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub_df)