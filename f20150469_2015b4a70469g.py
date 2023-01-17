import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, SpectralClustering, Birch

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

pd.options.mode.use_inf_as_na = True
data = pd.read_csv("../input/dataset.csv", sep=',')

X1=data[['id']].iloc[175:1031]

y=data['Class'].iloc[0:175]

data.head()
data.replace('?',np.NaN,inplace=True)



data['InstallmentCredit'] = data['InstallmentCredit'].astype('float')

data['Yearly Period'] = data['Yearly Period'].astype('float')



data.fillna(data.mode().iloc[0], inplace=True)

data['Plotsize']=data.Plotsize.astype(str).str.upper()



data['Account2']=data.Account2.astype(str).str.lower()        



data['Sponsors']=data.Sponsors.astype(str).str.upper()        

train = data.drop(['id','Class','Gender&Type','Monthly Period','Housing','Phone'],axis=1)

X=pd.get_dummies(train,columns=['Account1','Motive','History', 'Account2', 'Employment Period','Sponsors', 'Plotsize', 'Plan','#Credits', 'Post', 'Expatriate'])

X.head()
from sklearn import preprocessing



x = X.values

X_normalized = np.zeros_like(x)



X_T = np.transpose(x)

min_max_scaler = preprocessing.Normalizer()

X_normalized = min_max_scaler.fit_transform(X_T).T

pca = PCA(n_components=8)

X_new = pca.fit_transform(X_normalized)
h=[]

spec = SpectralClustering(n_clusters = 3, affinity='sigmoid')

spec.fit(X_new)



pred = spec.labels_



h.append(pred)
score = 0

ans = []
y_p1=[]

for i in range(len(h[0])):

    y_p1.append(h[0][i])
y_p2=[]

y_p3=[]

y_p4=[]

y_p5=[]

y_p6=[]

for i in range(len(y_p1)):

    if y_p1[i]==0:

        y_p2.append(1)

        y_p3.append(2)

        y_p4.append(0)

        y_p5.append(1)

        y_p6.append(2)

    elif y_p1[i]==1:

        y_p2.append(0)

        y_p3.append(1)

        y_p4.append(2)

        y_p5.append(2)

        y_p6.append(0)

    else:

        y_p2.append(2)

        y_p3.append(0)

        y_p4.append(1)

        y_p5.append(0)

        y_p6.append(1)
if accuracy_score(y,y_p1[0:175])>score:

    ans = y_p1

    score = accuracy_score(y,y_p1[0:175])
if accuracy_score(y,y_p2[0:175])>score:

    ans = y_p2

    score = accuracy_score(y,y_p2[0:175])
if accuracy_score(y,y_p3[0:175])>score:

    ans = y_p3

    score = accuracy_score(y,y_p3[0:175])
if accuracy_score(y,y_p4[0:175])>score:

    ans = y_p4

    score = accuracy_score(y,y_p4[0:175])

if accuracy_score(y,y_p5[0:175])>score:

    ans = y_p5

    score = accuracy_score(y,y_p5[0:175])
if accuracy_score(y,y_p6[0:175])>score:

    ans = y_p6

    score = accuracy_score(y,y_p6[0:175])

score
X1['Class']=ans[175:1031]
X1.to_csv("ansfinal.csv",index=False)
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



create_download_link(X1)