import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.cluster import AgglomerativeClustering as AC

from sklearn.metrics import confusion_matrix
data = pd.read_csv("/kaggle/input/dmassign1/data.csv")
data.head()
data.info()
data['Class'].count()
data.replace('?',np.nan,inplace=True)
data.isnull().sum()>0
sum(data.isnull().sum()>0)
y = data['Class']

x = data.iloc[:,0:-1]
y.head()
x.head()

sum(x.isnull().sum()>0)
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')

imputer = imputer.fit(x.iloc[:, 1:189])

x.iloc[:, 1:189] = imputer.transform(x.iloc[:, 1:189])
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'most_frequent')

imputer = imputer.fit(x.iloc[:, 189:])

x.iloc[:, 189:] = imputer.transform(x.iloc[:, 189:])
sum(x.isnull().sum()>0)
x.drop(['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197','ID'],axis = 1,inplace = True)
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

x=sc.fit_transform(x)
aggclus = AC(n_clusters = 40,affinity='cosine',linkage='average')

y_aggclus= aggclus.fit_predict(x)
pred = pd.Series(y_aggclus+1,index=data.index,dtype = np.float64)

classes = (confusion_matrix(y[:1300],pred[:1300]).argmax(axis=0)+1).astype(np.float64)

pred.replace({cluster+1:classes[cluster]for cluster in range(0,len(classes))},inplace = True)
from sklearn.metrics import accuracy_score

accuracy_score(y[:1300],pred[:1300])
answer=pd.read_csv('/kaggle/input/dmassign1/sample_submission.csv')
answer['Class']=list(pred[1300:].copy())
answer['Class']=answer['Class'].astype(int)
answer.to_csv('prediction_513.csv',index=False)
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

create_download_link(answer)