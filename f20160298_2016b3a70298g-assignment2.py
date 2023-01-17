!pip install --user numpy

!pip install --user pandas

!pip install --user matplotlib

!pip install --user seaborn

!pip install --user sklearn



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(42)



from sklearn.preprocessing import MinMaxScaler,RobustScaler

from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV,cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import make_scorer, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



%matplotlib inline
df_train= pd.read_csv(r'../input/data-mining-assignment-2/train.csv')

df_test=pd.read_csv(r'../input/data-mining-assignment-2/test.csv')
print(df_train.isnull().sum().sum(),df_test.isnull().sum().sum())
df_train.head()
df_train.info()
df_test.head()
df_test.info()
df_train.select_dtypes(include=['object'])
df_train=pd.get_dummies(df_train,columns=['col2','col11','col37','col44','col56']).drop(columns=['col44_No','col11_No'])

df_test=pd.get_dummies(df_test,columns=['col2','col11','col37','col44','col56']).drop(columns=['col44_No','col11_No'])
df_train.head()
df_test.head()
f, ax = plt.subplots(figsize=(20, 20))

corr = df_train.drop(columns=['ID','Class']).corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
min_max_scaler = MinMaxScaler()

robust_scaler= RobustScaler()
print(df_train.drop(columns=['ID','Class']).duplicated().any(), df_test.drop(columns=['ID']).duplicated().any())
X_train= df_train.drop(columns=['ID','Class'])

y_train= df_train.drop(columns=['ID'])['Class']

X_test=df_test.drop(columns=['ID'])

X_train = pd.DataFrame(robust_scaler.fit_transform(X_train))

X_test = pd.DataFrame(robust_scaler.fit_transform(X_test))
X_test.head()

X_train.head()
X_t1, X_t2, y_t1, y_t2 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
X_train['origin']=0

X_test['origin']=1
X_train1 = X_train.sample(300, random_state=42)

X_test1 = X_test.sample(300, random_state=11)



combi = X_train1.append(X_test1)

y = combi['origin']

combi.drop('origin',axis=1,inplace=True)



model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)

drop_list = []

for i in combi.columns:

    score = cross_val_score(model,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')

    if (np.mean(score) > 0.8):

        drop_list.append(i)

        print(i,np.mean(score))
rf_best = RandomForestClassifier(n_estimators=380, max_depth = 40, min_samples_split = 9)

rf_best.fit(X_t1, y_t1)

rf_best.score(X_t2,y_t2)
features = X_train.columns.values

imp = rf_best.feature_importances_

indices = np.argsort(imp)[::-1]



plt.figure(figsize=(20,20))

plt.bar(range(len(indices)), imp[indices], color = 'b', align='center')

plt.xticks(range(len(indices)), features[indices], rotation='vertical')

plt.xlim([-1,len(indices)])
X1=X_train.drop(columns=[68,8,67,61,63,69,62,59,23,66])

X2=X_test.drop(columns=[68,8,67,61,63,69,62,59,23,66])
rf_best.fit(X1,y_train)

y_test=rf_best.predict(X2)
df_test['Class']=y_test

df_test[['ID','Class']].to_csv('sub7.csv',index=False)
y_test
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df_test[['ID','Class']])