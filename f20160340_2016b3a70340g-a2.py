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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



np.random.seed(42)

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
train_df = pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv')

test_df = pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')
#get dummies



train_df=pd.get_dummies(train_df,columns=['col2','col11','col37','col44','col56'])

test_df=pd.get_dummies(test_df,columns=['col2','col11','col37','col44','col56'])
#col11_no and coll44_no can be dropped



train_df = train_df.drop(columns=['col11_No', 'col44_No'])

test_df = test_df.drop(columns=['col11_No', 'col44_No'])
#heatmap for correlation



f, ax = plt.subplots(figsize=(15, 15))

corr = train_df.drop(columns=['ID','Class']).corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
mat = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

#drop the columns with correlation greater that cap

cap = 0.90

col_to_drop = [column for column in mat.columns if any(mat[column] > cap)]
from sklearn.preprocessing import RobustScaler

robust_scaler= RobustScaler()



train_x= pd.DataFrame(robust_scaler.fit_transform(train_df.drop(columns=['ID','Class'])))

train_y= train_df.drop(columns=['ID'])['Class']

test_x= pd.DataFrame(robust_scaler.fit_transform(test_df.drop(columns=['ID'])))
from sklearn.model_selection import train_test_split



train1_x, train2_x, train1_y, train2_y = train_test_split(train_x, train_y, test_size=0.20, random_state=42)
train_x['origin']=0

test_x['origin']=1



from sklearn.model_selection import cross_val_score



X_train1 = train_x.sample(300, random_state=42)

X_test1 = test_x.sample(300, random_state=11)



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
rf_best = RandomForestClassifier(n_estimators=380, max_depth = 6, min_samples_split = 2)

rf_best.fit(train1_x, train1_y)

rf_best.score(train2_x,train2_y)
features = train_x.columns.values

imp = rf_best.feature_importances_

indices = np.argsort(imp)[::-1]



#plot

plt.figure(figsize=(20,20))

plt.bar(range(len(indices)), imp[indices], color = 'b', align='center')

plt.xticks(range(len(indices)), features[indices], rotation='vertical')

plt.xlim([-1,len(indices)])
X1=train_x.drop(columns=[67,8,62,60,63,61,69,59,66,23])

X2=test_x.drop(columns=[67,8,62,60,63,61,69,59,66,23])
rf_best.fit(X1,train_y)

test_y=rf_best.predict(X2)
test_y
test_df['Class']=test_y
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV", filename = "submission_1.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(test_df[['ID','Class']])