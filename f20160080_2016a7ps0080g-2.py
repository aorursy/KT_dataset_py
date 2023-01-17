import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import seaborn as sns



from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
data_orig = pd.read_csv("../input/train.csv", sep=',')

data = data_orig
data.head()
data.info()
data.duplicated().sum()
data.isnull().sum()
data.replace({'?':np.NaN},inplace=True)
data=data.drop(["Enrolled"],1)

data=data.drop(["MLU"],1)

data=data.drop(["Reason"],1)

data=data.drop(["Area"],1)

data=data.drop(["State"],1)

data=data.drop(["PREV"],1)

data=data.drop(["Fill"],1)
data=data.drop(['Worker Class'],1)

data=data.drop(['MIC'],1)

data=data.drop(['MOC'],1)

data=data.drop(['MSA'],1)

data=data.drop(['REG'],1)

data=data.drop(['Live'],1)

data=data.drop(['MOVE'],1)

data=data.drop(['Teen'],1)

data=data.drop(['COB FATHER'],1)

data=data.drop(['COB MOTHER'],1)

data=data.drop(['COB SELF'],1)
data=data.drop(['Schooling'],1)

data=data.drop(['Married_Life'],1)

data=data.drop(['Detailed'],1)
data=data.dropna()
import seaborn as sns

f, ax = plt.subplots(figsize=(20,16))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
data = pd.get_dummies(data, columns=['Cast'], prefix = ['Cast'])

data = pd.get_dummies(data, columns=['Hispanic'], prefix = ['Hispanic'])

data = pd.get_dummies(data, columns=['Sex'], prefix = ['Sex'])

data = pd.get_dummies(data, columns=['Full/Part'], prefix = ['Full/Part'])

data = pd.get_dummies(data, columns=['Tax Status'], prefix = ['Tax Status'])

data = pd.get_dummies(data, columns=['Summary'], prefix = ['Summary'])

data = pd.get_dummies(data, columns=['Citizen'], prefix = ['Citizen'])
y=data['Class']

X=data.drop(['Class'],axis=1)

# X.head()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.60, random_state=42)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

# X_train.head()
np.random.seed(42)
from sklearn.naive_bayes import GaussianNB as NB

#NB?
nb = NB()

nb.fit(X_train,y_train)

nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_val, y_pred_NB))
data_test_orig_test = pd.read_csv("../input/test.csv", sep=',')

data_test = data_test_orig_test





data_test.replace({'?':np.NaN},inplace=True)





data_test=data_test.drop(["Enrolled"],1)

data_test=data_test.drop(["MLU"],1)

data_test=data_test.drop(["Reason"],1)

data_test=data_test.drop(["Area"],1)

data_test=data_test.drop(["State"],1)

data_test=data_test.drop(["PREV"],1)

data_test=data_test.drop(["Fill"],1)



data_test2=data_test



data_test=data_test.drop(['Worker Class'],1)

data_test=data_test.drop(['MIC'],1)

data_test=data_test.drop(['MOC'],1)

data_test=data_test.drop(['MSA'],1)

data_test=data_test.drop(['REG'],1)

data_test=data_test.drop(['Live'],1)

data_test=data_test.drop(['MOVE'],1)

data_test=data_test.drop(['Teen'],1)

data_test=data_test.drop(['COB FATHER'],1)

data_test=data_test.drop(['COB MOTHER'],1)

data_test=data_test.drop(['COB SELF'],1)





data_test=data_test.drop(['Schooling'],1)

data_test=data_test.drop(['Married_Life'],1)

data_test=data_test.drop(['Detailed'],1)



data_test['Hispanic'].replace({np.NaN:'HA',},inplace=True)



data_test = pd.get_dummies(data_test, columns=['Cast'], prefix = ['Cast'])

data_test = pd.get_dummies(data_test, columns=['Hispanic'], prefix = ['Hispanic'])

data_test = pd.get_dummies(data_test, columns=['Sex'], prefix = ['Sex'])

data_test = pd.get_dummies(data_test, columns=['Full/Part'], prefix = ['Full/Part'])

data_test = pd.get_dummies(data_test, columns=['Tax Status'], prefix = ['Tax Status'])

data_test = pd.get_dummies(data_test, columns=['Summary'], prefix = ['Summary'])

data_test = pd.get_dummies(data_test, columns=['Citizen'], prefix = ['Citizen'])





X=data_test



min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X_test = pd.DataFrame(np_scaled)
y_pred_NB = nb.predict(X_test)
print(y_pred_NB)
res1 = pd.DataFrame(y_pred_NB)

final = pd.concat([data_test_orig_test["ID"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final['Class']=final.Class.astype(int)

final.to_csv('submission.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = final.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final)