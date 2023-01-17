import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from dateutil import parser

data = pd.read_csv('../input/app-data/appdata10.csv')
data.head()
data.shape
data.describe()
data['hour'] = data.hour.str.slice(1, 3).astype(int)
data
numerical = [f for f in data if data[f].dtypes!='O']
numerical
numerical.remove('user')
for f in numerical:

    dataC = data.copy()

    dataC[f].hist()

    plt.xlabel(f)

    plt.show()
data2 = data.copy().drop(columns = ['user', 'first_open', 'enrolled_date', 'screen_list', 'enrolled'])
data2
plt.figure(figsize=(25,18))

plt.suptitle('Histograms of Numerical Columns', fontsize=40)

for i in range(1, data2.shape[1] +1):

    plt.subplot(3, 3, i)

    f = plt.gca()

    f.set_title(data2.columns.values[i -1 ])



    vals = np.size(data2.iloc[:, i - 1].unique())

    plt.hist(data2.iloc[:, i - 1], bins=vals, color='blue')

    



plt.show()
data2.corrwith(data['enrolled']).plot.bar(figsize=(15,8), color = 'blue', rot=45, fontsize=15, 

                                          title = 'Correlation with Enrolled (Response) variable')
plt.figure(figsize=(10,15))



sns.heatmap(data2.corr(), square=True, linewidths=.5)

data.dtypes
data['first_open'] = [parser.parse(f) for f in data['first_open']]
data.dtypes
data['enrolled_date'] = [parser.parse(f) if isinstance(f, str) else f for f in data['enrolled_date']]
data.dtypes
data['difference'] = (data['enrolled_date']-data['first_open']).astype('timedelta64[h]')
response_hist = plt.hist(data["difference"].dropna(), color='#3F5D7D')
response_hist = plt.hist(data["difference"].dropna(), color='#3F5D7D', range = (0, 48))
data.loc[data.difference > 48, 'enrolled'] = 0

data = data.drop(columns=['enrolled_date', 'difference', 'first_open'])
data
top_screens = pd.read_csv('../input/app-data/top_screens.csv').top_screens.values

top_screens
len(top_screens)
data['screen_list'] = data.screen_list.astype(str) + ','
data
for sc in top_screens:

    data[sc] = data.screen_list.str.contains(sc).astype(int)

    data['screen_list'] = data['screen_list'].replace(sc+ "," + "")
data.head(20)
data['other'] = data['screen_list'].str.count(',')
data
data = data.drop(['screen_list'], axis=1)
savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9",

                    "Saving10"]
data["SavingCount"] = data[savings_screens].sum(axis=1)

data = data.drop(columns=savings_screens)
cm_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
data["CMCount"] = data[cm_screens].sum(axis=1)

data = data.drop(columns=cm_screens)
cc_screens = ["CC1", "CC1Category", "CC3"]
data["CCCount"] = data[cc_screens].sum(axis=1)

data = data.drop(columns=cc_screens)
loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]
data["LoansCount"] = data[loan_screens].sum(axis=1)

data = data.drop(columns=loan_screens)
datan = data.copy()
Y = data['enrolled']

X = data.drop(['enrolled'], axis=1)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
xtrain.shape
ytrain.shape
# Removing Identifiers



train_identity = xtrain['user']

xtrain = xtrain.drop(['user'], axis = 1)



test_identity = xtest['user']

xtest = xtest.drop(['user'], axis=1)
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()

sc_test = StandardScaler()
xtrain2 = pd.DataFrame(sc_train.fit_transform(xtrain))

xtest2 = pd.DataFrame(sc_test.fit_transform(xtest))
xtrain2.columns = xtrain.columns.values
xtest2.columns = xtest.columns.values
xtrain2.index = xtrain.index.values

xtest2.index = xtest.index.values
xtrain2
xtrain  = xtrain2

xtest = xtest2
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(random_state = 0,  max_iter=100)

model1.fit(xtrain, ytrain)
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=150)

model2.fit(xtrain, ytrain)
y_pred1 = model1.predict(xtest)

y_pred2 = model2.predict(xtest)
model1.score(xtest, ytest)
model2.score(xtest, ytest)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(ytest, y_pred1)

cm2 = confusion_matrix(ytest, y_pred2)
sns.heatmap(cm1, annot= True, fmt='g')

plt.title('Logistic Regression')
sns.heatmap(cm2, annot= True, fmt='g')

plt.title('Random Forest')