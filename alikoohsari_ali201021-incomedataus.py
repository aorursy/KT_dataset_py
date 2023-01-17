import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



from sklearn.linear_model import Perceptron, SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_table('/kaggle/input/income-classification/income_evaluation.csv', sep=',', skipinitialspace=True)

for col in df.columns:

    if df[col].dtypes == 'object':

        df[col] = df[col].astype('string')

        df[col] = df[col].str.strip()

df = df.drop('education',axis=1)

df.info()
plt.rc('figure', figsize=(12,8), dpi=100)

gs_kw = {'wspace':0.1, 'hspace':0.5, 'left':0.07,'right':0.93,'top':0.95,'bottom':0.17}

fig, axes = plt.subplots(3,3, gridspec_kw=gs_kw)

for i,col in enumerate(df.select_dtypes(include=['string'])):

    gp = df.groupby([col]).size().to_frame(name='totcount')#.reset_index()

    gp['num<50'] = df[df.income=='<=50K'].groupby(col).size().to_frame()

    gp['num>50'] = df[df.income=='>50K'].groupby(col).size().to_frame()

    gp['per>50'] = gp['num>50']/gp['totcount']

    gp = gp.fillna(0)

    gp = gp.sort_values(by = 'per>50')

    gp.plot(y='per>50', kind='bar', ax=axes[i//3, i%3])

    dic = {}

    for j,idx in enumerate(gp.index):

        dic[idx] = j

    df[col] = df[col].map( dic )
colsnum = df._get_numeric_data().columns

colscat = df.select_dtypes(include=['string']).columns
sns.catplot(x="age", y="income", data=df, kind='point')
X=df.drop(colscat,axis=1)

y=df['income']

X0, X9, y0, y9 = train_test_split( X, y, test_size = 0.3, random_state = 10)





sc = StandardScaler().fit(X0)

X0std = sc.transform(X0)

X9std = sc.transform(X9)
clf = Perceptron(max_iter=20, eta0=0.02, random_state=0)

#clf = SGDClassifier(max_iter=50, eta0=0.03, random_state=0)

#clf = MLPClassifier(max_iter=20, learning_rate_init=0.02)

#clf = DecisionTreeClassifier(criterion='gini')

#clf = KNeighborsClassifier(n_neighbors= 10)





clf.fit(X0std, y0)

y_pred = clf.predict(X9std)





print("Accuracy is:", accuracy_score(y9,y_pred))

#print(confusion_matrix(y9, y_pred))

print(classification_report(y9, y_pred))