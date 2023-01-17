%matplotlib inline 
import pandas as pd

import json

from pandas.io.json import json_normalize

import seaborn as sns
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df1 = pd.read_csv('../input/train.csv', index_col=0)
from IPython.display import display, HTML

display(HTML(df1.head().to_html(index=False)))
df1.dtypes
df1.skew()
import matplotlib.pyplot as plt

df1.corr()
df1.groupby('Cover_Type').size()
df1.shape
def plot_top_10(df,target_attr=None):

    corr = []

    tf = df.corr()

    for i in range(54):

        for j in range(54):

            val = tf.iloc[i,j]

            if (val >= 0.5 and val < 1) or (val < 0 and val <= -0.5):

                corr.append((tf.iloc[i,j],i,j))

    corr_sort = sorted(corr, key=lambda x:-abs(x[0]))[:10]

    cols = df.columns

    for v,i,j in corr_sort:

        if target_attr is None:

            sns.pairplot(df, size=5, x_vars=cols[i],y_vars=cols[j] )

        else:

            sns.pairplot(df, size=5, hue=target_attr,x_vars=cols[i],y_vars=cols[j] )

        plt.show()
plot_top_10(df1,'Cover_Type')
def show_voilin_plot(x,y,data):

    sns.violinplot(data=data,x=x,y=y)  

    plt.show()
cols = df1.columns[:-1]

for i in cols:

    show_voilin_plot('Cover_Type', i , df1)
df1["Wild"] = df1.iloc[:,10:14].idxmax(axis=1)

df1['Wild'].head()
df1['Soil'] = df1.iloc[:,14:54].idxmax(axis=1)

df1['Soil'].head()
df1.head()
plt.rc("figure", figsize=(25, 15))

sns.countplot(x="Wild", hue="Cover_Type", data=df1)

plt.show()
import warnings

warnings.filterwarnings('ignore')

df3 = df1.copy()

df3['Id'] = [i for i in range(df1.shape[0])]

df3['Soil'] = [int(i[-2:]) if i[-2] != 'e' else int(i[-1]) for i in df1['Soil']]

df3['Wild'] = [int(i[-1]) for i in df1['Wild']]

df3['Cover_Type'] = df1['Cover_Type']
df3.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.rc("figure", figsize=(25, 15))

sns.jointplot(x='Soil', y='Wild', data=df3)

plt.show()
sns.distplot(df3['Slope'])
sns.distplot(df3['Elevation'])
sns.distplot(df3['Aspect'])
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Elevation', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Aspect', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Slope', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Horizontal_Distance_To_Hydrology', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Horizontal_Distance_To_Hydrology', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Vertical_Distance_To_Hydrology', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Vertical_Distance_To_Hydrology', hue='Wild', data=df3)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x='Cover_Type', y='Horizontal_Distance_To_Fire_Points', hue='Wild', data=df3)

plt.show()
plt.figure(figsize=(15,10))

sns.stripplot(x="Cover_Type", y="Soil", data=df3, hue='Soil',jitter=True);

plt.show()
g = sns.PairGrid(df3,

                 x_vars=["Cover_Type","Wild"],

                 y_vars=["Hillshade_9am","Hillshade_Noon","Hillshade_3pm"],

                 aspect=.75, size=5.5)

g.map(sns.violinplot, palette="pastel");
sns.factorplot(x="Cover_Type", y="Slope",data=df3, hue="Wild", kind="box", size=10, aspect=.75);
import numpy as np

corr = df3.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)





sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df = pd.concat([df3.iloc[:,:10], df3.iloc[:, -4: ]], axis=1)
df.head()
from scipy import stats

plt.figure(figsize=(8,6))

sns.distplot(df['Horizontal_Distance_To_Hydrology'], fit = stats.norm)

fig = plt.figure(figsize=(8,6))

res = stats.probplot(df['Horizontal_Distance_To_Hydrology'], plot=plt)
plt.figure(figsize=(8,6))

sns.distplot(df['Vertical_Distance_To_Hydrology'], fit = stats.norm)

fig = plt.figure(figsize=(8,6))

res = stats.probplot(df['Vertical_Distance_To_Hydrology'], plot=plt)
plt.figure(figsize=(8,6))

sns.distplot(df['Elevation'], fit = stats.norm)

fig = plt.figure(figsize=(8,6))

res = stats.probplot(df['Elevation'], plot=plt)
plt.rc("figure", figsize=(25, 15))

sns.countplot(x="Soil", hue="Cover_Type", data=df)

plt.show()
corrmat = df.iloc[:,:-1].corr()

f, ax = plt.subplots(figsize = (10,8))

sns.heatmap(corrmat,vmax=0.8,square=True);
y = df['Cover_Type']

del df['Cover_Type']

X = df

df.head()
df = df.iloc[:,:-3]

df.head()
from sklearn.preprocessing import StandardScaler

for i in df.columns:

    df[i] = StandardScaler().fit_transform(df[i].reshape(-1, 1))

df.head()
import warnings

warnings.filterwarnings('ignore')

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
train_df = pd.read_csv('../input/train.csv', sep=',', header=0)

train_df.dropna()



train_df = train_df.drop('Id', axis=1)    

y_data = train_df['Cover_Type'].values

train_df = train_df.drop('Cover_Type', axis=1)





idx = 10 

cols = list(train_df.columns.values)[:idx]

train_df[cols] = StandardScaler().fit_transform(train_df[cols])



X_data = train_df.values            

X_train,X_test,y_train,y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=123)



svm_parameters = [{'kernel': ['rbf'], 'C': [1,10,100,1000]}]                

clf = GridSearchCV(svm.SVC(), svm_parameters, cv=3, verbose=2)

clf.fit(X_train, y_train)    

clf.best_params_

clf.grid_scores_
clf = svm.SVC(C=1000,kernel='rbf')

clf.fit(X_train,y_train)
pred=clf.predict(X_test)

acc=clf.score(X_test,y_test)

print(acc)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=42)

rf.fit(X_train,y_train)

pred=rf.predict(X_test)

acc=rf.score(X_test,y_test)

print(acc)
#predict test data    

test_df = pd.read_csv('../input/test.csv')

test_idx = test_df['Id'].values 

test_df = test_df.drop('Id', axis=1)

test_df[cols] = StandardScaler().fit_transform(test_df[cols])



y_pred = clf.predict(test_df)

y_pred
solution = pd.DataFrame({'Id':test_idx, 'Cover_Type':y_pred}, columns = ['Id','Cover_Type'])

solution.to_csv('submission.csv', index=False)