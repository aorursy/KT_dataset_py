import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns



df = pd.read_csv("/kaggle/input/data-mining-assignment-2/train.csv")

df_test = pd.read_csv("/kaggle/input/data-mining-assignment-2/test.csv")

df_old = df.copy()

df_test_old = df_test.copy()
df.head()
df.info(verbose = True,null_counts =True)
column_objects = []

for name in df.columns:

    if df[name].dtype=='object':

        column_objects.append(name)

column_objects
for name in column_objects:

    print(df[name].unique())
def replace_objs(s):

    mappy = {"Silver":0,"Gold":1,"Diamond":2,"Platinum":3,"Yes":1,"No":0,"Low":0,"Medium":1,"High":2,"Male":0,"Female":1}

    return mappy[s]
for name in column_objects:

    df[name] = df[name].apply(replace_objs)
for name in column_objects:

    df_test[name] = df_test[name].apply(replace_objs)
import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams.update({'figure.figsize':(10,10), 'figure.dpi':100})



for colname in df.columns:

    x = df[colname]

    y = df['Class']

    plt.hist([x[y==0],x[y==1],x[y==2],x[y==3]],label = ['Class 0','Class 1','Class 2','Class 3'])

    plt.gca().set(title=colname+' Histogram', ylabel='Frequency')

    plt.legend()

    plt.show()
corr = df.corr()

corr

graph ,axis= plt.subplots()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=axis, annot = False)
print(sorted(corr["Class"]))
drop_cols = ["Class"]

thresh = 0

for name in corr["Class"].axes[0]:

    if abs(corr["Class"][name]) < thresh:

        drop_cols.append(name)

drop_cols
df = df.drop(drop_cols,axis = 1)

taken_cols = []



lastcol = 63

drop_cols = []

for i in range(1,lastcol+1):

    colname = "col"+str(i)

    

    if colname not in drop_cols:

        taken_cols.append(colname)

taken_cols
from sklearn.preprocessing import StandardScaler



df = df[taken_cols]

x = df.loc[:].values

# Separating out the target

y = df_old.loc[:,['Class']].values

y = y.ravel()

# Standardizing the features

x = StandardScaler().fit_transform(x)
print(x[0])

x.shape
# from sklearn.manifold import TSNE

# from sklearn.decomposition import PCA



# # pca = TSNE(n_components=2,n_jobs = 4,verbose = 1,perplexity=10, n_iter=2000)

# pca = PCA(n_components=10)



# principalComponents = pca.fit_transform(x)
# X_labels = principalComponents

# Y_labels = df_old["Class"]

# plt.figure(figsize=(16,10))

# sns.scatterplot(x=X_labels[:,0], y=X_labels[:,1],hue=Y_labels,palette=sns.color_palette("hls", 4),

#     legend="full",

#     alpha=0.3

# )

# x = X_labels

# y = Y_labels
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



score_train_RF = []

score_test_RF = []

models = []

func = range(1,15,1)



for i in func:

    if i%5==0:

        print(i)

    rf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = i),n_estimators=100)

    models.append(rf)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(func,score_train_RF,color='blue',linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)



test_score,=plt.plot(func,score_test_RF,color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=5)



plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
models[12].score(X_test,y_test)
# # rf = RandomForestClassifier(n_estimators=1000, max_depth = 16)

# rf = AdaBoostClassifier(base_estimator = RandomForestClassifier(max_depth = 5,n_estimators = 1000))



# rf.fit(X_train, y_train)

# rf.score(X_test,y_test)
# from sklearn.model_selection import GridSearchCV

# # Create the parameter grid based on the results of random search 

# param_grid = {

#     'bootstrap': [True],

#     'max_depth': [80, 90, 100, 110],

#     'max_features': [2, 3],

#     'min_samples_leaf': [3, 4, 5],

#     'min_samples_split': [8, 10, 12],

#     'n_estimators': [100, 200, 300, 1000]

# }

# # Create a based model

# rf = RandomForestClassifier()

# # Instantiate the grid search model

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

#                           cv = 3, n_jobs = -1, verbose = 2)



# grid_search.fit(X_train, y_train)
# best_grid = grid_search.best_estimator_

# best_grid.score(X_test,y_test)
from sklearn.preprocessing import StandardScaler

df_test = df_test[taken_cols]

x = df_test.loc[:].values

# Separating out the target

# Standardizing the features

x = StandardScaler().fit_transform(x)
y_pred = models[12].predict(x)
cols = pd.DataFrame(y_pred)

cols["ID"] = df_test_old["ID"]

cols["Class"] = cols[0]

cols = cols.drop(0,axis = 1)
cols.head()
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

create_download_link(cols)