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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import preprocessing
df =  pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
df.head()
df.info()
df[pd.isnull(df).any(axis=1)]
df.nunique().sort_values()
df['banking_crisis'] = df['banking_crisis'] .apply(lambda x: 1 if x == 'crisis' else 0)
df.head()
def_palette = sns.color_palette()

cat_palette = sns.color_palette("hls", 16)

fig, ax = plt.subplots(figsize = (12,6)) 

fig = sns.lineplot(x='year', y='banking_crisis', data=df, palette=cat_palette, ax=ax).set_title('Strength of crisis in Africa')

con_columns = ['exch_usd','inflation_annual_cpi'] #columns with continuous variables
sns.set(style='whitegrid')

plt.figure(figsize=(10,10))

count = 1



for col in con_columns:

    plt.subplot(2,1,count)

    count += 1

    sns.distplot(df[col])
q1 = df[con_columns].quantile(0.25)

q3 = df[con_columns].quantile(0.75)

iqr = q3 - q1



df[con_columns] = df[con_columns].clip(q1 - 1.5*iqr, q3 + 1.5*iqr, axis=1)
sns.set(style='whitegrid')

plt.figure(figsize=(10,10))

count = 1



for col in df[con_columns]:

    plt.subplot(2,1,count)

    count += 1

    sns.distplot(df[col])
df.groupby(['country', 'banking_crisis']).size().sort_values(ascending=False)
individual_countries = list(df['country'].unique())
sns.set(style='whitegrid')

plt.figure(figsize=(30,30))



count = 1



for country in individual_countries:

    plt.subplot(5,3,count)

    count+=1

    

    sns.lineplot(df[df.country==country]['year'],

                 df[df.country==country]['exch_usd'],

                 label=country)              

            

    plt.plot([(df[np.logical_and(df.country==country,df.banking_crisis==1)]['year'].unique()),

                  (df[np.logical_and(df.country==country,df.banking_crisis==1)]['year']).unique()],

                 [0,np.max(df[df.country==country]['exch_usd'])],

                 color='black',

                 linestyle='dotted',

                 alpha = 0.8)    

    
sns.set(style='whitegrid')

plt.figure(figsize=(30,30))

count = 1



for country in individual_countries:

    plt.subplot(5,3,count)

    count+=1

    

    sns.lineplot(df[df.country==country]['year'],

                 df[df.country==country]['exch_usd'],

                 label=country)

                 

  

    plt.plot([np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

              np.min(df[np.logical_and(df.country==country,df.independence==1)]['year'])],

             [0,

              np.max(df[df.country==country]['exch_usd'])],

             color='black',

             linestyle='dotted',

             alpha=0.8)

    plt.text(np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),np.max(df[df.country==country]['exch_usd'])/2,

             'Independence',

             rotation=-90)



    

   

    plt.tight_layout()
sns.set(style='whitegrid')

plt.figure(figsize=(30,30))

count = 1



for country in individual_countries:

    plt.subplot(5,3,count)

    count+=1

    

    sns.lineplot(df[df.country==country]['year'],

                 df[df.country==country]['inflation_annual_cpi'],

                 label=country)

                 

  

    plt.plot([np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

              np.min(df[np.logical_and(df.country==country,df.independence==1)]['year'])],

             [0,

              np.max(df[df.country==country]['inflation_annual_cpi'])],

             color='black',

             linestyle='dotted',

             alpha=0.8)

    plt.text(np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),np.max(df[df.country==country]['inflation_annual_cpi'])/2,

             'Independence',

             rotation=-90)



    

   

    plt.tight_layout()
sns.set(style='whitegrid')

cols_countplot=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises']

plt.figure(figsize=(20,20))

count = 1

df_bank_crisis = df.loc[df['banking_crisis'] == 1]



for col in cols_countplot:

    plt.subplot(3,2,count)    

    count+= 1

    sns.countplot(y='country', hue = col, data = df_bank_crisis).set_title(col)    

    plt.legend(loc = 0)
sns.set(style='whitegrid')

cols_countplot=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises']

plt.figure(figsize=(20,20))

count = 1

df_no_bank_crisis = df.loc[df['banking_crisis'] == 0]



for col in cols_countplot:

    plt.subplot(3,2,count)    

    count+= 1

    sns.countplot(y='country', hue = col, data = df_no_bank_crisis).set_title(col)    

    plt.legend(loc = 0)
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), annot = True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix
X = df.drop(['banking_crisis','cc3','country','year','case'], axis = 1)

y = df['banking_crisis'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
y_predict_logm = logistic_model.predict(X_test)
column_label = list(X_train.columns)

model_Coeff = pd.DataFrame(logistic_model.coef_, columns = column_label)

model_Coeff['intercept'] = logistic_model.intercept_

print("Coefficient Values Of The Surface Are: ", model_Coeff)
logmodel_score = logistic_model.score(X_test,y_test)

print('Model score:\n', logmodel_score)
print(metrics.confusion_matrix(y_test, y_predict_logm)) #22 = true positive, 291 = true negative, 2 = false positive, 3 = false negative
print(classification_report(y_test,y_predict_logm))
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
y_predict_tree = decision_tree.predict(X_test)
print(metrics.confusion_matrix(y_test, y_predict_tree))
print(classification_report(y_test,y_predict_tree))
from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydot 



features = list( df.drop(['banking_crisis','cc3','country','year','case'], axis = 1))
dot_data = StringIO()  

export_graphviz(decision_tree, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph[0].create_png())  
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
from sklearn.svm import SVC
svc_model = SVC() #with predefined parameters
svc_model.fit(X_train,y_train)
svm_pred = svc_model.predict(X_test)
print(confusion_matrix(y_test,svm_pred))
print(classification_report(y_test,svm_pred))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_ 
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop(['banking_crisis','cc3','country','year','case'], axis = 1))
kmeans.cluster_centers_
df['Cluster'] = df['banking_crisis']
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(df['Cluster'],kmeans.labels_))

print(classification_report(df['Cluster'],kmeans.labels_))