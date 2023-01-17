# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.chdir('../input')

df = pd.read_csv('2004-2019.tsv', sep='\t',parse_dates=[1,2])

df.head()
df.info()
df.columns
df = df.drop("Unnamed: 0", axis=1)
df.rename(columns={'DATA INICIAL':'date_first', 

                     'DATA FINAL':'date_last',

                     'REGIÃO':'macro_region',

                     'ESTADO':'state',

                     'PRODUTO':'product',

                     'NÚMERO DE POSTOS PESQUISADOS':'num_gas_station', 

                     'UNIDADE DE MEDIDA':'unit',

                     'PREÇO MÉDIO REVENDA':'mean_market_value',

                     'DESVIO PADRÃO REVENDA':'standard_deviation',

                     'PREÇO MÍNIMO REVENDA':'min_price',

                     'PREÇO MÁXIMO REVENDA':'max_price',

                     'MARGEM MÉDIA REVENDA':'mean_price_margin',

                     'COEF DE VARIAÇÃO REVENDA':'coefficient_variation',

                     'PREÇO MÉDIO DISTRIBUIÇÃO':'mean_dist_price',

                     'DESVIO PADRÃO DISTRIBUIÇÃO':'dist_standard_deviation',

                     'PREÇO MÍNIMO DISTRIBUIÇÃO':'dist_min_price',

                     'PREÇO MÁXIMO DISTRIBUIÇÃO':'dist_max_price',

                     'COEF DE VARIAÇÃO DISTRIBUIÇÃO':'dist_coefficient_variation',

                     'MÊS':'month',

                     'ANO':'year'}

            , inplace=True)

df.dtypes
for col in ['mean_price_margin','mean_dist_price','dist_standard_deviation',

            'dist_min_price','dist_max_price','dist_coefficient_variation']:

    df[col]=pd.to_numeric(df[col],errors='coerce')
df.head()
df['macro_region']=pd.Categorical(df['macro_region'])

df['state']=pd.Categorical(df['state'])

df['product']=pd.Categorical(df['product'])

df['unit']=pd.Categorical(df['unit'])
data=df.copy()

data.head()
data.shape
data.describe().T
data.isnull().values.any()
data.isnull().sum().sort_values(ascending=False)
cat_df=data.select_dtypes(include=["category"])

cat_df.head()
cat_df["macro_region"].unique()
cat_df["state"].unique()
cat_df["product"].unique()
cat_df["unit"].unique()
cat_df["unit"].value_counts()
plt.figure(figsize=(15,4))

sns.barplot(x='product',y=data['mean_market_value'],hue='unit', data=data);
data1=data.loc[(data['unit'] == 'R$/l')]

data2=data.loc[(data['unit'] == 'R$/13Kg')]

data3=data.loc[(data['unit'] == 'R$/m3')]
data1["unit"].value_counts()
data1["macro_region"].value_counts().plot.barh();
plt.figure(figsize=(15,4))

sns.barplot(x='product',y=data1['product'].index, data=data1);
g = sns.catplot(data=data1, x='product', y='mean_market_value')

g.fig.set_figwidth(11)

g.fig.set_figheight(10)
plt.figure(figsize=(15,4))

sns.barplot(x='product',y=data['mean_market_value'],hue='macro_region', data=data);
plt.figure(figsize=(15,4))

sns.barplot(x='product',y=data1['mean_market_value'],hue='macro_region', data=data1);
sns.kdeplot(data.mean_market_value, shade=True);
(sns

.FacetGrid(data,

          hue="product",

          height=5,

          xlim=(0,10))

.map(sns.kdeplot, "mean_market_value", shade=True)

.add_legend()

);
(sns

.FacetGrid(data1,

          hue="product",

          height=5,

          xlim=(0,10))

.map(sns.kdeplot, "mean_market_value", shade=True)

.add_legend()

);
(sns

.FacetGrid(data2,

          hue="product",

          height=5,

          xlim=(25,90))

.map(sns.kdeplot, "mean_market_value", shade=True)

.add_legend()

);
sns.catplot(x="product", y="num_gas_station", hue="macro_region",

           kind="point", height=5, aspect=3,data=data);
data.pivot_table('mean_market_value','state','product')
sns.catplot(x="product", y="mean_market_value", hue="state",

           kind="point", height=10, aspect=2, data=data);
sns.catplot(x="product", y="mean_market_value", hue="state",

           kind="point", height=10, aspect=2, data=data1);
sns.boxplot(x="month",y='mean_market_value',data=data1);
sns.boxplot(x="month",y='mean_market_value',data=data2);
sns.boxplot(x="month",y='mean_market_value',data=data3);
plt.figure(figsize=(15,4))

sns.boxplot(x="year",y='mean_market_value',data=data1);
plt.figure(figsize=(15,4))

sns.boxplot(x="year",y='mean_market_value',data=data3);
plt.figure(figsize=(15,8))

sns.scatterplot(x='mean_market_value', y='standard_deviation',

                hue='product',style='macro_region',data=data1);
num_gas=pd.cut(data1['num_gas_station'],[0,300,600,900,2000,3000,4000,5000])
data1.pivot_table('mean_market_value',['macro_region', num_gas],'product')
plt.figure(figsize=(15,8))

sns.scatterplot(x='mean_market_value', y='num_gas_station',

                hue='macro_region',style='product', data=data1);
#correlation map

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
data['date_diff']=data['date_last']-data['date_first']

data.groupby("date_diff")["mean_market_value"].count()
#Line Plot



data.max_price.plot(kind='line',color='g',label='Max Price',linewidth=1.5,alpha=0.5,grid=True,linestyle='-')

data.min_price.plot(kind='line',color='b',label='Min Price',linewidth=0.75,alpha=0.5,grid=True,linestyle=':')



plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data_a=data1.sort_values(by=['date_first'])
sns.lineplot(x="date_first",y="mean_market_value",data=data_a);
data_a.head()
data_a.drop(['date_first', 'date_last','state','unit','month','year'], axis=1, inplace=True)
data_a.dropna(inplace=True)
#High Correlation between features

corr_matrix = data_a.corr().abs()

high_corr_var=np.where(corr_matrix>=0.8)

high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr_var
corr = data_a.corr()

k = 15 #number of variables for heatmap

cols = corr.nlargest(k, 'mean_market_value')['mean_market_value'].index

cm = np.corrcoef(data_a[cols].values.T)

sns.set(font_scale=1.25)

fig, ax = plt.subplots(figsize=(10,10))       

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},

                 yticklabels=cols.values, xticklabels=cols.values,cmap='RdYlGn')

plt.show()
data_a.isnull().sum().sort_values(ascending=False)
#Normalization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



# get numeric data

num_d = data_a.select_dtypes(exclude=['category'])



# update the cols with their normalized values

data_a[num_d.columns] = sc.fit_transform(num_d)



# convert string variable to One Hot Encoding

ohe_data = pd.get_dummies(data_a["macro_region"])
ohe_data.head()
data_a.head()
result = pd.concat([data_a, ohe_data], axis=1)

result.drop(['macro_region'], axis=1, inplace=True)

result.head()
X=result.drop(['product','min_price','dist_max_price','mean_dist_price','dist_min_price','dist_standard_deviation'], axis=1)

Y=result['product']
Y=pd.DataFrame(Y)



from sklearn.preprocessing import LabelEncoder



# LabelEncoder

le = LabelEncoder()



# apply "le.fit_transform"

Y_encoded = Y.apply(le.fit_transform)

Y_encoded=pd.DataFrame(Y_encoded)

Y_encoded.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.metrics import accuracy_score



#split data

x_train, x_test, y_train, y_test =train_test_split(X,Y_encoded, test_size=0.3, random_state=42)



#random forest classifiers

rf=RandomForestClassifier(random_state=42)

ab=rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)



acc=accuracy_score(y_test,y_pred)

print('Accuracy is: ', acc)



cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True, fmt="d")

from sklearn.feature_selection import SelectKBest, chi2 # for chi-squared feature selection

from sklearn import feature_selection



sf = SelectKBest(score_func=feature_selection.f_regression, k='all')

sf_fit = sf.fit(x_train,y_train)

# print feature scores

for i in range(len(sf_fit.scores_)):

    print(' %s: %f' % (x_train.columns[i], sf_fit.scores_[i]))
datset = pd.DataFrame()

datset['feature'] = x_train.columns[ range(len(sf_fit.scores_))]

datset['scores'] = sf_fit.scores_

datset = datset.sort_values(by='scores', ascending=False)

datset
x_train_2=sf_fit.transform(x_train)

x_test_2=sf_fit.transform(x_test)



#RandomForestClassifier

rf2=RandomForestClassifier()



xy=rf2.fit(x_train_2,y_train)

y_pred_2=rf2.predict(x_test_2)



ac_2=accuracy_score(y_test, y_pred_2)

print('Accuracy is: ', ac_2)



cm_2=confusion_matrix(y_test, y_pred_2)

sns.heatmap(cm_2, annot=True, fmt="d")
#Recursive feature elimination with cross validation and random forest classification.



from sklearn.feature_selection import RFECV



rf_4= RandomForestClassifier()

rfecv = RFECV (estimator=rf_4, step=1, cv=5, scoring='accuracy')  #5- fold cross validation



rfecv=rfecv.fit(x_train, y_train)
#Plot number of features with cross-validation scores



import matplotlib.pyplot as plt

plt.figure()



plt.xlabel("number of features selected")

plt.ylabel("cross validation score of number of selected features")



plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# In random forest classification has a featureImportances.(The higher, the more important the feature) 

#To use feature_importance method, in training data there should not be correlated features.



rf_5=RandomForestClassifier()

rfc_5=rf_5.fit(x_train,y_train)

importances=rfc_5.feature_importances_



std=np.std([tree.feature_importances_ for tree in rfc_5.estimators_], axis=0)

indices=np.argsort(importances)[::-1]



#Print the feature ranking

print("Feature ranking: ")



for f in range(x_train.shape[1]):

    print("%d.feature %d (%f) " % (f+1, indices[f], importances[indices[f]]))

    

#plot the feature importances

plt.figure(1, figsize=(14,13))

plt.title("Feature İmportance")

plt.bar(range(x_train.shape[1]),importances[indices], color="b", yerr=std[indices], align="center")

plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)

plt.xlim([-1, x_train.shape[1]])

plt.show()
#Feature Extraction





from sklearn.decomposition import PCA

pca=PCA()

pca.fit(x_train)



plt.figure(1, figsize=(14,13))

plt.clf()

plt.axes([.2, .2, .7, .7])

plt.plot(pca.explained_variance_ratio_, linewidth=2)

plt.axis('tight')

plt.xlabel('n_components')

plt.ylabel('explained_variance_ratio_')