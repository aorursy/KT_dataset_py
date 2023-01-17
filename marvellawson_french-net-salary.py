# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from glob import glob

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.compose import ColumnTransformer 



import os

import time



from mpl_toolkits.basemap import Basemap

from geopy.geocoders import Nominatim

import math



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sklearn as sk

import pylab as pl

import matplotlib.pyplot as plt

import re



from math import sqrt

from scipy.stats import norm

#from mca2 import *

import matplotlib.pyplot as plt

from collections import Counter

#Metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score











#Model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#kaggle

df_info = pd.read_csv("/kaggle/input/french-net-slary/name_geographic_information.csv",sep=",",low_memory=False);

df_salary = pd.read_csv("/kaggle/input/french-net-slary/net_salary_per_town_categories.csv",sep=",",low_memory=False);

df_firms = pd.read_csv("/kaggle/input/french-net-slary/base_etablissement_par_tranche_effectif.csv",sep=",",low_memory=False)

df_population = pd.read_csv("/kaggle/input/french-net-slary/population.csv",sep=",",low_memory=False)

#Local

# df_info = pd.read_csv("name_geographic_information.csv",sep=",",low_memory=False);

# df_salary = pd.read_csv("net_salary_per_town_categories.csv",sep=",",low_memory=False);

# df_firms = pd.read_csv("base_etablissement_par_tranche_effectif.csv",sep=",",low_memory=False)

# df_population = pd.read_csv("population.csv",sep=",",low_memory=False)
print(df_salary.shape)

print(df_firms.shape)

print(df_population.shape)

df_population.LIBGEO= df_population.LIBGEO.str.strip()

df_info.nom_commune= df_info.nom_commune.str.strip()

df_population.LIBGEO= df_population.LIBGEO.str.replace(r"[\"\',]", '')

df_info.nom_commune= df_info.nom_commune.str.replace(r"[\"\',]", '')

df_info.longitude= df_info.longitude.str.replace(r"[\"\',]", '')





new_df_salary = pd.merge(df_salary, df_info, left_on='LIBGEO', right_on='nom_commune') 

new_df_salary.drop(['préfecture', 'numéro_circonscription','numéro_département','éloignement','longitude','latitude','EU_circo','nom_région','codes_postaux','code_insee','code_insee','nom_département','nom_commune','code_région','CODGEO'], axis=1, inplace=True)

region_df = new_df_salary

df_firms = df_firms.merge(df_info, left_on='LIBGEO', right_on='nom_commune')  #  



df_firms.drop(['REG','DEP','préfecture', 'numéro_circonscription','numéro_département','éloignement','longitude','latitude','EU_circo','nom_région','codes_postaux','code_insee','code_insee','nom_département','nom_commune','code_région','CODGEO', 'LIBGEO'], axis=1, inplace=True)

#df_population = df_population.merge(df_info, left_on='LIBGEO', right_on='nom_commune')

#df_population.drop(['préfecture', 'numéro_circonscription','numéro_département','éloignement','longitude','latitude','EU_circo','nom_région','codes_postaux','code_insee','code_insee','nom_département','nom_commune','code_région','CODGEO','NIVGEO', 'LIBGEO'], axis=1, inplace=True)

df_population.drop(['CODGEO','NIVGEO'], axis=1, inplace=True)
#sort les valeurs

region_df.sort_values('LIBGEO', inplace=True) 

#suppression des doublons

region_df.drop_duplicates(keep=False,inplace=True) 

  
print(region_df.isnull().any().sum())

print(df_population.isnull().any().sum())

print(df_firms.isnull().any().sum())
sal_region_df = region_df[['chef.lieu_région', 'sal_moyen', 'sal_moy_cadres_sup', 'sal_moy_prof_inter','sal_hor_moy_employes',

            'sal_hor_moy_ouvriers','sal_hor_moyen_f','sal_hor_moy_f_cadres_sup','sal_hor_moy_f_prof_inter','sal_hor_moy_f_employes',

            'sal_net_hor_moy_f_ouvriers','sal_hor_moy_h','sal_hor_moy_h_cadres_sup','sal_hor_moy_h_prof_inter',

            'sal_hor_moy_h_employes','sal_hor_moy_h_ouvriers','sal_hor_moy_18_a_25_ans','sal_hor_moy_26_a_50_ans','sal_hor_moy_plus_50_ans',

            'sal_hor_moy_f_18_a_25_ans','sal_hor_moy_f_26_a_50_ans','sal_hor_moy_f_plus_50_ans','sal_hor_moy_h_18_a_25_ans',

            'sal_hor_moy_h_26_a_50_ans','sal_hor_moy_h_plus_50_ans']]



nbe_entreprise=df_firms[['chef.lieu_région','nbe_total_entreprise','entreprise_entre_1_a_5_employe','entreprise_entre_6_a_9_employe',

               'entreprise_entre_10_a_19_employe','entreprise_entre_20_a_49_employe','entreprise_entre_50_a_99_employe',

               'entreprise_entre_100_a_199_employe','entreprise_a_plus_500_employe']]



h_f_region_df = region_df[['chef.lieu_région','sal_hor_moy_h','sal_hor_moyen_f','sal_hor_moy_h_cadres_sup','sal_hor_moy_f_cadres_sup',

            'sal_hor_moy_h_prof_inter','sal_hor_moy_f_prof_inter','sal_hor_moy_h_employes','sal_hor_moy_f_employes',

            'sal_hor_moy_h_ouvriers','sal_net_hor_moy_f_ouvriers','sal_hor_moy_h_18_a_25_ans','sal_hor_moy_f_18_a_25_ans',

            'sal_hor_moy_h_26_a_50_ans','sal_hor_moy_f_26_a_50_ans','sal_hor_moy_h_plus_50_ans','sal_hor_moy_f_plus_50_ans']]
print(df_population.shape)

print(sal_region_df.shape)

print(nbe_entreprise.shape)
print(sal_region_df.dtypes)

print(nbe_entreprise.dtypes)

print(df_population.dtypes)

df_population_gr=df_population.groupby('MOCO').sum().sort_values(by=[('NB')], ascending=False)

df_population_gr.drop([11,12] , inplace=True)



df_population_gr = df_population_gr[df_population_gr['AGEQ80_17']>24]

df_population_gr['NB'].plot(kind='pie')
population_single = df_population[df_population['MOCO']== 32] 

population_single = population_single[population_single['AGEQ80_17']>24]

print(population_single.groupby('SEXE').sum()['NB'])

population_single.groupby('SEXE').sum()['NB'].plot(kind='bar')
population_single_parent = df_population[df_population['MOCO']== 23] 

population_parent=population_single_parent[population_single_parent['AGEQ80_17']>20]

population_parent.groupby('SEXE').sum()['NB'].plot(kind='pie')
df_population.describe()
nbe_entreprise_region=nbe_entreprise.groupby('chef.lieu_région').sum().sort_values(by=[('chef.lieu_région')], ascending=False) #['nbe_total_entreprise'].value_counts().plot(kind='pie')

nbe_entreprise_region.head()
ax = pd.DataFrame(nbe_entreprise_region)

ax.plot.pie(y='nbe_total_entreprise',figsize=(6, 6),legend=False)
paris = nbe_entreprise[nbe_entreprise['chef.lieu_région'] == 'Paris']

paris.drop(['nbe_total_entreprise'], axis=1, inplace=True)
paris.hist(figsize=(7,7))

plt.show()
bx = nbe_entreprise_region

bx.drop(['nbe_total_entreprise'], axis=1, inplace=True)

bx.mean().plot(kind='pie')
# g = sns.PairGrid(nbe_entreprise, vars=[ 'entreprise_entre_6_a_9_employe', 'entreprise_entre_10_a_19_employe', 'entreprise_entre_20_a_49_employe'],

#                  hue='nbe_total_entreprise', palette='RdBu_r')

# g.map(plt.scatter, alpha=0.8)

# g.add_legend();
nbe_entreprise_region.describe()
sal_region_df.describe()
# matplotlib histogram

plt.hist(sal_region_df['sal_moyen'], color = 'blue', edgecolor = 'black',

         bins = int(180/10))



# seaborn histogram

sns.distplot(sal_region_df['sal_moyen'], hist=True, kde=False, 

             bins=int(180/10), color = 'blue',

             hist_kws={'edgecolor':'black'})

# Add labels

plt.title('Histogram of Salaire horaire')

plt.xlabel('Salaire horaire (net)')

plt.ylabel('Villes')
#plt.hist([m_salaire_mean, f_salaire_mean],bins = int(180/25), ec="k")

sns.barplot( data=h_f_region_df, edgecolor = 'w')

plt.xticks(rotation=-90)

plt.title('Salaire comparaison entre les hommes et femmes')
sal_region_df['chef.lieu_région'].value_counts(normalize=True).plot(kind='pie')
c1 = list(sal_region_df[sal_region_df['chef.lieu_région'] == 'Lyon']['sal_moyen'])

c2 = list(sal_region_df[sal_region_df['chef.lieu_région'] == 'Paris']['sal_moyen'])

c3 = list(sal_region_df[sal_region_df['chef.lieu_région'] == 'Rennes']['sal_moyen'])

c4 = list(sal_region_df[sal_region_df['chef.lieu_région'] == 'Nantes']['sal_moyen'])

c5 = list(sal_region_df[sal_region_df['chef.lieu_région'] == 'Lille']['sal_moyen'])



# Assign colors for each airline and the names

colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']

names = ['Lyon', 'Paris', 'Rennes',

         'Nantes', 'Lille']

         # Make the histogram using a list of lists

# Normalize the city and assign colors and names

plt.hist([c1, c2, c3, c4, c5], bins = int(180/15), 

         color = colors, label=names)



# Plot formatting

plt.legend()

plt.xlabel('Salaire horaire (net)')

plt.ylabel('nombre de villes')

plt.title('Top. 5 grandes regions')
high_salary_reg= df_salary[df_salary['sal_moyen'] > 30]['LIBGEO']

low_salary_reg= df_salary[df_salary['sal_moyen'] < 11]['LIBGEO']
new_pop_high_sal=df_info.assign(inSalary=df_info.nom_commune.isin(high_salary_reg).astype(int))

print(new_pop_high_sal.shape)

new_pop_high_sal.drop( new_pop_high_sal[ new_pop_high_sal['inSalary'] == 0 ].index , inplace=True)

print(new_pop_high_sal.shape)



lon = new_pop_high_sal['longitude'].values.tolist()

lat = new_pop_high_sal['latitude'].values.tolist()




import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from geopy.geocoders import Nominatim

import math



fig = plt.figure(figsize=(10, 10))

m = Basemap(projection='lcc', resolution=None,

            width=1.2E6, height=1.5E6, 

            lat_0=43.17, lon_0=5.22)

m.etopo(scale=0.5, alpha=0.5)

m.shadedrelief()



m.scatter(lon, lat, latlon=True,    

          cmap='Reds', alpha=0.5)

# Map (long, lat) to (x, y) for plotting





new_pop_low_sal=df_info.assign(inSalary=df_info.nom_commune.isin(low_salary_reg).astype(int))

print(new_pop_low_sal.shape)

new_pop_low_sal.drop( new_pop_low_sal[ new_pop_low_sal['inSalary'] == 0 ].index , inplace=True)

print(new_pop_low_sal.shape)



lon = new_pop_low_sal['longitude'].values.tolist()

lat = new_pop_low_sal['latitude'].values.tolist()
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from geopy.geocoders import Nominatim

import math



fig = plt.figure(figsize=(10, 10))

m = Basemap(projection='lcc', resolution=None,

            width=1.2E6, height=1.7E6, 

            lat_0=43.17, lon_0=5.22)

m.etopo(scale=0.5, alpha=0.5)

m.shadedrelief()

m.scatter(lon, lat, latlon=True,    

          cmap='Reds', alpha=0.5)

# Map (long, lat) to (x, y) for plotting
sal_region_df.hist(figsize=(30,20))

plt.show()
Q1 = sal_region_df.quantile(0.25)

Q3 = sal_region_df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
sal_region_df.boxplot()

plt.boxplot(sal_region_df['sal_moyen'])

plt.show()
#print(sal_region_df < (Q1 - 1.5 * IQR)) |(sal_region_df > (Q3 + 1.5 * IQR))
temp_sal= sal_region_df

for column in temp_sal.iloc[:,1:25]:

    Q1 =temp_sal[column].quantile(0.10)

    Q3 =temp_sal[column].quantile(0.90)

    temp_sal[column] = np.where(temp_sal[column] <Q1, Q1,temp_sal[column])

    temp_sal[column] = np.where(temp_sal[column] >Q3, Q3,temp_sal[column])



    
temp_sal.describe()
temp_sal.boxplot()

corr = temp_sal.iloc[:, 1:25].corr()

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corr, ax = ax, cmap ="RdYlGn", linewidths = 0.1)
temp_sal['class'] = np.where(temp_sal['sal_moyen'] > 15,1,0)

print(temp_sal.shape)

temp_sal.head()
temp_sal['class'].value_counts()
temp_sal= pd.get_dummies(temp_sal, drop_first=True)

print(temp_sal.shape)

temp_sal.head()
target = temp_sal['class'].copy()

y=temp_sal.iloc[:,0]

X=pd.DataFrame(temp_sal.iloc[:,10]) 


X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.2,

                                                    random_state=40)  

print("Nombres villes X_train dataset: ", X_train.shape)

print("Nombres villes y_train dataset: ", y_train.shape)

print("Nombres villes X_test dataset: ", X_test.shape)

print("Nombres villes y_test dataset: ", y_test.shape)
scaler = StandardScaler()

X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
regressor = LinearRegression()

regressor.fit(X_train, y_train)

print("Accuracy: %.2f%%" % (regressor.score(X_test, y_test)*100.0)) 
plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')
y_pred = regressor.predict(X_test)

X_pred_train= regressor.predict(X_train)

print("Mean absolute error: %.2f" % (np.sqrt(mean_squared_error(y_train,X_pred_train))*100.0))

print("Variance score: %.2f%%" % (r2_score(y_train, X_pred_train)*100.0))



X_pred_test= regressor.predict(X_test)

print("Mean absolute error: %.2f" % (np.sqrt(mean_squared_error(y_test,X_pred_test))*100.0)) 

print("Variance score: %.2f%%" % (r2_score(y_test, X_pred_test)*100.0)) 
temp_sal.drop(['sal_moyen','class'], axis=1, inplace=True)
X = temp_sal.iloc[:,0:49]
X.shape
X.dtypes


# On place les variables d'entrée dans un dataframe

X=pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.2,

                                                    random_state=40)  

print("Nombres villes X_train d: ", X_train.shape)

print("Nombres villes y_train dataset: ", y_train.shape)

print("Nombres villes X_test dataset: ", X_test.shape)

print("Nombres villes y_test dataset: ", y_test.shape)
#normalisation

scaler = StandardScaler()

X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
#reduction des dimensions

pca = PCA(n_components=2).fit(X_train)

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)
# Création du modèle

lr = LinearRegression()

lr.fit(X_train, y_train)

y_predictions = lr.predict(X_test)

lr.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

regressor = LinearRegression()

scores = cross_val_score(regressor, X, y, cv=5)

print(scores.mean())

#print(scores)
pred_train_lr= lr.predict(X_train)

print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))

print(r2_score(y_train, pred_train_lr))



pred_test_lr= lr.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 

print(r2_score(y_test, pred_test_lr))
X_train_rg, X_test_rg, y_train_rg, y_test_rg = train_test_split(X,

                                                    target,

                                                    test_size=0.3,

                                                    random_state=40)  

#normalisation

scaler = StandardScaler()

X_train_rg= scaler.fit_transform(X_train_rg)

X_test_rg = scaler.transform(X_test_rg)

#reduction des dimensios

pca = PCA(n_components=2).fit(X_train_rg)

X_train_rg = pca.transform(X_train_rg)

X_test_rg = pca.transform(X_test_rg)

# Création du modèle

dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

dtree.fit(X_train_rg, y_train_rg)

result = dtree.score(X_test_rg, y_test_rg)

print("Accuracy: %.2f%%" % (result*100.0))


pred_train_tree= dtree.predict(X_train_rg)

print(np.sqrt(mean_squared_error(y_train_rg,pred_train_tree)))

print("Accuracy: %.2f%%" % (r2_score(y_train_rg, pred_train_tree)*100.0))





pred_test_tree= dtree.predict(X_test_rg)

print(np.sqrt(mean_squared_error(y_test_rg,pred_test_tree))) 

print(r2_score(y_test_rg, pred_test_tree))
X.shape


X_train, Xtest, Y_train, Ytest = model_selection.train_test_split(X, target, test_size=0.2, random_state=70)

#normalisation

scaler = StandardScaler()

X_train= scaler.fit_transform(X_train)

Xtest = scaler.transform(Xtest)

#reduction des dimensios

pca = PCA(n_components=2).fit(X_train)

X_train = pca.transform(X_train)

Xtest = pca.transform(Xtest)

# Création du modèle

model = LogisticRegression()

model.fit(X_train, Y_train)

result = model.score(Xtest, Ytest)

print("Accuracy: %.2f%%" % (result*100.0))
kfold = model_selection.KFold(n_splits=10, random_state=100)

model_kfold = LogisticRegression()

results_kfold = model_selection.cross_val_score(model_kfold, X, target, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
X_train_tr, X_test_tr, y_train_tr, y_test_tr = train_test_split(X,

                                                    target,

                                                    test_size=0.3,

                                                    random_state=40)  

#normalisation

scaler = StandardScaler()

X_train_tr= scaler.fit_transform(X_train_tr)

X_test_tr = scaler.transform(X_test_tr)

#reduction des dimensios

pca = PCA(n_components=2).fit(X_train_tr)

X_train_tr = pca.transform(X_train_tr)

X_test_tr = pca.transform(X_test_tr)

# Création du modèle

treeClassifier = DecisionTreeClassifier(max_depth=10)

treeClassifier = treeClassifier.fit(X_train_tr, y_train_tr)

result = treeClassifier.score(X_test_tr, y_test_tr)

print("Accuracy: %.2f%%" % (result*100.0))
from sklearn.model_selection import cross_val_score

treeClassifier = DecisionTreeClassifier(max_depth=6)

cross_val_score(treeClassifier, X, target, cv=10).mean()
complexityRange=range(1,11)

erreurAppr=[]

erreurTest=[]

from sklearn.model_selection import train_test_split

from numpy import mean



for complexity in complexityRange:

    erreurApprLocale=[]

    erreurTestLocale=[]

    for j in range(10):

        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)

        algo = DecisionTreeClassifier(max_depth=complexity)

        monModele = algo.fit(X_train, y_train)

        erreurApprLocale.append(1-monModele.score(X_train, y_train))

        erreurTestLocale.append(1-monModele.score(X_test, y_test))

    erreurAppr.append(mean(erreurApprLocale))

    erreurTest.append(mean(erreurTestLocale))

plt.plot(complexityRange,erreurAppr)

plt.plot(complexityRange,erreurTest)


X_train,X_test,y_train,y_test = train_test_split(X, target,test_size=0.2)

#normalisation

scaler = StandardScaler()

X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#reduction des dimensios

pca = PCA(n_components=2).fit(X_train)

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)
knn=KNeighborsClassifier(n_neighbors=3,algorithm="kd_tree",n_jobs=-1)

# Création du modèle

knn.fit(X_train,y_train)

print("Accuracy: %.2f%%" % (knn.score(X_train, y_train)*100.0))

print("Accuracy: %.2f%%" % (knn.score(X_test, y_test)*100.0))
# predicting labels for testing set

knn_predicted_test =knn.predict(X_test)



#scoring knn

knn_accuracy_score  = accuracy_score(y_test,knn_predicted_test)

knn_precison_score  = precision_score(y_test,knn_predicted_test)

knn_recall_score    = recall_score(y_test,knn_predicted_test)

knn_f1_score        = f1_score(y_test,knn_predicted_test)

knn_MCC             =    matthews_corrcoef(y_test,knn_predicted_test)
import seaborn as sns

LABELS = ['Normal Salaire ', 'Salaire Elevé']

conf_matrix = confusion_matrix(y_test, knn_predicted_test)

plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('Class vrai')

plt.xlabel('Class prédict')

plt.show()
lr_probs = model.predict_proba(Xtest)

knn_probs = model.predict_proba(X_test)

rg_probs = model.predict_proba(X_test_rg)

tr_probs = model.predict_proba(X_test_tr)



# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

knn_probs = knn_probs[:, 1]

rg_probs = rg_probs[:, 1]

tr_probs = tr_probs[:, 1]

# calculate scores



knn_auc = roc_auc_score(y_test, knn_probs)

lr_auc = roc_auc_score(Ytest, lr_probs)

rg_auc = roc_auc_score(y_test_rg, rg_probs)

tr_auc = roc_auc_score(y_test_tr, tr_probs)

# summarize scores

print('Knn: ROC AUC=%.3f' % (knn_auc))

print('Lr: ROC AUC=%.3f' % (lr_auc))

print('Rg: ROC AUC=%.3f' % (rg_auc))

print('Tr: ROC AUC=%.3f' % (tr_auc))

# calculate roc curves

knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

lr_fpr, lr_tpr, _ = roc_curve(Ytest, lr_probs)

rg_fpr, rg_tpr, _ = roc_curve(y_test_rg, rg_probs)

tr_fpr, tr_tpr, _ = roc_curve(y_test_tr, tr_probs)

# plot the roc curve for the model

plt.plot(knn_fpr, knn_tpr, linestyle='-.', label='Knn')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Lr')

plt.plot(rg_fpr, rg_tpr, linestyle='--', label='Rg')

plt.plot(tr_fpr, tr_tpr, linestyle='-', label='Tr')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()