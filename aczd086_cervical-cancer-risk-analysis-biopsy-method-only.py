

import numpy as np 
import pandas as pd 


import os
print(os.listdir("../input"))
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score



Cancer = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")
Cancer.info()
Cancer['Biopsy'].value_counts()
Cancerna = Cancer.replace('?', np.nan) 
Cancerna.isnull().sum() 
Cancer = Cancerna
Cancer.drop(Cancer.columns[[26,27,32,33,34]], axis=1, inplace=True)
Cancer.isnull().sum()
Cancer = Cancer.convert_objects(convert_numeric=True) 
Cancer['Number of sexual partners'] = Cancer['Number of sexual partners'].fillna(Cancer['Number of sexual partners'].mean())
Cancer['First sexual intercourse'] = Cancer['First sexual intercourse'].fillna(Cancer['First sexual intercourse'].mean())
Cancer['Num of pregnancies'] = Cancer['Num of pregnancies'].fillna(Cancer['Num of pregnancies'].median())
Cancer['Smokes'] = Cancer['Smokes'].fillna(Cancer['Smokes'].median())
Cancer['Smokes (years)'] = Cancer['Smokes (years)'].fillna(Cancer['Smokes (years)'].mean())
Cancer['Smokes (packs/year)'] = Cancer['Smokes (packs/year)'].fillna(Cancer['Smokes (packs/year)'].mean())
Cancer['Hormonal Contraceptives'] = Cancer['Hormonal Contraceptives'].fillna(Cancer['Hormonal Contraceptives'].median())
Cancer['Hormonal Contraceptives (years)'] = Cancer['Hormonal Contraceptives (years)'].fillna(Cancer['Hormonal Contraceptives (years)'].mean())
Cancer['IUD'] = Cancer['IUD'].fillna(Cancer['IUD'].median()) 
Cancer['IUD (years)'] = Cancer['IUD (years)'].fillna(Cancer['IUD (years)'].mean())
Cancer['STDs'] = Cancer['STDs'].fillna(Cancer['STDs'].median())
Cancer['STDs (number)'] = Cancer['STDs (number)'].fillna(Cancer['STDs (number)'].median())
Cancer['STDs:condylomatosis'] = Cancer['STDs:condylomatosis'].fillna(Cancer['STDs:condylomatosis'].median())
Cancer['STDs:cervical condylomatosis'] = Cancer['STDs:cervical condylomatosis'].fillna(Cancer['STDs:cervical condylomatosis'].median())
Cancer['STDs:vaginal condylomatosis'] = Cancer['STDs:vaginal condylomatosis'].fillna(Cancer['STDs:vaginal condylomatosis'].median())
Cancer['STDs:vulvo-perineal condylomatosis'] = Cancer['STDs:vulvo-perineal condylomatosis'].fillna(Cancer['STDs:vulvo-perineal condylomatosis'].median())
Cancer['STDs:syphilis'] = Cancer['STDs:syphilis'].fillna(Cancer['STDs:syphilis'].median())
Cancer['STDs:pelvic inflammatory disease'] = Cancer['STDs:pelvic inflammatory disease'].fillna(Cancer['STDs:pelvic inflammatory disease'].median())
Cancer['STDs:genital herpes'] = Cancer['STDs:genital herpes'].fillna(Cancer['STDs:genital herpes'].median())
Cancer['STDs:molluscum contagiosum'] = Cancer['STDs:molluscum contagiosum'].fillna(Cancer['STDs:molluscum contagiosum'].median())
Cancer['STDs:AIDS'] = Cancer['STDs:AIDS'].fillna(Cancer['STDs:AIDS'].median())
Cancer['STDs:HIV'] = Cancer['STDs:HIV'].fillna(Cancer['STDs:HIV'].median())
Cancer['STDs:Hepatitis B'] = Cancer['STDs:Hepatitis B'].fillna(Cancer['STDs:Hepatitis B'].median())
Cancer['STDs:HPV'] = Cancer['STDs:HPV'].fillna(Cancer['STDs:HPV'].median())
Cancer.isnull().sum()
Cancer.describe()
correlationMap = Cancer.corr()

plt.figure(figsize=(40,40))

sns.set(font_scale=3)
hm = sns.heatmap(correlationMap,cmap = 'Set1', cbar=True, annot=True,vmin=0,vmax =1,center=True, square=True, fmt='.2f', annot_kws={'size': 25},
             yticklabels = Cancer.columns, xticklabels = Cancer.columns)
plt.show()
correlationMap = Cancer.corr()
k = 16
correlations = correlationMap.nlargest(k, 'Biopsy')['Biopsy'].index



M = Cancer[correlations].corr()

plt.figure(figsize=(10,10))

sns.set(font_scale=1)
H = sns.heatmap(M, cbar=True, cmap='rainbow' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 12},
                 yticklabels = correlations.values, xticklabels = correlations.values)
plt.show()
new_col= Cancer.groupby('Biopsy').mean()
print(new_col.head().T)

cols = ['Age', 'Number of sexual partners',
        'First sexual intercourse', 'Smokes (packs/year)',
         'Hormonal Contraceptives (years)','IUD (years)', 'Smokes (years)']

sns.pairplot(Cancer,
             x_vars = cols,
             y_vars = cols,
             hue = 'Biopsy',)
Cancer['YRSS'] = Cancer['Age'] - Cancer['First sexual intercourse']
Cancer['NSPP'] = Cancer['Number of sexual partners'] / Cancer['YRSS']
Cancer['HPA'] = Cancer['Hormonal Contraceptives (years)'] / Cancer['Age']
Cancer['TPS'] = Cancer['Smokes (packs/year)'] * Cancer['Smokes (years)']
Cancer['NPA'] = Cancer['Num of pregnancies'] / Cancer['Age']
Cancer['NSA'] = Cancer['Number of sexual partners'] / Cancer['Age']
Cancer['NYHC'] = Cancer['Age'] - Cancer['Hormonal Contraceptives (years)']
Cancer['APP'] = Cancer['Num of pregnancies'] / Cancer['Number of sexual partners']
Cancer['NHCP'] = Cancer['Hormonal Contraceptives (years)'] / Cancer['YRSS']
X = Cancer.drop('Biopsy', axis =1)
Y = Cancer["Biopsy"]
x = X.replace([np.inf], 0)
x.isnull().sum()
x['NHCP'] = x['NHCP'].fillna(x['NHCP'].mean())
x.isnull().sum()
%timeit
model = RandomForestRegressor(max_features = 7, n_estimators = 100, n_jobs = -1, oob_score = True, random_state = 42)
model.fit(x,Y)
model.oob_score_

Y_oob = model.oob_prediction_
print("C-Stat: ", roc_auc_score(Y, Y_oob))
Y_oob
categorical_variables = ["Smokes", "Hormonal Contraceptives", "IUD", "STDs", "STDs:condylomatosis",                    
"STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", 
"STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV",                              
 "STDs:Hepatitis B", "STDs:HPV", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"]
for variable in categorical_variables :
    dummies= pd.get_dummies(x[variable], prefix=variable)
    x=pd.concat([x, dummies], axis=1)
    x.drop([variable], axis=1, inplace=True)
    
model = RandomForestRegressor(max_features = 8, n_estimators = 100, n_jobs = -1, oob_score = True, random_state = 42)
model.fit(x,Y)
print("C-stat : ", roc_auc_score(Y, model.oob_prediction_))
feature_importances= pd.Series(model.feature_importances_, index=x.columns)
feature_importances.plot(kind="bar", figsize=(20,20));
results=[]
n_estimator_values=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200]
for trees in n_estimator_values:
    model=RandomForestRegressor(trees, oob_score=True, n_jobs=-1,random_state=42)
    model.fit(x, Y)
    print(trees, "trees")
    roc_score=roc_auc_score(Y, model.oob_prediction_)
    print("C-stat : ", roc_score)
    results.append(roc_score)
    print(" ")
pd.Series(results, n_estimator_values).plot();
results=[]
max_features_values=["auto", "sqrt", "log2", None, 0.2, 0.9]
for max_features in max_features_values:
    model=RandomForestRegressor(n_estimators=140, oob_score=True, n_jobs=-1,random_state=42, 
                                max_features=max_features)
    model.fit(x, Y)
    print(max_features, "option")
    roc_score=roc_auc_score(Y, model.oob_prediction_)
    print("C-stat : ", roc_score)
    results.append(roc_score)
    print(" ")
pd.Series(results, max_features_values).plot(kind="barh", xlim=(0.10, 0.8));
results=[]
min_sample_leaf_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,100,140,200]
for min_sample in min_sample_leaf_values:
    model=RandomForestRegressor(n_estimators=140, oob_score=True, n_jobs=-1,random_state=42, 
                                max_features="log2", min_samples_leaf=min_sample)
    model.fit(x, Y)
    print(min_sample, "min sample")
    roc_score=roc_auc_score(Y, model.oob_prediction_)
    print("C-stat : ", roc_score)
    results.append(roc_score)
    print(" ")
pd.Series(results, min_sample_leaf_values).plot();
model=RandomForestRegressor(n_estimators=140, oob_score=True, n_jobs=-1,random_state=42,
                            max_features="log2", min_samples_leaf=1)
model.fit(x, Y)
roc_score=roc_auc_score(Y, model.oob_prediction_)
print("C-stat : ", roc_score)