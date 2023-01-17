import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler,SMOTE

from collections import Counter

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/credit-card/application_data.csv')

data1 = data.copy()

data.head()
data.info()
print("Object type values:",np.count_nonzero(data.select_dtypes('object').columns))

print("___________________________________________________________________________________________")

print(data.select_dtypes('object').columns)

print("___________________________________________________________________________________________")
le = LabelEncoder()

data['NAME_CONTRACT_TYPE'] = le.fit_transform(data['NAME_CONTRACT_TYPE'])

data['CODE_GENDER'] = le.fit_transform(data['CODE_GENDER'])

data['FLAG_OWN_CAR'] = le.fit_transform(data['FLAG_OWN_CAR'])

data['FLAG_OWN_REALTY'] = le.fit_transform(data['FLAG_OWN_REALTY'])

data['NAME_TYPE_SUITE'] = le.fit_transform(data['NAME_TYPE_SUITE'].astype(str))

data['NAME_INCOME_TYPE'] = le.fit_transform(data['NAME_INCOME_TYPE'])

data['NAME_EDUCATION_TYPE'] = le.fit_transform(data['NAME_EDUCATION_TYPE'])

data['NAME_FAMILY_STATUS'] = le.fit_transform(data['NAME_FAMILY_STATUS'])

data['NAME_HOUSING_TYPE'] = le.fit_transform(data['NAME_HOUSING_TYPE'])

data['OCCUPATION_TYPE'] = le.fit_transform(data['OCCUPATION_TYPE'].astype(str))

data['WEEKDAY_APPR_PROCESS_START'] = le.fit_transform(data['WEEKDAY_APPR_PROCESS_START'])

data['ORGANIZATION_TYPE'] = le.fit_transform(data['ORGANIZATION_TYPE'])

data['FONDKAPREMONT_MODE'] = le.fit_transform(data['FONDKAPREMONT_MODE'].astype(str))

data['HOUSETYPE_MODE'] = le.fit_transform(data['HOUSETYPE_MODE'].astype(str))

data['WALLSMATERIAL_MODE'] = le.fit_transform(data['WALLSMATERIAL_MODE'].astype(str))

data['EMERGENCYSTATE_MODE'] = le.fit_transform(data['EMERGENCYSTATE_MODE'].astype(str))
def colors(value):

    if value > 50 and value < 100:

        color = 'red'

    elif value > 154000 and value < 250000:

        color = 'red'

    elif value == 1 :

        color = 'blue'

    else:

        color = 'green'

    return 'color: %s' % color



def missing(df):

    total = df.isnull().sum().sort_values(ascending = False)

    total = total[total>0]

    percent = df.isnull().sum().sort_values(ascending = False)/len(df)*100

    percent = percent[percent>0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percentage']).style.applymap(colors)

missing(data1.select_dtypes('object'))
def mode_impute(df,col):

    return df[col].fillna(df[col].mode()[0])

data1['FONDKAPREMONT_MODE'] = mode_impute(data1,'FONDKAPREMONT_MODE')

data1['WALLSMATERIAL_MODE'] = mode_impute(data1,'WALLSMATERIAL_MODE')

data1['HOUSETYPE_MODE'] = mode_impute(data1,'HOUSETYPE_MODE')

data1['EMERGENCYSTATE_MODE'] = mode_impute(data1,'EMERGENCYSTATE_MODE')

data1['OCCUPATION_TYPE'] = mode_impute(data1,'OCCUPATION_TYPE')

data1['NAME_TYPE_SUITE'] = mode_impute(data1,'NAME_TYPE_SUITE')

missing(data1.select_dtypes('object'))
data1.describe(include=['O'])
print("___________________________________________________________________________________________")

print("Int type values:",np.count_nonzero(data1.select_dtypes('int').columns))

print(data.select_dtypes('int').columns)

print("___________________________________________________________________________________________")
missing(data1.select_dtypes('int'))
data1.select_dtypes('int').agg(['count','min', 'max','mad','mean','median','quantile','kurt','skew','var','std'])
plt.figure(figsize=(30,5))

sns.boxplot(data=data1.select_dtypes('int'))

plt.show()
data1.select_dtypes('int').hist(figsize=(25,25), ec='w')

plt.show()
def color_(value):

    if value < 0 :

        color = 'red'

    elif value == 1 :

        color = 'blue'

    else:

        color = 'green'

    return 'color: %s' % color

data1.select_dtypes('int').corr().style.applymap(color_)

data1.select_dtypes('int').cov().style.applymap(color_)
print("___________________________________________________________________________________________")

print("float type values:",np.count_nonzero(data1.select_dtypes('float').columns))

print(data1.select_dtypes('float').columns)

print("___________________________________________________________________________________________")
missing(data1.select_dtypes('float'))
data1 = data1.select_dtypes('float').interpolate(method ='linear', limit_direction ='forward')

missing(data1.select_dtypes('float'))
data1 = data1.dropna(axis = 1)

missing(data1)
plt.figure(figsize=(30,5))

sns.boxplot(data=data1.select_dtypes('float'))

plt.show()
data1.select_dtypes('float').hist(figsize=(25,25), ec='w')

plt.show()
def color_(value):

    if value < 0 :

        color = 'red'

    elif value == 1 :

        color = 'blue'

    else:

        color = 'green'

    return 'color: %s' % color

data1.select_dtypes('float').corr().style.applymap(color_)

data1.select_dtypes('float').cov().style.applymap(color_)
data = data.interpolate(method ='linear', limit_direction ='forward')

data = data.dropna(axis = 1)

missing(data)
corr = data.corrwith(data['TARGET'],method='spearman').reset_index()



corr.columns = ['Index','Correlations']

corr = corr.set_index('Index')

corr = corr.sort_values(by=['Correlations'], ascending = False).head(10)



plt.figure(figsize=(10, 15))

fig = sns.heatmap(corr, annot=True, fmt="g", cmap='Set3', linewidths=0.4, linecolor='green')



plt.title("Correlation of Variables with Class", fontsize=20)

plt.show()
X = data.drop(['TARGET'],axis = 1)

target = data['TARGET']

X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size= 0.3, random_state = 0)
def ml_model(X_train,X_test, Y_train, Y_test):

  MLA = [LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier()]

  MLA_columns = []

  MLA_compare = pd.DataFrame(columns = MLA_columns)

  row_index = 0

  for alg in MLA:

    predicted = alg.fit(X_train, Y_train).predict(X_test)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'Model Name'] = MLA_name

    MLA_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, Y_train), 2)

    MLA_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, Y_test), 2)

    MLA_compare.loc[row_index, 'Precision'] = round(precision_score(Y_test, predicted),2)

    MLA_compare.loc[row_index, 'Recall'] = round(recall_score(Y_test, predicted),2)

    MLA_compare.loc[row_index, 'F1 score'] = round(f1_score(Y_test, predicted),2)

    row_index+=1

  MLA_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    

  return MLA_compare  

ml_model(X_train,X_test, Y_train, Y_test)
from sklearn.feature_selection import SelectKBest,mutual_info_classif

bestfeatures = SelectKBest(score_func=mutual_info_classif, k=10)

fit = bestfeatures.fit(X,target,)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns) 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score'] 

print(featureScores.nlargest(10,'Score'))  
X = data[['FLAG_CONT_MOBILE','FLAG_MOBIL','FLAG_EMP_PHONE','NAME_TYPE_SUITE','NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE','REGION_RATING_CLIENT_W_CITY','REGION_RATING_CLIENT',

         'FLAG_DOCUMENT_3','FLAG_OWN_REALTY']]

X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size= 0.3, random_state = 0)

Feature_selection = ml_model(X_train,X_test, Y_train, Y_test)

Feature_selection
print('before Oversampling:',Counter(Y_train))

oversample = RandomOverSampler(sampling_strategy='minority')

X_train1, Y_train1 = oversample.fit_resample(X_train, Y_train)

print('After Oversampling:',Counter(Y_train1))
print(Counter(target))

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4.5))

fig.subplots_adjust(bottom=0.10, left=0.10, top = 0.900, right=1.00)

fig.suptitle(' Target Class Before and After Over Sampling', fontsize = 20)

sns.set_palette("bright")

sns.countplot(Y_train, ax=ax1)

ax1.margins(0.1)

ax1.set_facecolor("#e1ddbf")

for p in ax1.patches:

        ax1.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

sns.set_palette("bright")

sns.countplot(Y_train1, ax=ax2)

ax2.margins(0.1)

ax2.set_facecolor("#e1ddbf")

for p in ax2.patches:

        ax2.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

sns.set_style('dark')
oversampling = ml_model(X_train1,X_test, Y_train1, Y_test)

oversampling
print('before SMOTE:',Counter(Y_train))

sm = SMOTE(sampling_strategy='minority')

X_train2, Y_train2 = sm.fit_resample(X_train, Y_train)

print('After SMOTE:',Counter(Y_train2))
Smote = ml_model(X_train2,X_test, Y_train2, Y_test)

Smote