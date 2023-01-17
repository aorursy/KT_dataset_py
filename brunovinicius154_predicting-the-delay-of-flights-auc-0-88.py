# Importing Libs

import os

import numpy as np # Linear Algebra

import pandas as pd # Data Manipulation

pd.set_option('MAX_ROWS', None) # Setting pandas to display a N number of columns

from collections import Counter # Data Manipulation

import seaborn as sns # Data Viz

import matplotlib.pyplot as plt # Data Viz

from sklearn import tree # Modelling a tree

from sklearn.impute import SimpleImputer # Perform Imputation

from imblearn.over_sampling import SMOTE # Perform oversampling

from sklearn.preprocessing import OneHotEncoder # Perform OneHotEnconding

from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict # Cross Validation

from sklearn.linear_model import LogisticRegression # Modelling

from sklearn.metrics import classification_report, roc_auc_score,precision_score,recall_score # Evaluating the Model





#warnings

import warnings

warnings.filterwarnings("ignore")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Collecting data

df_2019 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')

df_2020 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')

df_2019.head()
#Creating year indicator.

df_2019['year'] = 2019

df_2020['year'] = 2020



#Checking if the bases have the same columns

print(set(df_2020.columns) == set(df_2019.columns))



#Generating the unique base

dataset = pd.concat([df_2019,df_2020])

print(dataset.shape)

print('\n')

dataset.head()
data = dataset.drop(['OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM', 'ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','Unnamed: 21'], axis=1)

data = data.set_index('OP_CARRIER_FL_NUM')

data.head()
data.head()
#Dataframe summary

pd.DataFrame({'unicos':data.nunique(),

              'missing': data.isna().sum()/data.count(),

              'tipo':data.dtypes})
#Missing values

data.dropna(inplace=True)



#Transformation of data types

colunas = ['DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15','CANCELLED','DIVERTED']

for col in colunas:

  data[col] = data[col].astype('category') 



#Discretization

data['DISTANCE_cat'] = pd.qcut(data['DISTANCE'], q=4)
#Dataframe summary after pre-processing

pd.DataFrame({'unicos':data.nunique(),

              'missing': data.isna().mean()*100,

              'tipo':data.dtypes})
#check data

data.head()
#The concentration of delay and timely arrivals both on departure and on arrival?

f, (ax,ax1) = plt.subplots(1,2, figsize=(12,6))

dep = sns.countplot(data['DEP_DEL15'], ax=ax)

dep.set_title('Depatures')

dep.set_xlabel('Labels')

dep.set_ylabel('Freq')



arr = sns.countplot(data['ARR_DEL15'], ax=ax1)

arr.set_title('Arrivals')

arr.set_xlabel('Labels')

arr.set_ylabel('Freq')
# Percentage of delayed flights that are canceled or diverted?

voos_atrasados = data.loc[data['ARR_DEL15'] == 1,['DIVERTED']]





f, ax= plt.subplots(figsize=(12,6))



#Desvios

desv = sns.countplot(voos_atrasados['DIVERTED'], ax=ax)

desv.set_title('Diverted Flights')

desv.set_xlabel('Labels')

desv.set_ylabel('Freq')
# Delays due to day_of_week and day_of_month?



week = data[['DAY_OF_WEEK','ARR_DEL15']].groupby('DAY_OF_WEEK').sum().sort_values(by='ARR_DEL15',ascending=False)

week['PERCENTUAL'] = week['ARR_DEL15']/(week['ARR_DEL15'].sum())*100

month = data[['DAY_OF_MONTH','ARR_DEL15']].groupby('DAY_OF_MONTH').sum().sort_values(by='ARR_DEL15',ascending=False)

month['PERCENTUAL'] = month['ARR_DEL15']/(month['ARR_DEL15'].sum())*100



print('>> Delayed flights by weekday<<')

print(week)

print('\n')

print('>> Delayed flights by monthday <<')

print(month)
# Concentration of delays due to 'DEP_TIME_BLK'?

time_blk = data[['DEP_TIME_BLK','ARR_DEL15']].groupby('DEP_TIME_BLK').sum().sort_values(by='ARR_DEL15',ascending=False)

time_blk['PERCENTUAL'] = time_blk['ARR_DEL15']/(time_blk['ARR_DEL15'].sum())*100

time_blk
# Which 'Origin' airport stands out in delay?

origin_later = data[['ORIGIN','DEP_DEL15']].groupby('ORIGIN').sum().sort_values(by='DEP_DEL15',ascending=False)

origin_later['PERCENTUAL'] = origin_later['DEP_DEL15']/(origin_later['DEP_DEL15'].sum())*100

origin_later.head()
# Which airport of Destination stands out in delays?

dest_later = data[['DEST','ARR_DEL15']].groupby('DEST').sum().sort_values(by='ARR_DEL15',ascending=False)

dest_later['PERCENTUAL'] = dest_later['ARR_DEL15']/(dest_later['ARR_DEL15'].sum())*100

dest_later.head()
# Helper function to create ARR_TIME_BLOCK

def arr_time(x):



  if x >= 600 and x <= 659:

    return '0600-0659'

  elif x>=1400 and x<=1459:

    return '1400-1459'

  elif x>=1200 and x<=1259:

    return '1200-1259'

  elif x>=1500 and x<=1559:

    return '1500-1559'

  elif x>=1900 and x<=1959:

    return '1900-1959'

  elif x>=900 and x<=959:

    return '0900-0959'

  elif x>=1000 and x<=1059:

    return  '1000-1059'

  elif x>=2000 and x<=2059:

    return '2000-2059'

  elif x>=1300 and x<=1359:

    return '1300-1359'

  elif x>=1100 and x<=1159:

    return '1100-1159'

  elif x>=800 and x<=859:

    return '0800-0859'

  elif x>=2200 and x<=2259:

    return '2200-2259'

  elif x>=1600 and x<=1659:

    return '1600-1659'

  elif x>=1700 and x<=1759:

    return '1700-1759'

  elif x>=2100 and x<=2159:

    return '2100-2159'

  elif x>=700 and x<=759:

    return '0700-0759'

  elif x>=1800 and x<=1859:

    return '1800-1859'

  elif x>=1 and x<=559:

    return '0001-0559'

  elif x>=2300 and x<=2400:

    return '2300-2400'
# We can create ARR_TIME_BLOCK.

data['ARR_TIME'] = data['ARR_TIME'].astype('int')

data['ARR_TIME_BLOCK'] = data['ARR_TIME'].apply(lambda x :arr_time(x))

data.reset_index(inplace=True)

data.head()
# Amount of delays within a DEP_TIME_BLK.

count_time_blk = data[['DEP_TIME_BLK','ARR_DEL15']].groupby('DEP_TIME_BLK').sum().sort_values(by='ARR_DEL15',ascending=False)

count_time_blk.reset_index(inplace=True)

count_time_blk.head()

data1 = data.merge(count_time_blk, left_on='DEP_TIME_BLK', right_on='DEP_TIME_BLK') 

data1.rename({'ARR_DEL15_y':'quant_dep_time_blk','ARR_DEL15_x':'ARR_DEL15' }, inplace=True, axis=1)

data1.head()
# Number of delays DEP_DEL15 per ORIGIN.

count_later_origin = data[['ORIGIN','DEP_DEL15']].groupby('ORIGIN').sum().sort_values(by='DEP_DEL15',ascending=False)

count_later_origin.reset_index(inplace=True)

count_later_origin.head()

data2 = data1.merge(count_later_origin, left_on='ORIGIN', right_on='ORIGIN')

data2.rename({'DEP_DEL15_y':'count_later_origin','DEP_DEL15_x':'DEP_DEL15' }, inplace=True, axis=1)

data2.head() 
# Number of delays ARR_DEL15 per DEST.

count_later_dest = data[['DEST','ARR_DEL15']].groupby('DEST').sum().sort_values(by='ARR_DEL15',ascending=False)

count_later_dest.reset_index(inplace=True)

count_later_dest.head()

data3 = data2.merge(count_later_dest, left_on='DEST', right_on='DEST')

data3.rename({'ARR_DEL15_y':'count_later_dest','ARR_DEL15_x':'ARR_DEL15' },inplace=True, axis=1)

data3.head() 
#Data Preparation

base_final = data3.copy()

base_final.drop(['DEP_TIME','ARR_TIME','OP_CARRIER_FL_NUM'], inplace=True, axis=1)

base_final.set_index('year',inplace=True)
# Separate target, numeric and categorical variables 'ORIGIN', 'DEST'



target_final = base_final[['ARR_DEL15']]



cat_vars_final = base_final.select_dtypes(['object','category'])

cat_vars_final = cat_vars_final.loc[:, ['DAY_OF_MONTH', 'DAY_OF_WEEK','DEP_DEL15','DEP_TIME_BLK','CANCELLED',

                            'DIVERTED','DISTANCE_cat','ARR_TIME_BLOCK']]



#One Hot Encoder



enc = OneHotEncoder().fit(cat_vars_final)



cat_vars_ohe_final = enc.transform(cat_vars_final).toarray()

cat_vars_ohe_final = pd.DataFrame(cat_vars_ohe_final, index= cat_vars_final.index, 

                      columns=enc.get_feature_names(cat_vars_final.columns.tolist()))
#Logisitc Regression Model





#Dividing into training and test data: 2019 - training, 2020 - testing

target_2019_final = target_final[target_final.index == 2019]

target_2020_final = target_final[target_final.index == 2020]



cat_vars_ohe_2019_final = cat_vars_ohe_final[cat_vars_ohe_final.index == 2019]

cat_vars_ohe_2020_final = cat_vars_ohe_final[cat_vars_ohe_final.index == 2020]





#Instantizing Model

lr_model_final = LogisticRegression(C=1.0,n_jobs=-1,verbose=1, random_state=154)



#training

lr_model_final.fit(cat_vars_ohe_2019_final, target_2019_final)
#Validação Cruzada -Treino

cv = StratifiedKFold(n_splits=3, shuffle=True)

result = cross_val_score(lr_model_final,cat_vars_ohe_2019_final,target_2019_final, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f'A média: {np.mean(result)}')

print(f'Limite Inferior: {np.mean(result)-2*np.std(result)}')

print(f'Limite Superior: {np.mean(result)+2*np.std(result)}')
#Test Data



# Predict

pred = lr_model_final.predict(cat_vars_ohe_2020_final)

pred_prob = lr_model_final.predict_proba(cat_vars_ohe_2020_final)



# print classification report

print("Relatório de Classificação:\n", 

       classification_report(target_2020_final, pred, digits=4))



# print the area under the curve

print(f'AUC: {roc_auc_score(target_2020_final,pred_prob[:,1])}')
#ROC Curve

from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(lr_model_final, classes=["nao_atraso", "atraso"])



visualizer.fit(cat_vars_ohe_2019_final, target_2019_final)         

visualizer.score(cat_vars_ohe_2020_final, target_2020_final)                                   

visualizer.show() 
from yellowbrick.classifier import precision_recall_curve

viz = precision_recall_curve(lr_model_final, cat_vars_ohe_2019_final, target_2019_final, cat_vars_ohe_2020_final, target_2020_final)
y_scores_final = lr_model_final.decision_function(cat_vars_ohe_2020_final)

y_pred_recall = (y_scores_final > -3)



print(f'New precision: {precision_score(target_2020_final,y_pred_recall)}')

print(f'New recall: {recall_score(target_2020_final,y_pred_recall)}')