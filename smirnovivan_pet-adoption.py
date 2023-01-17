# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.utils.testing import ignore_warnings
#def get_best_score(model):

#    

#    print(model.best_score_)    

#    print(model.best_params_)

#    print(model.best_estimator_)

#    

#    return model.best_score_
os.listdir('../input/petfinder-adoption-prediction')
#Вдруг пригодится

breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

colors = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')

states = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')



df_train = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")

df_test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_train['AdoptionSpeed'].value_counts().sort_index().plot('barh', color='teal');

plt.title('Adoption speed classes counts');
plt.figure(figsize=(10, 5));

sns.countplot(x='Type', data=df_train);

plt.title('Number of cats and dogs in train data');

# 1-Cat 2-Dog
plt.figure(figsize=(10, 5));

sns.countplot(x='Type', data=df_train, hue ='AdoptionSpeed');

plt.title('Number of cats and dogs in train data');

# 1-Cat 2-Dog
plt.figure(figsize=(10, 5));

sns.countplot(x='Type', data=df_test);

plt.title('Number of cats and dogs in test data');
cols = ['Health', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized']
nr_rows = 2

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*5,nr_rows*6))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        

        i = r*nr_cols+c       

        ax = axs[r][c]

        sns.countplot(df_train[cols[i]], hue=df_train["AdoptionSpeed"], ax=ax)

        ax.set_title(cols[i], fontsize=14, fontweight='bold')

        ax.legend(title="AdoptionSpeed", loc='upper right') 

        

plt.tight_layout() 
plt.figure(figsize=(10, 6));

sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=df_train);

plt.title('AdoptionSpeed by Type and age');
df_train['Free'] = df_train['Fee'].apply(lambda x: 1 if x == 0 else 0)

df_test['Free'] = df_test['Fee'].apply(lambda x: 1 if x == 0 else 0)
plt.figure(figsize=(10, 5));

sns.countplot(x='Free', data=df_train, hue ='AdoptionSpeed');

plt.title('Free cats and dogs');
for df in [df_train, df_test]:

    df['Name']=df['Name'].fillna('No Name')
for df in [df_train, df_test]:

    df['No Name Flag']=df['Name'].apply(lambda x: 1 if x =='No Name' else 0)
plt.figure(figsize=(10, 5));

sns.countplot(x='No Name Flag', data=df_train, hue ='AdoptionSpeed');

plt.title('Name Adoption');
pop_name = list(df_train['Name'].value_counts()[df_train['Name'].value_counts()>=30].index)

rare_name = list(df_train['Name'].value_counts()[df_train['Name'].value_counts()==1].index)
pop_name.remove('No Name')
for df in [df_train, df_test]:

    df['Popular Name']=df['Name'].apply(lambda x: 1 if x in pop_name else 0)

    df['Rare Name']=df['Name'].apply(lambda x: 1 if x in rare_name else 0)
plt.figure(figsize=(10, 5));

sns.countplot(x='Rare Name', data=df_train, hue ='AdoptionSpeed');

plt.title('Popular Name Adoption');
df_train_ml = df_train.copy()

df_test_ml = df_test.copy()



pet_id = df_test_ml['PetID']
df_train_ml.drop(['PetID','Name','RescuerID', 'Description'],axis=1,inplace=True)

df_test_ml.drop(['PetID','Name','RescuerID', 'Description'],axis=1,inplace=True)

#X_train=df_train_ml.iloc[:, :-1]

X_train=df_train_ml.drop(['AdoptionSpeed'],axis=1)

y_train=df_train_ml['AdoptionSpeed']



X_test=df_test_ml
from catboost import CatBoostClassifier



cat=CatBoostClassifier()

cat.fit(X_train,y_train)



predictions=cat.predict(X_test)
nr_f = 15

imp = pd.Series(data = cat.feature_importances_,index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(7,5))

plt.title("Feature importance")

ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')
sub_cat = pd.DataFrame()

sub_cat['PetId'] = pet_id

sub_cat['AdoptionSpeed'] = predictions
sub_cat.to_csv('submission.csv',index=False)

#Score = 0.28738
from sklearn.svm import SVC

svc = SVC(gamma = 0.01, C = 100)

svc.fit(X_train,y_train)

predictions=svc.predict(X_test)
sub_svc = pd.DataFrame()

sub_svc['PetId'] = pet_id

sub_svc['AdoptionSpeed'] = predictions
#sub_svc.to_csv('submission.csv',index=False)

#Score = 0.16022
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
sub_knn = pd.DataFrame()

sub_knn['PetId'] = pet_id

sub_knn['AdoptionSpeed'] = predictions
#sub_knn.to_csv('submission.csv',index=False)

#Score = 0.17268