# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()





import warnings

warnings.filterwarnings('ignore')
df_income_train = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")

df_income_test =  pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")
df_income_train.head()
df_income_train.info()
df_income_test.head()
### List the columns for different datatypes:

print('Integer Type: ')

print(df_income_train.select_dtypes(np.int64).columns)

print('\n')

print('Float Type: ')

print(df_income_train.select_dtypes(np.float64).columns)

print('\n')

print('Object Type: ')

print(df_income_train.select_dtypes(np.object).columns)
df_income_train.select_dtypes('int64').head()
#Find columns with null values

null_counts=df_income_train.select_dtypes('int64').isnull().sum()

null_counts[null_counts > 0]
df_income_train.select_dtypes('float64').head()
#Find columns with null values

null_counts=df_income_train.select_dtypes('float64').isnull().sum()

null_counts[null_counts > 0]
df_income_train.select_dtypes('object').head()
#Find columns with null values

null_counts=df_income_train.select_dtypes('object').isnull().sum()

null_counts[null_counts > 0]
mapping={'yes':1,'no':0}



for df in [df_income_train, df_income_test]:

    df['dependency'] =df['dependency'].replace(mapping).astype(np.float64)

    df['edjefe'] =df['edjefe'].replace(mapping).astype(np.float64)

    df['edjefa'] =df['edjefa'].replace(mapping).astype(np.float64)

    

df_income_train[['dependency','edjefe','edjefa']].describe()
data = df_income_train[df_income_train['v2a1'].isnull()].head()



columns=['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']

data[columns]
# Variables indicating home ownership

own_variables = [x for x in df_income_train if x.startswith('tipo')]





# Plot of the home ownership variables for home missing rent payments

df_income_train.loc[df_income_train['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'green',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 20)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
#Looking at the above data it makes sense that when the house is fully paid, there will be no monthly rent payment.

#Lets add 0 for all the null values.

for df in [df_income_train, df_income_test]:

    df['v2a1'].fillna(value=0, inplace=True)



df_income_train[['v2a1']].isnull().sum()
# Heads of household### NOTE

heads = df_income_train.loc[df_income_train['parentesco1'] == 1].copy()

heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
plt.figure(figsize = (8, 6))

col='v18q1'

df_income_train[col].value_counts().sort_index().plot.bar(color = 'blue',

                                             edgecolor = 'k',

                                             linewidth = 2)

plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')

plt.show();
for df in [df_income_train, df_income_test]:

    df['v18q1'].fillna(value=0, inplace=True)



df_income_train[['v18q1']].isnull().sum()
# Lets look at the data with not null values first.

df_income_train[df_income_train['rez_esc'].notnull()]['age'].describe()
df_income_train.loc[df_income_train['rez_esc'].isnull()]['age'].describe()
df_income_train.loc[(df_income_train['rez_esc'].isnull() & 

                     ((df_income_train['age'] > 7) & (df_income_train['age'] < 17)))]['age'].describe()

#There is one value that has Null for the 'behind in school' column with age between 7 and 17 
df_income_train[(df_income_train['age'] ==10) & df_income_train['rez_esc'].isnull()].head()

df_income_train[(df_income_train['Id'] =='ID_f012e4242')].head()

#there is only one member in household for the member with age 10 and who is 'behind in school'. This explains why the member is 

#behind in school.
#from above we see that  the 'behind in school' column has null values 

# Lets use the above to fix the data

for df in [df_income_train, df_income_test]:

    df['rez_esc'].fillna(value=0, inplace=True)

df_income_train[['rez_esc']].isnull().sum()
data = df_income_train[df_income_train['meaneduc'].isnull()].head()



columns=['edjefe','edjefa','instlevel1','instlevel2']

data[columns][data[columns]['instlevel1']>0].describe()
#from the above, we find that meaneduc is null when no level of education is 0

#Lets fix the data

for df in [df_income_train, df_income_test]:

    df['meaneduc'].fillna(value=0, inplace=True)

df_income_train[['meaneduc']].isnull().sum()
data = df_income_train[df_income_train['SQBmeaned'].isnull()].head()



columns=['edjefe','edjefa','instlevel1','instlevel2']

data[columns][data[columns]['instlevel1']>0].describe()
#from the above, we find that SQBmeaned is null when no level of education is 0

#Lets fix the data

for df in [df_income_train, df_income_test]:

    df['SQBmeaned'].fillna(value=0, inplace=True)

df_income_train[['SQBmeaned']].isnull().sum()
#Lets look at the overall data

null_counts = df_income_train.isnull().sum()

null_counts[null_counts > 0].sort_values(ascending=False)
# Groupby the household and figure out the number of unique values

all_equal = df_income_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Households where targets are not all equal

not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
#Lets check one household

df_income_train[df_income_train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
#Lets use Target value of the parent record (head of the household) and update rest. But before that lets check

# if all families has a head. 



households_head = df_income_train.groupby('idhogar')['parentesco1'].sum()



# Find households without a head

households_no_head = df_income_train.loc[df_income_train['idhogar'].isin(households_head[households_head == 0].index), :]



print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))
# Find households without a head and where Target value are different

households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

print('{} Households with no head have different Target value.'.format(sum(households_no_head_equal == False)))
#Lets fix the data

#Set poverty level of the members and the head of the house within a family.

# Iterate through each household

for household in not_equal.index:

    # Find the correct label (for the head of household)

    true_target = int(df_income_train[(df_income_train['idhogar'] == household) & (df_income_train['parentesco1'] == 1.0)]['Target'])

    

    # Set the correct label for all members in the household

    df_income_train.loc[df_income_train['idhogar'] == household, 'Target'] = true_target

    

    

# Groupby the household and figure out the number of unique values

all_equal = df_income_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Households where targets are not all equal

not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
# 1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households 

target_counts = heads['Target'].value_counts().sort_index()

target_counts
target_counts.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'k',title="Target vs Total_Count")
#Lets remove them

print(df_income_train.shape)

cols=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 

        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']





for df in [df_income_train, df_income_test]:

    df.drop(columns = cols,inplace=True)



print(df_income_train.shape)
id_ = ['Id', 'idhogar', 'Target']



ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone']



ind_ordered = ['rez_esc', 'escolari', 'age']



hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']



hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
#Check for redundant household variables

heads = df_income_train.loc[df_income_train['parentesco1'] == 1, :]

heads = heads[id_ + hh_bool + hh_cont + hh_ordered]

heads.shape
# Create correlation matrix

corr_matrix = heads.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
['coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total']
corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]
sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],

            annot=True, cmap = plt.cm.Accent_r, fmt='.3f');
cols=['tamhog', 'hogar_total', 'r4t3']

for df in [df_income_train, df_income_test]:

    df.drop(columns = cols,inplace=True)



df_income_train.shape
#Check for redundant Individual variables

ind = df_income_train[id_ + ind_bool + ind_ordered]

ind.shape
# Create correlation matrix

corr_matrix = ind.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
# This is simply the opposite of male! We can remove the male flag.

for df in [df_income_train, df_income_test]:

    df.drop(columns = 'male',inplace=True)



df_income_train.shape
#lets check area1 and area2 also

# area1, =1 zona urbana 

# area2, =2 zona rural 

#area2 redundant because we have a column indicating if the house is in a urban zone



for df in [df_income_train, df_income_test]:

    df.drop(columns = 'area2',inplace=True)



df_income_train.shape
#Finally lets delete 'Id', 'idhogar'

cols=['Id','idhogar']

for df in [df_income_train, df_income_test]:

    df.drop(columns = cols,inplace=True)



df_income_train.shape
df_income_train.iloc[:,0:-1]
df_income_train.iloc[:,-1]
x_features=df_income_train.iloc[:,0:-1] # feature without target

y_features=df_income_train.iloc[:,-1] # only target

print(x_features.shape)

print(y_features.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report



x_train,x_test,y_train,y_test=train_test_split(x_features,y_features,test_size=0.2,random_state=1)

rmclassifier = RandomForestClassifier()
rmclassifier.fit(x_train,y_train)
y_predict = rmclassifier.predict(x_test)
print(accuracy_score(y_test,y_predict))

print(confusion_matrix(y_test,y_predict))

print(classification_report(y_test,y_predict))
y_predict_testdata = rmclassifier.predict(df_income_test)
y_predict_testdata
from sklearn.model_selection import KFold,cross_val_score
seed=7

kfold=KFold(n_splits=5,random_state=seed,shuffle=True)



rmclassifier=RandomForestClassifier(random_state=10,n_jobs = -1)

print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))

results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')

print(results.mean()*100)
num_trees= 100



rmclassifier=RandomForestClassifier(n_estimators=100, random_state=10,n_jobs = -1)

print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))

results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')

print(results.mean()*100)
rmclassifier.fit(x_features,y_features)

labels = list(x_features)

feature_importances = pd.DataFrame({'feature': labels, 'importance': rmclassifier.feature_importances_})

feature_importances=feature_importances[feature_importances.importance>0.015]

feature_importances.head()
y_predict_testdata = rmclassifier.predict(df_income_test)

y_predict_testdata
feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)

feature_importances['positive'] = feature_importances['importance'] > 0

feature_importances.set_index('feature',inplace=True)

feature_importances.head()



feature_importances.importance.plot(kind='barh', figsize=(11, 6),color = feature_importances.positive.map({True: 'blue', False: 'red'}))

plt.xlabel('Importance')