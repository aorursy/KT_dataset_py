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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





fighter_details = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")

df = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv", sep=';')
df.head() # The Match Data
fighter_details.head() #The fighter data 
df.info()
df.columns
columns = ['R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.',

       'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY','B_BODY', 'R_LEG', 'B_LEG', 

        'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH','B_CLINCH', 'R_GROUND', 'B_GROUND']
attemp = '_att'

landed = '_landed'



for column in columns:

    df[column+attemp] = df[column].apply(lambda X: int(X.split('of')[1]))

    df[column+landed] = df[column].apply(lambda X: int(X.split('of')[0]))

    

df.drop(columns, axis=1, inplace=True)
df.head()
pct_columns = ['R_SIG_STR_pct','B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']



for column in pct_columns:

    df[column] = df[column].apply(lambda X: float(X.replace('%', ''))/100)
def Division(X):

    for Division in weight_classes:

        if Division in X:

            return Division

    if X == 'Catch Weight Bout' or 'Catchweight Bout':

        return 'Catch Weight'

    else:

        return 'Open Weight'
weight_classes = ['Women\'s Strawweight', 'Women\'s Bantamweight', 

                  'Women\'s Featherweight', 'Women\'s Flyweight', 'Lightweight', 

                  'Welterweight', 'Middleweight','Light Heavyweight', 

                  'Heavyweight', 'Featherweight','Bantamweight', 'Flyweight', 'Open Weight']



df['weight_class'] = df['Fight_type'].apply(Division)
df['weight_class'].value_counts()
def get_rounds(X):

    if X == 'No Time Limit':

        return 1

    else:

        return len(X.split('(')[1].replace(')', '').split('-'))



df['no_of_rounds'] = df['Format'].apply(get_rounds)
df['Winner'].isnull().sum()
df['Winner'].fillna('Draw', inplace=True) #fill the null value with draw
def get_renamed_winner(row):

    if row['R_fighter'] == row['Winner']:

        return 'Red'

    elif row['B_fighter'] == row['Winner']:

        return 'Blue'

    elif row['Winner'] == 'Draw':

        return 'Draw'



df['Winner'] = df[['R_fighter', 'B_fighter', 'Winner']].apply(get_renamed_winner, axis=1)
df['Winner'].value_counts()
def convert_to_cms(X):

    if X is np.NaN:

        return X

    elif len(X.split("'")) == 2:

        feet = float(X.split("'")[0])

        inches = int(X.split("'")[1].replace(' ', '').replace('"',''))

        return (feet * 30.48) + (inches * 2.54)

    else:

        return float(X.replace('"','')) * 2.54
fighter_details['Height'] = fighter_details['Height'].apply(convert_to_cms)

fighter_details['Reach'] = fighter_details['Reach'].apply(convert_to_cms)
fighter_details['Weight'] = fighter_details['Weight'].apply(lambda X: float(X.replace(' lbs.', '')) if X is not np.NaN else X)
fighter_details.head()
new = df.merge(fighter_details, left_on='R_fighter', right_on='fighter_name', how='left')
new = new.drop('fighter_name', axis=1)
new.rename(columns={'Height':'R_Height',

                          'Weight':'R_Weight',

                          'Reach':'R_Reach',

                          'Stance':'R_Stance',

                          'DOB':'R_DOB'}, 

                 inplace=True)
new = new.merge(fighter_details, left_on='B_fighter', right_on='fighter_name', how='left')
new = new.drop('fighter_name', axis=1)
new.rename(columns={'Height':'B_Height',

                          'Weight':'B_Weight',

                          'Reach':'B_Reach',

                          'Stance':'B_Stance',

                          'DOB':'B_DOB'}, 

                 inplace=True)
new.head()
new['R_DOB'] = pd.to_datetime(new['R_DOB'])

new['B_DOB'] = pd.to_datetime(new['B_DOB'])

new['date'] = pd.to_datetime(new['date'])
new['R_year'] = new['R_DOB'].apply(lambda x: x.year)

new['B_year'] = new['B_DOB'].apply(lambda x: x.year)

new['date_year'] = new['date'].apply(lambda x: x.year)
def get_age(row):

    B_age = (row['date_year'] - row['B_year'])

    R_age = (row['date_year'] - row['R_year'])

    if np.isnan(B_age)!=True:

        B_age = B_age

    if np.isnan(R_age)!=True:

        R_age = R_age

    return pd.Series([B_age, R_age], index=['B_age', 'R_age'])
new[['B_age', 'R_age']]= new[['date_year', 'R_year', 'B_year']].apply(get_age, axis=1)
new.drop(['R_DOB', 'B_DOB','date_year','R_year','B_year'], axis=1, inplace=True)
new['country'] = new['location'].apply(lambda x : x.split(',')[-1])
new['date_year'] = new['date'].apply(lambda x: x.year)
values = new['date_year'].sort_values(ascending=False).value_counts().sort_index()

labels = values.index



clrs = ['navy' if (y < max(values)) else 'black' for y in values ]



plt.figure(figsize=(15,8))

bar = sns.barplot(x=labels, y=values, palette=clrs)





ax = plt.gca()

y_max = values.max() 

ax.set_ylim(1)

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), 

        fontsize=10, color='black', ha='center', va='bottom')

    

plt.xlabel('Tahun')

plt.ylabel('Jumlah Pertandingan')

plt.title('UFC Event Per Year')

plt.show()
plt.figure(figsize=(10,5))

bar = sns.countplot(new['country'])

plt.xticks(rotation=90)

ax = plt.gca()

y_max = new['country'].value_counts().max() 

ax.set_ylim(1)

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), 

        fontsize=10, color='black', ha='center', va='bottom')



plt.title('Event by Country')    

plt.show()
women = new.weight_class.str.contains('Women')
women1 = len(new[women])

men = (len(new['weight_class'])) - len(new[women])
labels = 'Men Fight', 'Women Fight'

sizes = [men,women1]

explode = (0, 0.1,)  



fig1, ax1 = plt.subplots(figsize=(10,8))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90 )

ax1.axis('equal') 



plt.show()
plt.figure(figsize=(15,8))

new['Winner'].value_counts()[:10].plot.pie(explode=[0.05,0.05,0.05],autopct='%1.1f%%',shadow=True)

plt.show()
new['R_age'] = new['R_age'].fillna(new['R_age'].median())
new['B_age'] = new['B_age'].fillna(new['B_age'].median())
f,ax=plt.subplots(1,2,figsize=(10,8))

sns.distplot(new['R_age'], ax=ax[0])



ax[0].set_title('R_age')

ax[0].set_ylabel('')

hist = sns.distplot(new['B_age'],ax=ax[1])



ax[1].set_title('B_age')

plt.show()
f,ax=plt.subplots(1,2,figsize=(10,8))

new[new['Winner']=='Red']['R_age'].value_counts().plot.bar(ax=ax[0])



ax[0].set_title('R_age')

ax[0].set_ylabel('')

bar = new[new['Winner']=='Blue']['B_age'].value_counts().plot.bar(ax=ax[1])



ax[1].set_title('B_age')

plt.show()
sns.lmplot(x='R_Height', y='R_Reach', data=new)

plt.show()
new['R_Height'] = new['R_Height'].fillna(new['R_Height'].mean())
new['B_Height'] = new['B_Height'].fillna(new['R_age'].mean())
fig, ax = plt.subplots(figsize=(14, 6))

sns.kdeplot(new.R_Height, shade=True, color='indianred', label='Red')

sns.kdeplot(new.B_Height, shade=True, label='Blue')

plt.xlabel('Height')

plt.title('Height Different')



plt.show()
plt.figure(figsize=(15,8))

sns.countplot(y=new['weight_class'])



sns.set()

sns.set(style="white")

plt.show()
values = new['win_by'].value_counts()

labels = values.index



plt.figure(figsize=(15,8))



sns.barplot(x=values,y=labels, palette='RdBu')



plt.title('UFC Fight Win By')

plt.show()
bar = new.groupby(['weight_class', 'win_by']).size().reset_index().pivot(columns='win_by', index='weight_class', values=0)

bar.plot(kind='barh', stacked=True, figsize=(15,8))

plt.legend(bbox_to_anchor=(1.23, 0.99), loc=1, borderaxespad=0.)

plt.title('UFC Fight Outcome by Division')

plt.xlabel('Jumlah')

plt.ylabel('Divisi')

plt.show()
bar = new.groupby(['date_year', 'win_by']).size().reset_index().pivot(columns='win_by', index='date_year', values=0)

bar.plot(kind='barh', stacked=True, figsize=(15,8))

plt.legend(bbox_to_anchor=(1.23, 0.99), loc=1, borderaxespad=0.)

plt.title('UFC Fight Outcome over the Years')

plt.xlabel('Jumlah')

plt.ylabel('Tahun')

plt.show()
Attempt = pd.concat([new['R_TOTAL_STR._att'], new['B_TOTAL_STR._att']], ignore_index=True)

Landed = pd.concat([new['R_TOTAL_STR._landed'], new['B_TOTAL_STR._landed']], ignore_index=True)
sns.jointplot(x=Attempt , y=Landed)

plt.show()
r_landed = new['R_TOTAL_STR._landed']

r_index = r_landed.index
b_landed = new['B_TOTAL_STR._landed']

b_index = b_landed.index
new['Winner'].head(9)
sns.lineplot(x=r_index[0:9], y=r_landed[0:9], color='r')

sns.lineplot(x=b_index[0:9], y=b_landed[0:9])

plt.show()
Fighter = pd.concat([new['R_fighter'], new['B_fighter']], ignore_index=True)
plt.figure(figsize=(10,8))

sns.countplot(y = Fighter, order=pd.value_counts(Fighter).iloc[:10].index)

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, RandomizedSearchCV

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings("ignore")
df = new.copy()
df.isnull().sum()
df = df.fillna(df.mean())
from statistics import mode 

df['B_Stance'] = df['B_Stance'].fillna(df['B_Stance'].mode()[0])

df['R_Stance'] = df['R_Stance'].fillna(df['R_Stance'].mode()[0])
enc = LabelEncoder()
data_enc1 = df['weight_class']

data_enc1 = enc.fit_transform(data_enc1)



data_enc2 = df['R_Stance']

data_enc2 = enc.fit_transform(data_enc2)



data_enc3 = df['B_Stance']

data_enc3= enc.fit_transform(data_enc3)
data_enc1 = pd.DataFrame(data_enc1, columns=['weight_class'])

data_enc2 = pd.DataFrame(data_enc2, columns=['R_Stance'])

data_enc3 = pd.DataFrame(data_enc3, columns=['B_Stance'])
df[['weight_class']] = data_enc1[['weight_class']]

df[['R_Stance']] = data_enc2[['R_Stance']]

df[['B_Stance']] = data_enc3[['B_Stance']]
df = pd.concat([df,pd.get_dummies(df['win_by'], prefix='win_by')],axis=1)

df.drop(['win_by'],axis=1, inplace=True)
df['Winner_num'] = df.Winner.map({'Red':0,'Blue':1,'Draw':2})
df.head()
encode = df[['R_fighter','B_fighter','weight_class']].apply(enc.fit_transform)

encode.head()
df[['R_fighter','B_fighter','weight_class']] = encode[['R_fighter','B_fighter','weight_class']] 
df = df.dropna()

sum(df.isnull().sum())
plt.figure(figsize=(10,15))

sns.heatmap(df.corr()[['Winner_num']].sort_values(by='Winner_num', ascending=False),annot=True)

plt.show()
numerical = df.drop(['R_fighter','B_fighter','weight_class','no_of_rounds','Winner_num'], axis=1)
std = StandardScaler()

df_num = numerical.select_dtypes(include=[np.float, np.int])
numerical[list(df_num.columns)] = std.fit_transform(numerical[list(df_num.columns)])
df_fix = numerical.join(df[['R_fighter','B_fighter','weight_class','no_of_rounds','Winner_num']])
df_fix.head()
df_fix = df_fix.drop(['country','location','date_year','date','Referee','Format','last_round_time','Fight_type','Winner'], axis=1)
model = XGBClassifier()
X = df_fix.drop(['Winner_num'], axis=1)

y = df_fix['Winner_num']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
Score = model.score(X_test,y_test)

print("Score: %.2f%%" % (Score * 100.0))
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc

from itertools import cycle

lw=1
X1 = df_fix.drop(['Winner_num'], axis=1)

y1 = df_fix['Winner_num']

y1 = label_binarize(y1, classes=[0, 1, 2])

n_classes = y1.shape[1]
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size = 0.25, random_state = 42)
pred_proba = model.predict_proba(X1_test)
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y1_test[:, i], pred_proba[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['blue', 'red', 'green'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic for multi-class data')

plt.legend(loc="lower right")

plt.show()