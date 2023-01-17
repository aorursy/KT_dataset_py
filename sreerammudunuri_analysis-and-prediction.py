#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 8

import warnings

warnings.filterwarnings(action="ignore")
# reading the data

df = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

df.head()
df.info()
df.isnull().sum()
df_non_nota = df.dropna()

df_non_nota.head()
df_non_nota.isnull().sum()
# removing unwanted data



df_non_nota.drop(['SYMBOL','ASSETS','LIABILITIES','GENERAL\nVOTES','POSTAL\nVOTES','OVER TOTAL ELECTORS \nIN CONSTITUENCY','TOTAL ELECTORS'],axis=1,inplace=True)

df_non_nota.head()
df_non_nota.info()
# Education Qualification of Candidates for LOK SABHA - 2019



df_non_nota.EDUCATION.value_counts()
sns.countplot(df_non_nota['EDUCATION'])
# differents categories in LOK SHABA ELECTIONS

df_non_nota.CATEGORY.value_counts()
sns.countplot(df_non_nota['CATEGORY'])
a = sns.countplot(data=df_non_nota,x='CATEGORY',hue='WINNER')

a.set_title('No of Candidates won or lost by category wise',fontsize=20)
df_non_nota.AGE.value_counts()
df_non_nota.AGE.value_counts().plot.bar(title='AGE of LOK SABHA candidates')
df_non_nota.GENDER.value_counts()
df_non_nota.GENDER.value_counts().plot.bar(title='No of MALES and FEMALES Participated in LOK SHABA Elections')
a = sns.countplot(data=df_non_nota,x='GENDER',hue='WINNER')

a.set_title('No of Males and Females Won and Lost in LOK SHABA Elections',fontsize=20)
df_non_nota.PARTY.value_counts()
# Number of Seats Contested by PARTIES 

df_non_nota.PARTY.value_counts().plot.bar(title='Number of Seats Contested by PARTIES (TOP 10)',figsize=(26,10))

# Number of Seats WON by Parties (TOP 10)

df_non_nota['PARTY'][(df_non_nota['WINNER']==1)].value_counts().head(10).plot.bar()
top_20parties = pd.Series(df_non_nota['PARTY'].value_counts().head(21))

top_20parties = top_20parties.index.drop(['IND'])



top_20parties
# Creating DataFrame which consists of Top 20 Parties on the basis of Seats Contested



df_party_wise_seats_comparison = pd.DataFrame(columns=df_non_nota.columns)



for count,party in enumerate(df['PARTY']):

    if party in top_20parties:

        df_party_wise_seats_comparison = df_party_wise_seats_comparison.append(df.loc[count],ignore_index=True)
# Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)



a = sns.countplot(x='PARTY',hue='WINNER',data=df_party_wise_seats_comparison,palette='Set1')

a.set_title('Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)',fontsize=20)

a.legend(['Lost','Won'],loc='upper right',frameon=False),

def fun(num):

    if num == '0':

        return 0

    else:

        return 1
df_non_nota['CRIMINAL_BACKGROUND'] = df_non_nota['CRIMINAL\nCASES'].apply(fun)

df_non_nota.head()
df_non_nota.CRIMINAL_BACKGROUND.value_counts()
a = sns.countplot(df_non_nota['CRIMINAL_BACKGROUND'])

a=sns.countplot(data=df_non_nota,x='WINNER',hue='CRIMINAL_BACKGROUND')

a.legend(['WON','LOST'],loc='upper right',frameon=False)

a.set_title('Number of NON-CRIMINAL AND CRIMINAL BACKGROUND CANDIDATES Won and Lost',fontsize=20)


top_20_crime_cand = df_non_nota['PARTY'][df_non_nota['CRIMINAL\nCASES']!='0'].sort_index().value_counts().head(20)

top_20_crime_cand = top_20_crime_cand.index



top_20_crime_cand
# Creating DataFrame consisting of Top 10 Political Parties having most number of CRIMINAL CANDIDATES



df_top_20_crime_parties = df_non_nota.copy()



for party,index in zip(df_top_20_crime_parties['PARTY'],df_top_20_crime_parties['PARTY'].index):

    if party not in top_20_crime_cand:

        df_top_20_crime_parties.drop(index=index, inplace=True)
# Political Party Candidates CRIMINAL BACKGROUND check (TOP 20)

ax = sns.countplot(data=df_top_20_crime_parties,x='PARTY',hue='CRIMINAL_BACKGROUND')



ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],loc='upper right',frameon=False)

ax.set_title('Political Party Candidates CRIMINAL BACKGROUND check (TOP 20)',fontsize=20)
crime_background  = df_non_nota[df_non_nota['CRIMINAL_BACKGROUND'] == 1]

crime_background.head()
top_20_crimebkg =  crime_background.PARTY.value_counts().head(21)

top_20_crimebkg = top_20_crimebkg.index.drop(['IND'])

top_20_crimebkg
top_20_crime_cand_parties = crime_background



for party,index in zip(top_20_crime_cand_parties['PARTY'],top_20_crime_cand_parties['PARTY'].index):

    if party not in top_20_crimebkg:

        top_20_crime_cand_parties.drop(index=index, inplace=True)

a = sns.countplot(data=top_20_crime_cand_parties,x='PARTY',hue='WINNER')

a.set_title('Top 20 CRIMINAL BACKGROUND CANDIDATES WON OR LOST ',fontsize=20)
df.dropna(inplace=True)
def categorizing(dat):

    cat = dat.astype('category').cat.codes

    return cat
df['STATE'] = categorizing(df['STATE'])

df['CONSTITUENCY'] = categorizing(df['CONSTITUENCY'])

df['NAME'] = categorizing(df['NAME'])

df['PARTY'] = categorizing(df['PARTY'])

df['SYMBOL'] = categorizing(df['SYMBOL'])

df['GENDER'] = categorizing(df['GENDER'])

df['CATEGORY'] = categorizing(df['CATEGORY'])

df['EDUCATION'] = categorizing(df['EDUCATION'])

df['ASSETS'] = categorizing(df['ASSETS'])

df['LIABILITIES'] = categorizing(df['LIABILITIES'])
df.drop(['CRIMINAL\nCASES'],axis=1,inplace=True)

df.head()
df.dtypes
df.isnull().sum()
y = df['WINNER']

X = df.drop(['WINNER'],axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
logistic_regression = LogisticRegression(random_state=1)
logistic_regression.fit(x_train,y_train)
prediction=logistic_regression.predict(x_test)

prediction
acc1 = logistic_regression.score(x_test,y_test)

acc1
from sklearn.tree import DecisionTreeClassifier
D_tree = DecisionTreeClassifier()
D_tree.fit(x_train,y_train)
pred_dtree = D_tree.predict(x_test)

pred_dtree
acc2 = D_tree.score(x_test,y_test)

acc2
from sklearn.ensemble import RandomForestClassifier
Rforest_model = RandomForestClassifier(random_state=1,max_depth=10,n_estimators=50)
Rforest_model.fit(x_train,y_train)
pred_cv_forest=Rforest_model.predict(x_test)
acc3 = accuracy_score(pred_cv_forest,y_test)

acc3
from keras.models import Sequential

from keras.layers import Dense

from keras import regularizers
model=Sequential()

model.add(Dense(12, input_dim=17, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=32)
_,acc4=model.evaluate (x_test,y_test)

acc4
acc_data = pd.DataFrame(index=['Logistic Regression','Decision Tree','Random Forest','ANN'])

acc_data['Accuracy'] = [acc1,acc2,acc3,acc4]

acc_data