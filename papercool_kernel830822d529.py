import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



pd.set_option('display.max_columns', 30)

pd.set_option('display.max_colwidth', 120)



import warnings

warnings.filterwarnings(action='ignore')



import os

print(os.listdir("../input"))
# df = pd.read_csv('drive/python/UCI_Credit_Card.csv')

df = pd.read_csv('../input/UCI_Credit_Card.csv')
print(df.shape)

df.sample(10)

df.head()

df.dtypes
print(df.columns)

df.describe()
df.rename(inplace=True, columns={'default.payment.next.month': 'DEFAULT'}) 

df.columns
df.groupby('DEFAULT').size()
df.groupby('DEFAULT').hist(figsize=(20,20))

plt.show()

plt.close()
df.drop('ID', axis=1, inplace=True)

df.head()
df.isnull().any()
df_bad = df[(df['BILL_AMT1'] <= 0) & (df['DEFAULT'] == 1)]

print(df_bad.shape)

df_bad[['BILL_AMT1','DEFAULT']].sample(10)
for index, row in df.iterrows():

    if (row['BILL_AMT1'] <= 0) & (row['DEFAULT'] == 1):

        df.drop(index, axis=0, inplace=True)

df.shape
for index, row in df.iterrows():

    if (row['EDUCATION'] >= 4) | (row['EDUCATION'] == 0):

        df.drop(index, axis=0, inplace=True)

df.shape
for index, row in df.iterrows():

    if (row['MARRIAGE'] == 0):

        df.drop(index, axis=0, inplace=True)

df.shape
df.groupby('DEFAULT').size()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



from sklearn.linear_model import Ridge



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVM', SVC()))

models.append(('LR', LogisticRegression()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('GB', GradientBoostingClassifier()))
array = df.values

X_sel = array[:,0:23]

Y_sel = array[:,23]

features = df.columns[:-1]
main_df = pd.DataFrame(columns=['Num_Of_Feature','Features_Sel','KNN','SVM','LR','DT','GNB','RF','GB'])

model_feat = LogisticRegression()



s_highest = 0



for n in range(3,11):

    print("Running selecting ", n ," features to run on all models..." )

    rfe = RFE(model_feat, n)

    fit = rfe.fit(X_sel, Y_sel)

    

    features_sel = []

    for sel, col in zip((fit.support_),features):

        if sel == True:

            features_sel.append(col)

    

    x = df[(features_sel)]

    y = df.DEFAULT



    x_train, x_test, y_train, y_test = train_test_split(

    x, y, stratify = df.DEFAULT, random_state=123)

    

    names = []

    scores = []

    for name, model in models:

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        score = accuracy_score(y_test, y_pred)

        scores.append(score)

        names.append(name)

        if score > s_highest:

            s_highest = score

            f_highest = features_sel

            n_highest = name

            m_highest = model

            

    main_df = main_df.append({'Num_Of_Feature':n,

                              'Features_Sel':(", ".join(features_sel)),

                              names[0]:scores[0],

                              names[1]:scores[1],

                              names[2]:scores[2],

                              names[3]:scores[3],

                              names[4]:scores[4],

                              names[5]:scores[5],

                              names[6]:scores[6]},

                             ignore_index=True)



print('The highest score is',s_highest,'with these features',f_highest,'on model',n_highest)
main_df
fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

main_df.plot(kind='line',x='Num_Of_Feature',y='KNN',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='SVM',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='LR',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='DT',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='GNB',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='RF',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='GB',ax=ax)



ax.set_xticks(np.arange(3, 11, step=1.0))

ax.set_yticks(np.arange(0.73, 0.85, step=0.01))



plt.show()

plt.close()
main_df.describe()
x = df[(f_highest)]

y = df.DEFAULT



x_train, x_test, y_train, y_test = train_test_split(

x, y, stratify = df.DEFAULT, random_state=123)
model = m_highest

model
# from sklearn.model_selection import GridSearchCV

# parameters = {

#     "loss":["deviance"],

#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

#     "min_samples_split": np.linspace(0.1, 0.5, 12),

#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),

#     "max_depth":[3,5,8],

#     "max_features":["log2","sqrt"],

#     "criterion": ["friedman_mse",  "mae"],

#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

#     "n_estimators":[10]

#     }

#

# clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



parameters = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.1, 0.2],

    "max_depth":[3,5,8],

    "criterion": ["friedman_mse"],

    "subsample":[0.5, 0.8, 1.0],

    "n_estimators":[100]

    }



clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

# clf = RandomizedSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_iter=100, n_jobs=-1)
import datetime

#Starting time

print("Start time is",datetime.datetime.now())



#Beware: This line of code can takes hours to run depend of the parameters setting above 

clf.fit(x, y)



#Stop time

print("Stop time is",datetime.datetime.now())
print(clf.best_params_)
print(clf.best_estimator_)
final_score = cross_val_score(clf.best_estimator_, x, y, 

                              cv=10, scoring='accuracy').mean()

print("Final accuracy : {} ".format(final_score))
df1 = df[df['DEFAULT']==1]

df1.shape



temp_list = [x for x in df1['EDUCATION'] if x == 1]

GS = len(temp_list)/len(df1)

temp_list = [x for x in df1['EDUCATION'] if x == 2]

UNI = len(temp_list)/len(df1)

temp_list = [x for x in df1['EDUCATION'] if x == 3]

HS = len(temp_list)/len(df1)



data = {'Graduate School': [GS], 'University': [UNI], 'High School':[HS]}

df2 = pd.DataFrame.from_dict(data)



df2.plot.bar(stacked=True, title ='EDUCATION %',figsize=(10,6))

plt.show()

plt.close()



df2.rename(index={0: 'EDUCATION'})
temp_list = [x for x in df1['MARRIAGE'] if x == 1]

MA = len(temp_list)/len(df1)

temp_list = [x for x in df1['MARRIAGE'] if x == 2]

SG = len(temp_list)/len(df1)

temp_list = [x for x in df1['MARRIAGE'] if x == 3]

DV = len(temp_list)/len(df1)



data = {'Married': [MA], 'Single': [SG], 'Divorce':[DV]}

df3 = pd.DataFrame.from_dict(data)



df3.plot.bar(stacked=True, title ='MARRIAGE %',figsize=(10,6))

plt.show()

plt.close()



df3.rename(index={0: 'MARRIAGE'})
temp_list = [x for x in df1['SEX'] if x == 1]

MA = len(temp_list)/len(df1)

temp_list = [x for x in df1['SEX'] if x == 2]

FE = len(temp_list)/len(df1)



data = {'Male': [MA], 'Female': [FE]}

df4 = pd.DataFrame.from_dict(data)



df4.plot.bar(stacked=True, title ='SEX %',figsize=(10,6))

plt.show()

plt.close()



df4.rename(index={0: 'SEX'})
df1['AGE GRP'] = pd.cut(df1['AGE'], [0, 31, 41, 51, 61, 101], labels=['Below 30', '31-40', '41-50', '51-60', 'Above 61'])



temp_list = [x for x in df1['AGE GRP'] if x == 'Below 30']

GP1 = len(temp_list)/len(df1)

temp_list = [x for x in df1['AGE GRP'] if x == '31-40']

GP2 = len(temp_list)/len(df1)

temp_list = [x for x in df1['AGE GRP'] if x == '41-50']

GP3 = len(temp_list)/len(df1)

temp_list = [x for x in df1['AGE GRP'] if x == '51-60']

GP4 = len(temp_list)/len(df1)

temp_list = [x for x in df1['AGE GRP'] if x == 'Above 61']

GP5 = len(temp_list)/len(df1)



data = {'Below 30': [GP1], '31-40': [GP2], '41-50':[GP3], '51-60':[GP4], 'Above 61':[GP5]}

df4 = pd.DataFrame.from_dict(data)



df4.plot.bar(stacked=True, title ='AGE GROUP %',figsize=(10,6))

plt.show()

plt.close()



df4.rename(index={0: 'AGE GROUP'})