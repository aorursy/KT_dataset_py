import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.simplefilter('ignore')



import os

print(os.listdir("../input"))



pd.set_option('display.max_columns', 40)

pd.set_option('display.max_colwidth', 120)
df = pd.read_csv('../input/organics.csv')
df.shape
df.head()
df.columns
df.describe()
df.dtypes
df.groupby('Gender').size()
print('Gender',df['Gender'].unique())

print('Geographic Region',df['Geographic Region'].unique())

print('Loyalty Status',df['Loyalty Status'].unique())

print('Neighborhood Cluster-7 Level',df['Neighborhood Cluster-7 Level'].unique())

print('Television Region',df['Television Region'].unique())

print('Affluence Grade',df['Affluence Grade'].unique())

print('Age',df['Age'].unique())

print('Frequency Percent',df['Frequency Percent'].unique())

print('Loyalty Card Tenure',df['Loyalty Card Tenure'].unique())
i = 0

for x in df['Gender']:

    if x == 'U':

        df.iloc[i, df.columns.get_loc('Gender')] = np.NaN

    i = i + 1
i = 0

for x in df['Affluence Grade']:

    try:

        df.iloc[i, df.columns.get_loc('Affluence Grade')] = int(x)

    except:

        df.iloc[i, df.columns.get_loc('Affluence Grade')] = np.NaN

    finally:

        i = i + 1
i = 0

for x in df['Age']:

    try:

        df.iloc[i, df.columns.get_loc('Age')] = int(x)

    except:

        df.iloc[i, df.columns.get_loc('Age')] = np.NaN

    finally:

        i = i + 1
i = 0

for x in df['Loyalty Card Tenure']:

    try:

        df.iloc[i, df.columns.get_loc('Loyalty Card Tenure')] = int(x)

    except:

        df.iloc[i, df.columns.get_loc('Loyalty Card Tenure')] = np.NaN

    finally:

        i = i + 1
print('Gender',df['Gender'].unique())

print('Geographic Region',df['Geographic Region'].unique())

print('Loyalty Status',df['Loyalty Status'].unique())

print('Neighborhood Cluster-7 Level',df['Neighborhood Cluster-7 Level'].unique())

print('Television Region',df['Television Region'].unique())

print('Affluence Grade',df['Affluence Grade'].unique())

print('Age',df['Age'].unique())

print('Frequency Percent',df['Frequency Percent'].unique())

print('Loyalty Card Tenure',df['Loyalty Card Tenure'].unique())
df.isnull().any()

df.isnull().sum(axis=0)
df.dropna(subset=['Gender','Age','Affluence Grade'],inplace=True)
df.isnull().any()

df.isnull().sum(axis=0)
df.shape
col_name = df.columns

col_name
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="most_frequent")

temp_arr = imp.fit_transform(df)
df2 = pd.DataFrame.from_records(data=temp_arr,columns=col_name)
df2.sample(10)
df2.isnull().any()

df2.isnull().sum(axis=0)
print('Gender',df2['Gender'].unique())

print('Geographic Region',df2['Geographic Region'].unique())

print('Loyalty Status',df2['Loyalty Status'].unique())

print('Neighborhood Cluster-7 Level',df2['Neighborhood Cluster-7 Level'].unique())

print('Television Region',df2['Television Region'].unique())

print('Affluence Grade',df2['Affluence Grade'].unique())

print('Age',df2['Age'].unique())

print('Frequency Percent',df2['Frequency Percent'].unique())

print('Loyalty Card Tenure',df2['Loyalty Card Tenure'].unique())
df2.rename(inplace=True, columns={'Organics Purchase Indicator': 'ORGANICS'}) 

df2.columns
df2.groupby('ORGANICS').hist(figsize=(20,20))

plt.show()

plt.close()
df2.drop(['Customer Loyalty ID','Organics Purchase Count'], axis=1, inplace=True)

df2.head()
df2.drop(['Frequency','Frequency Percent'], axis=1, inplace=True)

df2.head()
df2.dtypes
df3 = pd.get_dummies(data=df2, columns=['Gender','Geographic Region','Loyalty Status',

                                        'Neighborhood Cluster-7 Level','Television Region'])
df3.shape
df3.head()
df3.columns
df3 = df3[['Neigborhood Cluster-55 Level', 'Affluence Grade', 'Age',

       'Loyalty Card Tenure', 'Total Spend', 'Gender_F',

       'Gender_M', 'Geographic Region_Midlands', 'Geographic Region_North',

       'Geographic Region_Scottish', 'Geographic Region_South East',

       'Geographic Region_South West', 'Loyalty Status_Gold',

       'Loyalty Status_Platinum', 'Loyalty Status_Silver',

       'Loyalty Status_Tin', 'Neighborhood Cluster-7 Level_A',

       'Neighborhood Cluster-7 Level_B', 'Neighborhood Cluster-7 Level_C',

       'Neighborhood Cluster-7 Level_D', 'Neighborhood Cluster-7 Level_E',

       'Neighborhood Cluster-7 Level_F', 'Neighborhood Cluster-7 Level_U',

       'Television Region_Border', 'Television Region_C Scotland',

       'Television Region_East', 'Television Region_London',

       'Television Region_Midlands', 'Television Region_N East',

       'Television Region_N Scot', 'Television Region_N West',

       'Television Region_S & S East', 'Television Region_S West',

       'Television Region_Ulster', 'Television Region_Wales & West',

       'Television Region_Yorkshire','ORGANICS']]
df3.head()
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



# Build a list with all models

models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVM', SVC()))

models.append(('LR', LogisticRegression()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('GB', GradientBoostingClassifier()))
features = df3.columns[:-1]

x_m = df3[features]

y_m = df3.ORGANICS
from sklearn.model_selection import train_test_split



X_F_TRAIN, X_F_TEST, Y_F_TRAIN, Y_F_TEST = train_test_split(

    x_m, y_m, test_size=0.2, stratify = y_m, random_state=42)
main_df = pd.DataFrame(columns=['Num_Of_Feature','Features_Sel','KNN','SVM','LR','DT','GNB','RF','GB'])

model_feat = LogisticRegression()



s_highest = 0



for n in range(3,38):

    print("Running selecting ", n ," features to run on all models..." )

    rfe = RFE(model_feat, n)

    fit = rfe.fit(X_F_TRAIN, Y_F_TRAIN)

    

    features_sel = []

    for sel, col in zip((fit.support_),features):

        if sel == True:

            features_sel.append(col)

    

    x = X_F_TRAIN[(features_sel)]

    y = Y_F_TRAIN



    x_train, x_test, y_train, y_test = train_test_split(

    x, y, stratify = y, random_state=42)

    

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



#print('The highest score is',s_highest,'with these features',f_highest,'on model',n_highest)
fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

main_df.plot(kind='line',x='Num_Of_Feature',y='KNN',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='SVM',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='LR',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='DT',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='GNB',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='RF',ax=ax)

main_df.plot(kind='line',x='Num_Of_Feature',y='GB',ax=ax)



ax.set_xticks(np.arange(3, 38, step=1.0))

ax.set_yticks(np.arange(0.70, 0.83, step=0.01))



plt.show()

plt.close()
main_df
strlist = list(main_df[main_df['Num_Of_Feature'] == 20]['Features_Sel'])
str1 = str(strlist)

str2 = str1[2:-2]

f_selected = str2.split(", ")

f_selected
x = X_F_TRAIN[(f_selected)]

y = Y_F_TRAIN



x_train, x_test, y_train, y_test = train_test_split(

x, y, stratify = y, random_state=42)
model = m_highest

model
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
model = clf.best_estimator_

x_train = X_F_TRAIN[(f_selected)]

y_train = Y_F_TRAIN



x_test = X_F_TEST[(f_selected)]

y_test = Y_F_TEST
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("The accuracy for unseen data is",score)
df4 = df2[df2['ORGANICS']==1]



temp_list = [x for x in df4['Gender'] if x == 'M']

MA = len(temp_list)/len(df4)

temp_list = [x for x in df4['Gender'] if x == 'F']

FE = len(temp_list)/len(df4)



data = {'Male': [MA], 'Female': [FE]}

df5 = pd.DataFrame.from_dict(data)



df5.plot.bar(stacked=True, title ='Gender %',figsize=(10,6))

plt.show()

plt.close()



df5.rename(index={0: 'Gender'})
df4['AGE GRP'] = pd.cut(df4['Age'], [0, 31, 41, 51, 61, 101], labels=['Below 30', '31-40', '41-50', '51-60', 'Above 61'])



temp_list = [x for x in df4['AGE GRP'] if x == 'Below 30']

GP1 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AGE GRP'] if x == '31-40']

GP2 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AGE GRP'] if x == '41-50']

GP3 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AGE GRP'] if x == '51-60']

GP4 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AGE GRP'] if x == 'Above 61']

GP5 = len(temp_list)/len(df4)



data = {'Below 30': [GP1], '31-40': [GP2], '41-50':[GP3], '51-60':[GP4], 'Above 61':[GP5]}

df5 = pd.DataFrame.from_dict(data)



df5.plot.bar(stacked=True, title ='AGE GROUP %',figsize=(10,6))

plt.show()

plt.close()



df5.rename(index={0: 'AGE GROUP'})
df4['AFF GRP'] = pd.cut(df4['Affluence Grade'], [0, 11, 16, 21, 26, 40], labels=['Below 10', '11-15', '16-20', '21-25', 'Above 26'])



temp_list = [x for x in df4['AFF GRP'] if x == 'Below 10']

GP1 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AFF GRP'] if x == '11-15']

GP2 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AFF GRP'] if x == '16-20']

GP3 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AFF GRP'] if x == '21-25']

GP4 = len(temp_list)/len(df4)

temp_list = [x for x in df4['AFF GRP'] if x == 'Above 26']

GP5 = len(temp_list)/len(df4)



data = {'Below 10': [GP1], '11-15': [GP2], '16-20':[GP3], '21-25':[GP4], 'Above 26':[GP5]}

df5 = pd.DataFrame.from_dict(data)



df5.plot.bar(stacked=True, title ='Affluence Grade %',figsize=(10,6))

plt.show()

plt.close()



df5.rename(index={0: 'AFFLUENCE GROUP'})
temp_list = [x for x in df4['Loyalty Status'] if x == 'Gold']

GO = len(temp_list)/len(df4)

temp_list = [x for x in df4['Loyalty Status'] if x == 'Silver']

SI = len(temp_list)/len(df4)

temp_list = [x for x in df4['Loyalty Status'] if x == 'Tin']

TI = len(temp_list)/len(df4)

temp_list = [x for x in df4['Loyalty Status'] if x == 'Platinum']

PL = len(temp_list)/len(df4)





data = {'Gold': [GO], 'Silver': [SI], 'Tin': [TI], 'Platinum': [PL]}

df5 = pd.DataFrame.from_dict(data)



df5.plot.bar(stacked=True, title ='Loyalty Status %',figsize=(10,6))

plt.show()

plt.close()



df5.rename(index={0: 'Loyalty Status'})