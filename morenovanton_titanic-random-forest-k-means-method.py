import numpy as np 

import pandas as pd 

import math

import random

import seaborn as sns  

import matplotlib.pyplot as plt

import plotly.express as px



from sklearn.preprocessing import LabelEncoder



from sklearn import tree

from sklearn.model_selection import train_test_split



from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV





from sklearn.cluster import KMeans
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')
X_train = train.reset_index()

X_test = test.reset_index()

Y = train.Survived



X_dropna = X_train.dropna()

X_test_dropna = X_test.dropna()



Dframe = [X_train, X_test]
print(X_train.isnull().sum())

print(X_test.isnull().sum())
for frame in Dframe:

    

    #frame['military'] = 0

    #frame['aristo'] = 0

    

    frame['Title'] = frame.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

    frame['Title'] = frame['Title'].replace('Ms','Miss')

    frame['Title'] = frame['Title'].replace('Mlle','Miss')

    frame['Title'] = frame['Title'].replace('Mme','Mrs')



    frame['famous_cabin'] = frame["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    frame['Deck'] = frame['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

    frame.loc[(frame['Deck'] == 'T'), 'Deck'] = 'A'

    

    frame['IsWomanOrBoy'] = ((frame.Title == 'Master') | (frame.Sex == 'female'))

    

    frame['FamilySize'] = frame['SibSp'] + frame['Parch'] + 1

    frame['IsAlone'] = 0

    frame.loc[frame['FamilySize'] == 1, 'IsAlone'] = 1

    

    frame['Embarked'] = frame['Embarked'].fillna('S')

    frame['Fare'] = frame['Fare'].fillna(frame['Fare'].median())

    

    frame['Name_length'] = frame['Name'].apply(len)

    

    #frame.loc[frame['Title'] == 'Capt', 'military'] = 1

    #frame.loc[frame['Title'] == 'Major', 'military'] = 1

    

    #frame.loc[frame['Title'] == 'Don', 'aristo'] = 1

    #frame.loc[frame['Title'] == 'Jonkheer', 'aristo'] = 1

    #frame.loc[frame['Title'] == 'Sir', 'aristo'] = 1

    #frame.loc[frame['Title'] == 'the Countess', 'aristo'] = 1
X_train.head(2)
X_test.head(2)
plt.figure(figsize=[12,10])



plt.subplot(331)

sns.distplot(X_train['Pclass'], color="k")



plt.subplot (332)                         

sns.barplot('Sex', 'Survived', data=X_train)



plt.subplot (333)

sns.distplot(X_dropna['Age']) 



plt.subplot (334)

#sns.distplot(X['SibSp'])

sns.barplot('SibSp', 'Survived', data=X_train)



plt.subplot (335)

#sns.distplot(X['Parch'])

#px.histogram(X, x="Parch").show()

sns.barplot('Parch', 'Survived', data=X_train)



plt.subplot (336)

sns.distplot(X_train['Fare'])

#train['Fare'].hist()



plt.subplot (337)

sns.barplot('Embarked', 'Survived', data=X_train)
X_dropna_categor = pd.get_dummies(X_dropna) 

Xtest_dropna_categor = pd.get_dummies(X_test_dropna) 



# Surviving girls:

Sex_female_Survived = X_dropna_categor.loc[(X_dropna_categor.Sex_female == 1) & (X_dropna_categor.Survived == 1)]

# Dead girls:

Sex_female_NoSurvived = X_dropna_categor.loc[(X_dropna_categor.Sex_female == 1) & (X_dropna_categor.Survived == 0)]

# Surviving guys:

X_Sex_male_Survived = X_dropna_categor.loc[(X_dropna_categor.Sex_male == 1) & (X_dropna_categor.Survived == 1)] 

# Dead guys:

X_Sex_male_NoSurvived = X_dropna_categor.loc[(X_dropna_categor.Sex_male == 1) & (X_dropna_categor.Survived == 0)]



X_test_male = Xtest_dropna_categor.loc[Xtest_dropna_categor.Sex_male == 1]

X_test_female = Xtest_dropna_categor.loc[Xtest_dropna_categor.Sex_female == 1]
# age distribution of survivors and non-survivors:

sns.set(rc={'figure.figsize': (15, 9)})

plt.subplot (221)

sns.distplot(Sex_female_Survived['Age'] , kde_kws = {'color': 'g', 'lw':1, 'label': 'Sex_female_Survived' })

plt.subplot (222)

sns.distplot(Sex_female_NoSurvived['Age'] , kde_kws = {'color': 'r', 'lw':1, 'label': 'Sex_female_NoSurvived' })

plt.subplot (223)

sns.distplot(X_Sex_male_Survived['Age'] , kde_kws = {'color': 'blue', 'lw':1, 'label': 'X_Sex_male_Survived' })

plt.subplot (224)

sns.distplot(X_Sex_male_NoSurvived['Age'] , kde_kws = {'color': 'gray', 'lw':1, 'label': 'X_Sex_male_NoSurvived' })
female_Survived_mean, female_NoSurvived_mean = Sex_female_Survived['Age'].mean(), Sex_female_NoSurvived['Age'].mean()

male_Survived_mean, male_NoSurvived_mean = X_Sex_male_Survived['Age'].mean(), X_Sex_male_NoSurvived['Age'].mean()



female_Survived_std, female_NoSurvived_std = Sex_female_Survived['Age'].std(), Sex_female_NoSurvived['Age'].std()

male_Survived_std, male_NoSurvived_std = X_Sex_male_Survived['Age'].std(), X_Sex_male_NoSurvived['Age'].std()



female_std, female_mean = X_test_female['Age'].std(), X_test_female['Age'].mean()

male_std, male_mean = X_test_male['Age'].std(), X_test_male['Age'].mean()
# Confidence interval calculation function: 

def derf(sample, mean, std):

    age_shape = sample['Age'].shape[0] # sample size

    standard_error_ofthe_mean = std / math.sqrt(age_shape)

    random_mean = random.uniform(mean-(1.96*standard_error_ofthe_mean), mean+(1.96*standard_error_ofthe_mean))

    return round(random_mean, 2)    
for i in X_train.loc[(X_train['Sex']=='female') & (X_train['Survived']==1) & (X_train['Age'].isnull())].index:

    X_train.at[i, 'Age'] = derf(Sex_female_Survived, female_Survived_mean, female_Survived_std)



for h in X_train.loc[(X_train['Sex']=='female') & (X_train['Survived']==0) & (X_train['Age'].isnull())].index:

    X_train.at[h, 'Age'] = derf(Sex_female_NoSurvived, female_NoSurvived_mean, female_NoSurvived_std)

    

for l in X_train.loc[(X_train['Sex']=='male') & (X_train['Survived']==1) & (X_train['Age'].isnull())].index:

    X_train.at[l, 'Age'] = derf(X_Sex_male_Survived, male_Survived_mean, male_Survived_std)

    

for b in X_train.loc[(X_train['Sex']=='male') & (X_train['Survived']==0) & (X_train['Age'].isnull())].index:

    X_train.at[b, 'Age'] = derf(X_Sex_male_NoSurvived, male_NoSurvived_mean, male_NoSurvived_std)

    

for p in X_test.loc[(X_test['Sex']=='male') & (X_test['Age'].isnull())].index:

    X_test.at[p, 'Age'] = derf(X_test_male, male_mean, male_std)



for y in X_test.loc[(X_test['Sex']=='female') & (X_test['Age'].isnull())].index:

    X_test.at[y, 'Age'] = derf(X_test_female, female_mean, female_std)

print(X_train.isnull().sum())

print(X_test.isnull().sum())
X_train = X_train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'index'], axis=1) 

X_test = X_test.drop(['Name', 'Ticket', 'Cabin', 'index'], axis=1)
# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns_train = []

features = X_train.columns.values.tolist()

for col in features:

    if X_train[col].dtype in numerics: continue

    categorical_columns_train.append(col)

categorical_columns_train
# Encoding categorical features

for col in categorical_columns_train:

    if col in X_train.columns:

        le = LabelEncoder()

        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))

        X_train[col] = le.transform(list(X_train[col].astype(str).values))

        X_test[col] = le.transform(list(X_test[col].astype(str).values))   
cor_map = plt.cm.RdBu

plt.figure(figsize=(15,17))

plt.title('Pearson Correlation', y=1.05, size=15)

sns.heatmap(X_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=cor_map, linecolor='white', annot=True)  
X_train_rf_1, X_train_rf_2, y_train_rf_1, y_train_rf_2 = train_test_split(

    X_train, Y, test_size=0.33, random_state=42)



print(X_train_rf_1.shape)

print(X_train_rf_2.shape)
rf = RandomForestClassifier()

best_rf = {}
random_grid = {'criterion': ['gini', 'entropy'],

               'bootstrap': [True, False],

               'max_depth': [3, 5, 7, 9, 11, 13, 16, 19, 20],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [5, 10, 15, 20, 25],

               'min_samples_split': [40, 50, 60, 62, 64, 66, 68],

               'n_estimators': [300, 600, 900, 1200, 1500, 1800]}
rf_search_one = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv=5, n_iter = 100)

rf_search_two = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv=5, n_iter = 200)
#rf_search_one.fit(X_train, Y)

rf_search_one.fit(X_train_rf_1, y_train_rf_1)

best_rf_s1 = rf_search_one.best_estimator_

#best_rf_s1.fit(X_train, Y)

best_rf_s1.fit(X_train_rf_1, y_train_rf_1)

y_predicted_prob_1 = best_rf_s1.predict_proba(X_test)  

print(best_rf_s1.classes_)

y_predicted_prob_1 = list(y_predicted_prob_1)

best_rf['rf_ndom'] = y_predicted_prob_1

print('finish')
rf_search_two.fit(X_train_rf_2, y_train_rf_2)

best_rf_s2 = rf_search_two.best_estimator_

best_rf_s2.fit(X_train_rf_2, y_train_rf_2)

y_predicted_prob_2 = best_rf_s2.predict_proba(X_test) 

print(best_rf_s1.classes_)

y_predicted_prob_2 = list(y_predicted_prob_2)

best_rf['rf_ndom2'] = y_predicted_prob_2

print('finish')
print(rf_search_one.best_params_)

print(rf_search_two.best_params_)
survivor_probability_dictionary = {}

Dictionary_Probability_Dead = {}

for ary in best_rf:

    survivor_probability = []

    Nosurvivor_probability = []

    for i in best_rf[ary]:

        survivor_probability.append(i[1])

        Nosurvivor_probability.append(i[0])

    survivor_probability_dictionary[ary] = survivor_probability

    Dictionary_Probability_Dead[ary] = Nosurvivor_probability
temp_y = pd.DataFrame.from_dict({

    'PassengerId': list(test.PassengerId), 

    'rf_1_survivor': survivor_probability_dictionary['rf_ndom'],

    'rf_2_survivor': survivor_probability_dictionary['rf_ndom2'],

    'rf_1_Nosurvivor': Dictionary_Probability_Dead['rf_ndom'],

    'rf_2_Nosurvivor': Dictionary_Probability_Dead['rf_ndom2'] })
temp_y.head()
plt.figure(figsize=[20,20])



plt.subplot(341)

ax = sns.scatterplot(x="rf_1_survivor", y="rf_1_Nosurvivor", data=temp_y)

plt.subplot(342)

ax = sns.scatterplot(x="rf_2_survivor", y="rf_2_Nosurvivor", data=temp_y)



plt.subplot(343)

ax = sns.scatterplot(x="rf_1_survivor", y="rf_2_survivor", data=temp_y)

plt.subplot(344)

ax = sns.scatterplot(x="rf_1_Nosurvivor", y="rf_2_Nosurvivor", data=temp_y)   
ftest = ['rf_1_survivor', 'rf_2_survivor']
n_clusters = 2

km = KMeans(n_clusters=n_clusters)

data = temp_y[[ftest[0], ftest[1]]]

temp_y['claster_rf1_surv'] = km.fit_predict(data)
ax = sns.lmplot(x='rf_1_survivor', y='rf_2_survivor', data = temp_y, hue='claster_rf1_surv', fit_reg=False)
#cluster_r = temp_y['claster_rf1_surv'].replace({0:1, 1:0})

#temp_y['claster_rf1_reversurv'] = cluster_r
#ax = sns.lmplot(x='rf_1_survivor', y='rf_2_survivor', data = temp_y, hue='claster_rf1_reversurv', fit_reg=False)
finall_F = pd.DataFrame.from_dict({'PassengerId': list(temp_y.PassengerId), 

                                   'Survived': list(temp_y.claster_rf1_surv)})
finall_F.head(6)
finall_F.to_csv("Submission.csv", index=False)