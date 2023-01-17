# data analysis packages - let us do staff

import pandas as pd

import numpy as np



#visualization packages - let us make things look pretty

import matplotlib.pyplot as plt

import seaborn as sns





#ML packages - cause what's the point without this?

#models

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet

from sklearn.linear_model import SGDClassifier as SGD

from sklearn.linear_model import Ridge

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import Perceptron

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

import xgboost as xgb



#around

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split

import random

#metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error as MSE



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



random.seed(42)
train_ds = pd.read_csv('../input/titanic/train.csv')

test_ds = pd.read_csv('../input/titanic/test.csv')
train_ds.info()
train_ds.head()
train_ds.describe()
train_ds.describe(include=['O'])

#calculate gender based survival rate:

males_survival_rate = train_ds[(train_ds['Sex'] == 'male')]['Survived'].mean() * 100

females_survival_rate = train_ds[(train_ds['Sex'] == 'female')]['Survived'].mean() * 100

print("males survival rate: " + str(males_survival_rate)+ "%\nfemales survival rate: "+str(females_survival_rate)+"%")
emb_sr = train_ds[['Embarked','Survived']].groupby(['Embarked']).mean()

print(emb_sr)

plt.bar(['C','Q','S'], emb_sr['Survived'])

plt.show()



pcls_sr = train_ds[['Pclass','Survived']].groupby(['Pclass']).mean()

print(pcls_sr)

plt.bar(['1','2','3'], pcls_sr['Survived'])

plt.show()



parch_sr = train_ds[['Parch','Survived']].groupby(['Parch'], as_index=False).mean()

print(parch_sr)

plt.bar(parch_sr['Parch'], parch_sr['Survived'])

plt.show()



sibsp_sr = train_ds[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean()

print(sibsp_sr)

plt.bar(sibsp_sr['SibSp'], sibsp_sr['Survived'])

plt.show()
train_ds[['Cabin', 'Survived']].groupby(['Cabin']).mean().head()
sur_ages = train_ds[train_ds['Survived'] == 1]['Age']

per_ages = train_ds[train_ds['Survived'] == 0]['Age']

per_total = train_ds[:]['Age']

plt.hist(per_total, 16, alpha=0.2, label='total', color='lightgray')

plt.hist(sur_ages, 16, alpha=0.4, label='survivors', color='blue')

plt.hist(per_ages, 16, alpha=0.4, label='perished', color='darkred')

plt.legend(loc='upper right')

plt.show()
sur_fares = train_ds[(train_ds['Survived'] == 1)&(train_ds['Fare'] <= 200)]['Fare']

per_fares = train_ds[(train_ds['Survived'] == 0)&(train_ds['Fare'] <= 200)]['Fare']

fares_tot = train_ds[(train_ds['Fare'] <= 200)]['Fare']

plt.hist(fares_tot, 15, alpha=0.5, label='total', color='lightgray')

plt.hist(sur_fares, 15, alpha=0.5, label='survivors', color='green')

# plt.hist(per_fares, 15, alpha=0.4, label='perished', color='darkred')

plt.legend(loc='upper right')

plt.show()
train_ds["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train_ds['Cabin'] ])



test_ds["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test_ds['Cabin'] ])

train_ds['Deck'] = 'C'

train_ds.loc[(train_ds['Cabin'] == 'A') | (train_ds['Cabin'] == 'B') | (train_ds['Cabin'] == 'K') | (train_ds['Cabin'] == 'L') | (train_ds['Cabin'] == 'M') | (train_ds['Cabin'] == 'O') | (train_ds['Cabin'] == 'P'), 'Deck'] = 'D'

train_ds.loc[(train_ds['Cabin'] == 'C') | (train_ds['Cabin'] == 'D') | (train_ds['Cabin'] == 'E') | (train_ds['Cabin'] == 'F') | (train_ds['Cabin'] == 'J'), 'Deck'] = 'E'



test_ds['Deck'] = 'C'

test_ds.loc[(test_ds['Cabin'] == 'A') | (test_ds['Cabin'] == 'B') | (test_ds['Cabin'] == 'K') | (test_ds['Cabin'] == 'L') | (test_ds['Cabin'] == 'M') | (test_ds['Cabin'] == 'O') | (test_ds['Cabin'] == 'P'), 'Deck'] = 'D'

test_ds.loc[(test_ds['Cabin'] == 'C') | (test_ds['Cabin'] == 'D') | (test_ds['Cabin'] == 'E') | (test_ds['Cabin'] == 'F') | (test_ds['Cabin'] == 'J'), 'Deck'] = 'E'

print(test_ds.head())
print(train_ds['Name'].head(10))

print("\n")

# Now we noticed that after each title comes '.' so we'll use this to select only the title word:



train_ds['Title'] = train_ds.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(train_ds.groupby('Title')['Title'].count())

print("\n")

# Lets group together things that are practically the same, 

train_ds.loc[(train_ds['Title'] == 'Ms') | (train_ds['Title'] == 'Mlle'), 'Title'] = 'Miss'

train_ds.loc[(train_ds['Title'] == 'Mme'), 'Title'] = 'Mrs'

print(train_ds.groupby('Title')['Title'].count())

print("\n")



#As we have many titles with a very small number of samples, we want to group them together to avoid overfitting

train_ds['Title'] = train_ds['Title'].replace(['Capt', 'Col','Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Special')

print(train_ds.groupby('Title')['Title'].count())

print("\n")
# making this change in the Test set as well to see it looks around the same

test_ds['Title'] = test_ds.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(test_ds.groupby('Title')['Title'].count())

print("\n")



test_ds.loc[(test_ds['Title'] == 'Ms'), 'Title'] = 'Miss'

test_ds['Title'] = test_ds['Title'].replace(['Col', 'Dona', 'Dr','Rev'], 'Special')

print(test_ds.groupby('Title')['Title'].count())

print("\n")
#updating the datasest:

train_ds = train_ds.drop('Name', axis=1)

test_ds = test_ds.drop('Name', axis=1)
train_ds.loc[train_ds['Embarked'].isnull(),'Embarked'] = 'S'

print(train_ds.info())
Sex_mapping = {"female": 0, "male": 1}

train_ds['Sex'] = train_ds['Sex'].map(Sex_mapping)

test_ds['Sex'] = test_ds['Sex'].map(Sex_mapping)

train_ds = pd.get_dummies(train_ds[['Survived','Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare','Deck','Cabin']])

test_ds = pd.get_dummies(test_ds[['PassengerId','Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare','Deck','Cabin']])



train_cols = train_ds.columns.tolist()

test_cols = test_ds.columns.tolist()

for col in train_cols:

    if col not in test_cols:

        test_ds.loc[:,col] = 0

for col in test_cols:

    if col not in train_cols:

        train_ds.loc[:,col] = 0

        

train_ds = train_ds.drop('PassengerId', axis=1)

test_ds = test_ds.drop('Survived', axis=1)
joined_ds = pd.concat(objs=[train_ds.drop('Survived', axis=1), test_ds.drop('PassengerId', axis=1)], axis=0).reset_index(drop=True)

Ages_to_train = joined_ds[joined_ds['Age'] >= 0]

Ages_to_train = Ages_to_train.dropna()

X = Ages_to_train.dropna().drop('Age', axis=1)

y = Ages_to_train['Age'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = GridSearchCV(Ridge(normalize=True), {'alpha': np.logspace(-1, 1, 21)}, n_jobs=-1, cv=5)

model.fit(X_train, y_train)



#from this link: https://www.kaggle.com/startupsci/titanic-data-science-solutions

guess_ages = np.zeros((2,3))

X_test.loc['Age_guessed'] = 0

for i in range(0, 2):

    for j in range(0, 3):

        guess_df = y_train[(X_train['Sex'] == i) & \

                              (X_train['Pclass'] == j+1)]

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        X_test.loc[(X_test.Sex == i) & (X_test.Pclass == j+1),'Age_guessed'] = guess_ages[i,j]



 # my part   

y_guessed = X_test['Age_guessed'].dropna()

y_ridged = model.predict(X_test.dropna().drop('Age_guessed', axis=1))

print("the guessed MSE score is:" + str(MSE(y_test, y_guessed)))

print("My MSE score is:" + str(MSE(y_test, y_ridged)))

X_test = X_test.drop('Age_guessed', axis=1)
joined_ds = pd.concat(objs=[train_ds.drop('Survived', axis=1), test_ds.drop('PassengerId', axis=1)], axis=0).reset_index(drop=True)

Ages_to_fill = train_ds[train_ds['Age'].isnull()]

Ages_to_train = joined_ds[joined_ds['Age'] >= 0]

Ages_to_train = Ages_to_train.dropna()

X_train_complete = Ages_to_train.drop('Age', axis=1)

y_train_complete = Ages_to_train['Age']

X_missing = Ages_to_fill.drop(['Age', 'Survived'], axis=1)



# lab_enc = preprocessing.LabelEncoder()

# y_enc = lab_enc.fit_transform(y_train).ravel()



model = GridSearchCV(Ridge(normalize=True), {'alpha': np.logspace(-1, 1, 21)}, n_jobs=-1, cv=5)

model.fit(X_train, y_train)

# print(model.best_params_, model.best_score_)



missing_ages = model.predict(X_missing)

train_ds.loc[train_ds['Age'].isnull(),'Age'] = missing_ages

cols = test_ds.columns.tolist()

cols = cols[:-5] + cols[-4:-3] + cols[-5:-4] + cols[-3:]

test_ds = test_ds[cols]

print(test_ds.info())

print(test_ds.head(10))
Ages_to_fill_t = test_ds[test_ds['Age'].isnull()]

missing_ages_t = model.predict(Ages_to_fill_t.drop(['Age', 'PassengerId'], axis=1))

test_ds.loc[test_ds['Age'].isnull(),'Age'] = missing_ages_t

test_ds.loc[test_ds['Fare'].isnull(),'Fare'] = test_ds['Fare'].mean()

print(test_ds.info())

X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

models = []

model_names = []

model_scores = []



k_results = []

k_results.append(0.69)

for i in range(24):

    knn_model = KNN(n_neighbors=(i+1))

    k_results.append(cross_val_score(knn_model,X,y,cv=5).mean())

plt.plot(k_results)

plt.show()
knn_model = KNN(n_neighbors=3)

print(cross_val_score(knn_model,X,y,cv=5).mean())

knn_model.fit(X,y)

print(knn_model.predict(test_ds.drop('PassengerId', axis=1).head()))





models.append(knn_model)

model_names.append('KNN(k=3)')

model_scores.append(cross_val_score(knn_model,X,y,cv=5).mean())
logreg_model = LogisticRegression()

print(cross_val_score(logreg_model,X,y,cv=5).mean())



models.append(logreg_model)

model_names.append('logistic regression')

model_scores.append(cross_val_score(logreg_model,X,y,cv=5).mean())
SVC_model = SVC(gamma='scale')

print(cross_val_score(SVC_model,X,y,cv=5).mean())



models.append(SVC_model)

model_names.append('SVC')

model_scores.append(cross_val_score(SVC_model,X,y,cv=5).mean())
SGD_model = SGD()

print(cross_val_score(SGD_model,X,y,cv=5).mean())



models.append(SGD_model)

model_names.append('SGD')

model_scores.append(cross_val_score(SGD_model,X,y,cv=5).mean())
est_results = []

est_results.append(0.8)

for i in range(10):

    RF_model = RandomForestClassifier(n_estimators=((i+1)*50))

    est_results.append(cross_val_score(RF_model,X,y,cv=5).mean())

plt.plot(est_results)

plt.show()
RF_model = RandomForestClassifier(n_estimators=50)

print(cross_val_score(RF_model,X,y,cv=5).mean())



models.append(RF_model)

model_names.append('Random forest')

model_scores.append(cross_val_score(RF_model,X,y,cv=5).mean())
percept_model = Perceptron()

print(cross_val_score(percept_model,X,y,cv=5).mean())



models.append(percept_model)

model_names.append('Perceptron')

model_scores.append(cross_val_score(percept_model,X,y,cv=5).mean())
dt = DecisionTreeClassifier(max_depth=2)

est_results = []

est_results.append(0.8)

for i in range(20):

    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=((i+1)*10))

    est_results.append(cross_val_score(ada,X,y,cv=5).mean())

plt.plot(est_results)

plt.show()

ada_model = AdaBoostClassifier(n_estimators=50)

print(cross_val_score(ada_model,X,y,cv=5).mean())



models.append(ada_model)

model_names.append('Adaboost')

model_scores.append(cross_val_score(ada_model,X,y,cv=5).mean())
GB_model = GradientBoostingClassifier(n_estimators=200, max_depth=2)

print(cross_val_score(GB_model,X,y,cv=5).mean())



models.append(GB_model)

model_names.append('GradientBoost')

model_scores.append(cross_val_score(GB_model,X,y,cv=5).mean())
xgb_model = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9,                        

 subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1)

print(cross_val_score(xgb_model,X,y,cv=5).mean())



models.append(xgb_model)

model_names.append('XGboost')

model_scores.append(cross_val_score(xgb_model,X,y,cv=5).mean())
res = pd.DataFrame({'Model':model_names,'Score':model_scores})

res.sort_values(by='Score', ascending=False)
def tune_and_eval(X_train, y_train, cv=5):

    gb_param = {"n_estimators": [10+(i*10) for i in range(31)], "max_depth": [2,3],}

    ada_param = {"n_estimators": [10+(i*5) for i in range(31)], "base_estimator": [DecisionTreeClassifier(max_depth=2),DecisionTreeClassifier(max_depth=3)],}

    logreg_param = {'C': np.logspace(-3, 3, 21)}

    knn_param = {'n_neighbors': [i+3 for i in range(13)]}

    

    

    inner_models = []

    inner_models.append(GridSearchCV(GradientBoostingClassifier(),gb_param, n_jobs=-1, cv=cv))

    inner_models.append(GridSearchCV(RandomForestClassifier(),gb_param, n_jobs=-1, cv=cv))

    inner_models.append(GridSearchCV(AdaBoostClassifier(),ada_param, n_jobs=-1, cv=cv))

    inner_models.append(GridSearchCV(LogisticRegression(), logreg_param, n_jobs=-1, cv=cv))

    inner_models.append(GridSearchCV(SVC(), logreg_param, n_jobs=-1, cv=cv))

    inner_models.append(GridSearchCV(KNN(), knn_param, n_jobs=-1, cv=cv))

    

    names = ['Gradient boosting','Random Forest','Adaboosting', 'logistic regression', 'SVC', 'KNN', 'XGBoost']

    scores = []

    params = []

    for idx, mod in enumerate(inner_models):

        print("tuning "+str(names[idx])) # for time feeling. can be commented out.

        mod.fit(X_train,y_train)

        scores.append(mod.best_score_)

        params.append(mod.best_params_)

    

    xgb_model = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1)

    scores.append(cross_val_score(xgb_model,X,y,cv=5).mean())

    params.append('Untuned')

    res = pd.DataFrame({'Model':names,'Score':scores, 'Params':params})

    res = res.sort_values(by='Score', ascending=False)

    return res    
X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

print(tune_and_eval(X,y, cv=5))
for dataset in [train_ds, test_ds]:

    for i in range(16):

        dataset.loc[(dataset['Age'] > (i*5)) & (dataset['Age'] <= ((i+1)*5)), 'Age'] = i

    dataset['Age'] = dataset['Age'].astype(int)

print(pd.qcut(train_ds['Fare'],5))



for dataset in [train_ds, test_ds]:

    dataset.loc[ dataset['Fare'] <= 7.85, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.85) & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3

    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4

    dataset['Fare'] = dataset['Fare'].astype(int)
X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

print(tune_and_eval(X,y, cv=5))
train_ds['Fam_size'] = train_ds['Parch'] + train_ds['SibSp'] + 1



#adjusting the test_set

test_ds['Fam_size'] = test_ds['Parch'] + test_ds['SibSp'] + 1
X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

print(tune_and_eval(X,y, cv=5))
main_features = train_ds[['Survived','Sex', 'Pclass','Age','Fare','SibSp','Parch','Fam_size','Embarked_S','Embarked_C','Embarked_Q','Title_Special','Title_Mr','Title_Miss','Title_Mrs','Title_Master','Cabin_X']]

plt.figure(figsize=(14,12))

colormap = plt.cm.RdBu

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(main_features.astype(float).corr(), square=True, cmap=colormap, annot=True)
for dataset in [train_ds, test_ds]:

    for i in range(16):

        dataset.loc[(dataset['Age'] >= (i*16)) & (dataset['Age'] < ((i+1)*16)), 'Age_group'] = i

    dataset.loc[(dataset['Age']>=80), 'Age_group'] = 5           

    dataset['Age'] = dataset['Age'].astype(int)



for dataset in [train_ds, test_ds]:

    dataset.loc[:,'Alone'] = 0

    dataset.loc[(dataset['Fam_size'] == 1), 'Alone'] = 1



main_features = train_ds[['Survived','Sex', 'Pclass','Age','Fare','SibSp','Parch','Fam_size','Embarked_S','Embarked_C','Embarked_Q','Cabin_X','Alone','Age_group']]



    

plt.figure(figsize=(14,12))

colormap = plt.cm.RdBu

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(main_features.astype(float).corr(), square=True, cmap=colormap, annot=True)
for dataset in [train_ds, test_ds]:

    dataset.loc[:,'Infant'] = 0

    dataset.loc[(dataset['Age'] <= 4), 'Infant'] = 1

    dataset.loc[:,'Kid'] = 0

    dataset.loc[((dataset['Age'] > 4) & (dataset['Age'] <= 8)), 'Kid'] = 1    

    dataset.loc[:,'cared_and_careless'] = 0

    dataset.loc[(dataset['Parch'] <= 1)&(dataset['Title_Mrs'] == 1), 'cared_and_careless'] = 1 



features = train_ds[['Survived','Infant','Kid', 'cared_and_careless']]  

plt.figure(figsize=(14,12))

colormap = plt.cm.RdBu

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(features.astype(float).corr(), square=True, cmap=colormap, annot=True)
#it seems only cared_and_careless should help, but let's check:

X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

print(tune_and_eval(X, y, cv=5).iloc[:,0:2])



train_ds = train_ds.drop(['Infant','Kid'], axis=1)

test_ds = test_ds.drop(['Infant','Kid'], axis=1)

X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

print(tune_and_eval(X,y, cv=5).iloc[:,0:2])
#lets drop some features that only distract (Embarked_q is not needed as it is the same of 00 on the other results)

test_ds = test_ds.drop(['Age','SibSp','Parch','Embarked_Q'], axis=1)

train_ds = train_ds.drop(['Age','SibSp','Parch','Embarked_Q'], axis=1)
X = train_ds.drop('Survived', axis=1)

y = train_ds['Survived']

res = tune_and_eval(X,y, cv=5)

print(res)

print(res.iloc[6])
plt.figure(figsize=(14,12))

colormap = plt.cm.RdBu

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_ds.astype(float).corr(), square=True, cmap=colormap, annot=True)
#Playground:



experiments = []

res = []

experiments.append(train_ds.drop('Fam_size', axis=1))

experiments.append(train_ds.drop('Title_Special', axis=1))

experiments.append(train_ds.drop(['Title_Special','Title_Master'], axis=1))

experiments.append(train_ds.drop('Age_group', axis=1))

experiments.append(train_ds.drop('Cabin_G', axis=1))

experiments.append(train_ds.drop(['Cabin_T','Cabin_F','Cabin_A','Cabin_G'], axis=1))



experiments.append(train_ds.drop(['Title_Special','Age_group'], axis=1))

experiments.append(train_ds.drop(['Cabin_G','Age_group'], axis=1))

experiments.append(train_ds.drop(['Title_Special','Cabin_G'], axis=1))

experiments.append(train_ds.drop(['Fam_size','Age_group'], axis=1))

experiments.append(train_ds.drop(['Fam_size','Cabin_G'], axis=1))

experiments.append(train_ds.drop(['Fam_size','Title_Special'], axis=1))





experiments.append(train_ds.drop(['Title_Special','Age_group', 'Cabin_G'], axis=1))

experiments.append(train_ds.drop(['Fam_size','Age_group', 'Cabin_G'], axis=1))

experiments.append(train_ds.drop(['Title_Special','Fam_size', 'Cabin_G'], axis=1))

experiments.append(train_ds.drop(['Fam_size','Title_Special','Age_group'], axis=1))



experiments.append(train_ds.drop(['Fam_size','Title_Special','Age_group','Cabin_G'], axis=1))

experiments.append(train_ds.drop(['Fam_size','Title_Special','Age_group','Cabin_G','Title_Master','Cabin_T','Cabin_F','Cabin_A',], axis=1))





for exp in experiments:

    X_exp = exp.drop('Survived', axis=1)

    y_exp = exp['Survived']

    res.append(tune_and_eval(X_exp, y_exp, cv=5))
for i in range(len(res)):

    print("experiment "+str(i))

    print(res[i].iloc[:,1].max(), res[i].iloc[:,1].mean())
print(res[7])
# xgb_param = {"n_estimators": [1500+(i*250) for i in range(5)], "max_depth": [4,5,6], "min_child_weight":[1,2],"gamma":[0,0.5,0.9,1.5],"subsample":[0.8,1],"colsample_bytree":[0.8,1],"objective":['binary:logistic'], "nthread": [-1],}

# tuned_xgb = GridSearchCV(xgb.XGBClassifier(),xgb_param, n_jobs=-1, cv=5)



# X_7 = experiments[7].drop('Survived', axis=1)

# y_7 = experiments[7]['Survived']

# tuned_xgb.fit(X_7, y_7)

# print(tuned_xgb.best_score_)

# print(tuned_xgb.best_params_)

# names = ['Gradient boosting','Random Forest','Adaboosting', 'logistic regression', 'SVC', 'KNN', 'XGBoost']

# ens_models = []

# ens_models.append(GradientBoostingClassifier(n_estimators=30, max_depth=3))

# ens_models.append(RandomForestClassifier(n_estimators=160, max_depth=3))

# ens_models.append(AdaBoostClassifier(n_estimators=10, base_estimator=DecisionTreeClassifier(max_depth=2)))

# ens_models.append(LogisticRegression(C=0.501187))

# ens_models.append(SVC(C=1,probability=True))

# ens_models.append(KNN(n_neighbors=13))

# ens_models.append(xgb.XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1, probability=True))



# est=[]

# for idx, mod in enumerate(ens_models):

#     est.append((names[idx],mod))



# ens_model = VotingClassifier(est, voting='soft', n_jobs=-1)

# print(cross_val_score(ens_model,X_7,y_7,cv=5).mean()) # this is mean as opposing to the former max scores!

# print(X_7.info())

# print(test_ds.info())
# ens_model.fit(X_7,y_7)



# # test_ds = test_ds.drop(['Cabin_G','Age_group'], axis=1)

# ids = test_ds['PassengerId']

# real_test = test_ds.drop('PassengerId', axis=1)

# real_pred = ens_model.predict(real_test)



# submission = pd.DataFrame({'PassengerId':ids, 'Survived':real_pred})

# print(submission)



# submission.to_csv('./submission.csv', index=False) # this one has done 0.789

svc = SVC()

svc.fit(X_7,y_7)

svc_submission = svc.predict(real_test)

(pd.DataFrame({'PassengerId':ids, 'Survived':svc_submission})).to_csv('./SVC_submission.csv', index=False)