# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

from matplotlib import style

style.use('fivethirtyeight')

%matplotlib inline



# Load the dataset

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



# Print the dataset information

#train_data.info()

#train_data.head()
### Select features that only make sense for survival prediction, SibSp and Parch are combined into Family

train = train_data[['PassengerId','Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

test = test_data[['PassengerId','Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

target = train_data['Survived']
### Create Title column from Name information, Create also Special_Title column

import re

whole = pd.concat([train, test])#.reset_index()

whole['Title'] = whole.Name.map(lambda x: re.search('\w+\.',x).group(0))

whole['Offi'] = (whole.Title.str.contains(r'Major\.|Col\.|Capt\.')).astype(int)

whole['High'] = (whole.Title.str.contains(r'Don\.|Jonkheer\.|Sir\.|Countess\.|Dona\.')).astype(int)

whole['Dr'] = (whole.Title.str.contains('Dr.')).astype(int)
### Extract Surname and Maiden name

whole['Surname'] = whole.Name.map(lambda x: re.search('(.+)(?=\,)',x).group(1))

def extract(row):

    if row.Sex=='male': return np.nan

    st = re.search('(\w+)(?=\))',row.Name.replace('"',''))

    if st is None: return np.nan

    else: return st.group(0).replace('"', '')

whole['Maiden'] = whole.apply(lambda x: extract(x),axis=1)
whole.columns=['Id','Name','Pclass','Sex','Age','SibSp','Parch','Fare','Emb','Title','Offi','High','Dr','Surname','Maiden']
whole = whole.set_index('Id')
### Assign family id to those who travel together

fid=1

whole['Fid']= np.nan

for idx in whole.index:

    row = whole.loc[idx]

    ## Consider only those not assigned and having family on board

    if (np.isnan(row.Fid))&(row.SibSp+row.Parch>0):        

        ## Those who have the same surname, emb, class, and fare (some variation accepted)

        temp = whole[(whole.Surname==row.Surname)#|(whole.Surname==row.Maiden)|(whole.Maiden==row.Maiden))

                    &((whole.Fare.round(1)==row.Fare.round(1))|(whole.Fare.round(1)==(row.Fare/2).round(1))

                      |(whole.Fare.round(1)==(row.Fare*2).round(1))|(whole.Fare.round(1)==(row.Fare/3).round(1))

                      |(whole.Fare.round(1)==(row.Fare*3).round(1)))

                    &(whole.Emb==row.Emb)&(whole.Pclass==row.Pclass)&(whole.SibSp+whole.Parch>0)]

        ## if sibsp, parch numbers match with family size for all family members, then skip below

        ## if sibsp, parch numbers each match themselves, then it is also fine, skip below

        if (((temp.SibSp+temp.Parch+1)==len(temp)).mean()!= 1) & ~(((temp.SibSp.sum())%2==0)&((temp.Parch.sum())%2==0)):

            fname = row.Surname

            maiden_coming = temp[(temp.Maiden.notnull())&(temp.Surname==fname)]

            for r in maiden_coming.itertuples():

                ext1 = whole[(whole.Surname!=r.Surname)

                             &((whole.Surname==r.Maiden)|(whole.Maiden==r.Maiden))#|(whole.Maiden==t.Surname))

                             &((whole.Fare.round(1)==r.Fare.round(1))|(whole.Fare.round(1)==(r.Fare/2).round(1))

                               |(whole.Fare.round(1)==(r.Fare*2).round(1))|(whole.Fare.round(1)==(r.Fare/3).round(1))

                               |(whole.Fare.round(1)==(r.Fare*3).round(1)))

                             &(whole.Emb==r.Emb)&(whole.Pclass==r.Pclass)&(whole.SibSp+whole.Parch!=0)]

                test = pd.concat([temp, ext1])

                ## if the new group of family seems complete, then stop

                if (((test.SibSp+test.Parch+1)==len(test)).mean()== 1) | (((test.SibSp.sum())%2==0)&((test.Parch.sum())%2==0)):

                    temp = test

                    #print('EXT1 worked, Fid, surname =',fid, row.Surname)

                ## otherwise, extend search to those have the surname as maiden name

                else:

                    ext2 = whole[(whole.Maiden==row.Surname) #|(whole.Maiden==r.Maiden))#|(whole.Maiden==t.Surname))

                             &((whole.Fare.round(1)==row.Fare.round(1))|(whole.Fare.round(1)==(row.Fare/2).round(1))

                               |(whole.Fare.round(1)==(row.Fare*2).round(1))|(whole.Fare.round(1)==(row.Fare/3).round(1))

                               |(whole.Fare.round(1)==(row.Fare*3).round(1)))

                             &(whole.Emb==row.Emb)&(whole.Pclass==row.Pclass)&(whole.SibSp+whole.Parch!=0)]

                    test = pd.concat([temp, ext2])

                    if (((test.SibSp+test.Parch+1)==len(test)).mean()== 1) | (((test.SibSp.sum())%2==0)&((test.Parch.sum())%2==0)):

                        temp = test

                        #print('EXT2 worked, Fid, surname =',fid, row.Surname)

                    ## in case it is still incomplete, then we need to scrutinize one by one

                    else: print('Need hand work, Fid, surname =',fid,row.Surname)



        whole.set_value(temp.index, 'Fid', fid)

        fid+=1
### Some hard coding for family gathering (these are usually due to fare difference, or large family links 3-4 different surnames together)

whole.set_value(268,'Fid',73) #display(whole[whole.Name.str.contains(r'Strom|Persson|Lindell')])

whole.set_value([581,1133],'Fid',76) #display(whole[whole.Name.str.contains(r'Jacobsohn|Christy')])

whole.set_value(881,'Fid',88) #display(whole[whole.Name.str.contains(r'Parrish')])

whole.set_value(1296,'Fid',107) #display(whole[whole.Name.str.contains(r'Frauenthal|Heinsheimer')])

whole.set_value([530,775,893,944],'Fid',120) #display(whole[(whole.Surname=='Richards')|(whole.Surname=='Hocking')|(whole.Maiden=='Needs')])

whole.set_value(665,'Fid',137) #display(whole[whole.Name.str.contains(r'Hirvonen|Lindqvist')])

whole.set_value(600,'Fid',149) #display(whole[whole.Name.str.contains(r'Duff Gordon')])

whole.set_value(1025,'Fid',186) #display(whole[whole.Surname=='Thomas']) #[1008,1025,1224]

whole.set_value(705,'Fid',192) #display(whole[whole.Name.str.contains('Hansen')]) #(624,'Fid',172)

whole.set_value(176,'Fid',211) #display(whole[whole.Name.str.contains('Klasen')])

whole.set_value(1197,'Fid',144) #display(whole[whole.Name.str.contains('Crosby')])

whole.set_value([70,1268],'Fid',68) #display(whole[whole.Name.str.contains('Kink')])

whole.set_value([672,984],'Fid',188) #display(whole[whole.Name.str.contains(r'Davidson|Hays')])

whole.set_value(69,'Fid',215) #display(whole[whole.Name.str.contains(r'Andersson')&(whole.Fid!=8)]) #[147,1212]

whole.set_value(913,'Fid',72) #display(whole[whole.Surname=='Olsen'])

print('Update done')
### Exceptions when family size==1

whole.set_value(478,'Fid',1) # Braund, couple with different price

whole.set_value(193,'Fid',174) # Andersen-Jensen with Jensen

whole.set_value(540,'Fid',155) # Frolicher with Frolicher-Stehli

whole.set_value([969,1248],'Fid',152) # Lamson female siblings



### Exceptions when family size>=2 

whole.set_value([105,393],'Fid',36) # Gustafsson, Backstrom

whole.set_value(137,'Fid',83) # Beckwith, Monypeny (Newsom)

whole.set_value(418,'Fid',102) # Lahtinen, Silven

whole.set_value([923,1211],'Fid',135) # Renouf, Jefferys

whole.set_value(386,'Fid',147) # Davies Mr. Charles Henry SibSp, Parch =0 but seems like an error

print('Update done')
### Add a new feature of family size

whole['Family'] = 1

for idx in whole.index:

    row = whole.loc[idx]

    temp = whole[whole.Fid==row.Fid]

    size = len(temp)

    if size==1: 

        whole.set_value(idx,'Fid',np.nan)

    elif size>=2: whole.set_value(idx,'Family',size)
### Impute Age based on their title by using sample age in the same group

whole_orig = whole.copy()

lookup  = whole[whole.Age.notnull()].groupby('Title')

np.random.seed(42)

def impute(index):

    imp = lookup.get_group(whole.loc[index].Title).sample(1).Age.iloc[0]

    whole.set_value(idx, 'Age', imp)

    

for idx in whole[whole.Age.isnull()].index:

    row = whole.loc[idx]

    fid = row.Fid

    

    ## in case he/she has no siblings/spouses

    if np.isnan(fid): 

        impute(idx)



    ## in case he/she has siblings/spouses

    else:

        temp = whole[(whole.Fid==fid)&(whole.index!=idx)] ## People in the same family but not himself/herself

        ## in case all four family members have SibSp=1, Parch=2 - special case

        if (row.SibSp==1)&(row.Parch==2):

            if (row.Title=='Master.')|(row.Title=='Miss.'): 

                age = temp[(temp.Title=='Master.')|(temp.Title=='Miss.')].Age.mean() + np.random.randint(-2,2)

                if age>=0: whole.set_value(idx, 'Age', age)

                else: whole.set_value(idx, 'Age', np.random.randint(1,15,1)[0])

            else: 

                age = temp[(temp.Title!='Master.')&(temp.Title!='Miss.')].Age.mean() + np.random.randint(-2,2)

                if age>=0: whole.set_value(idx, 'Age', age)

                else: whole.set_value(idx, 'Age', np.random.randint(30,45,1)[0])

        ## otherwise, find a sibling or spouse to estimate age

        else:

            sibsp = temp[(temp.SibSp==row.SibSp)&(temp.Parch==row.Parch)]

            parch = temp[~temp.isin(sibsp)].dropna(how='all')

            if (len(parch)>2)&(len(sibsp)>2): print('Need to check, id =',idx)

            ## in case sibsp>0 and found the same sibsp number in family

            else: 

                age = sibsp.Age.mean() + np.random.randint(-2,2)

                if age>=0: whole.set_value(idx, 'Age', age)

                else: 

                    if (row.Title=='Master.')|((parch.Age>=40).sum()>0): 

                        whole.set_value(idx, 'Age', np.random.randint(1,15,1)[0])

                    elif (parch.Title.isin(['Master.']).sum()>0)|((parch.Age<20).sum()>0): 

                        whole.set_value(idx, 'Age', np.random.randint(30,45,1)[0]) 

                    else: 

                        whole.set_value(idx, 'Age', np.random.randint(20,45,1)[0])
### Age data original distribution

fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)

age = whole_orig.Age.round(0)

grouped = age.groupby(age).count()

plt.sca(axes[0])

plt.bar(grouped.index,grouped,color='grey')

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Age count without imputation')



### Imputed Age dataset (new values added shown in orange)

age_imp = whole.Age.round(0)

grouped_imp = age_imp.groupby(age_imp).count()

plt.sca(axes[1])

plt.bar(grouped_imp.index, grouped_imp, color='orange')

plt.bar(grouped.index, grouped, color='grey')

plt.xlabel('Age')

#plt.ylabel('Count')

plt.title('Age count with imputation')

plt.tight_layout()
### Imputation of Emb and Fare with mean value

whole[whole.Emb.isnull()]

whole[(whole.Pclass==1)&(whole.Emb=='S')].Fare.mean()

whole.set_value([62,830],'Emb','S')

meanfare = whole[(whole.Pclass==3)&(whole.Emb=='S')].Fare.mean()

whole.set_value(1044,'Fare',meanfare)

whole.isnull().sum()
### Handling of Fare==0 (there are even first class passenger with free ticket so assign mean value of each class)

z = whole[whole.Fare==0]

for idx in z.index:

    row = whole.loc[idx]

    if row.Fare==0:

        m = whole[(whole.Pclass==row.Pclass)&(whole.Emb==row.Emb)&(whole.Fare!=0)].Fare.mean()

        whole.set_value(idx,'Fare',m)
s = pd.concat([whole[:891].reset_index(),target],axis=1).set_index('Id')

x = pd.concat([s,whole[891:]])#.set_index('Id')
### Create new features related to family members' survival - whether family members (male, female, child) have survived or died

x['MP_Surv']=0

x['MP_Died']=0

x['FP_Surv']=0

x['FP_Died']=0

x['CP_Surv']=0

x['CP_Died']=0



for row in x.itertuples():

    if ~np.isnan(row.Fid):

        temp = x[(x.Fid==row.Fid)&(x.Name!=row.Name)]

        if len(temp)>=1:

            m = temp[(temp.Sex=='male')&(temp.Age>=20)]

            f = temp[(temp.Sex=='female')&(temp.Age>=20)]

            c = temp[(temp.Age<20)|(temp.Title=='Master.')]

            x.set_value(row.Index,'MP_Surv',len(m[m.Survived==1]))

            x.set_value(row.Index,'MP_Died',len(m[m.Survived==0]))

            x.set_value(row.Index,'FP_Surv',len(f[f.Survived==1]))

            x.set_value(row.Index,'FP_Died',len(f[f.Survived==0]))

            x.set_value(row.Index,'CP_Surv',len(c[c.Survived==1]))

            x.set_value(row.Index,'CP_Died',len(c[c.Survived==0]))
whole = x
### Sex column into numeric values

#features_long.loc[:,'Sex'] = LabelEncoder().fit_transform(features_long['Sex'])

whole['Sex_d'] = (whole.Sex=='male').astype(int)

whole.drop('Sex',axis=1, inplace=True)
### Pclass column into numeric values

whole = pd.concat([whole, pd.get_dummies(whole.Pclass, prefix='Pclass')], axis=1)

whole = whole.drop(['Pclass'], axis=1)
whole = whole.drop(['SibSp','Parch','Emb','Name','Surname','Maiden','Title','Fid'],axis=1)
### Final selection of features

whole.head()
### Split again train and test data sets

train_df = whole.drop(['Survived'],axis=1).iloc[:891]

test_df = whole.drop(['Survived'],axis=1).iloc[891:]
target = whole['Survived'][:891]
### Prediction optimization of Random Forest using GridSearchCV (You can increase n_splits value for more rigorous test)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.3, random_state=123)

cv = StratifiedKFold(n_splits=10, random_state=42)

grid_param = {'n_estimators': [60,80,100,120],

             'min_samples_split': [4,6,8],

             'min_samples_leaf': [2,3,4],

             'criterion': ['gini','entropy']}

clf_g = RandomForestClassifier()



grid_search = GridSearchCV(clf_g, grid_param, cv=cv)

grid_search.fit(X_train, y_train)

pred = grid_search.predict(X_test)

clf_rf = grid_search.best_estimator_

print("best parameters : {}".format(grid_search.best_estimator_))

print("score = {}".format(accuracy_score(pred, y_test)))
### K-Fold cross validation to check variance of results

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kfold = KFold(n_splits=10)

cross_val_score(clf_rf, train_df, target, cv=kfold, n_jobs=-1)
### Prediction optimization of Decision Tree using GridSearchCV

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.3, random_state=123)

cv = StratifiedKFold(n_splits=10, random_state=42)

grid_param = {'max_depth': range(2,8,1),

             'min_samples_leaf': range(2,6,1),

             'min_samples_split': range(3,7,1)}

clf_g = DecisionTreeClassifier(random_state=0)



grid_search = GridSearchCV(clf_g, grid_param, cv=cv)

grid_search.fit(X_train, y_train)

pred = grid_search.predict(X_test)

clf_dt = grid_search.best_estimator_

print("best parameters : {}".format(grid_search.best_estimator_))

print("score = {}".format(accuracy_score(pred, y_test)))
### Prediction optimization of Support Vector Machine using GridSearchCV 

from sklearn.svm import SVC



X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.3, random_state=123)

cv = StratifiedKFold(n_splits=10, random_state=42)

grid_param = {'C': [100000,120000,140000],

             'gamma': [1e-6,1e-5,1e-4]}

clf_g = SVC(random_state=0)



grid_search = GridSearchCV(clf_g, grid_param, cv=cv)

grid_search.fit(X_train, y_train)

pred = grid_search.predict(X_test)

clf_svc = grid_search.best_estimator_

print("best parameters : {}".format(grid_search.best_estimator_))

print("score = {}".format(accuracy_score(pred, y_test)))
### K-Fold cross validation to check variance of results

kfold = KFold(n_splits=10)

cross_val_score(clf_svc, train_df, target, cv=kfold, n_jobs=-1)
### Plot learning curve

from sklearn.model_selection import learning_curve

cv = StratifiedKFold(n_splits=10,shuffle=True)

training_sizes, training_scores, testing_scores = learning_curve(clf_svc, train_df, target, train_sizes=np.linspace(0.1, 1.0, 10), cv=cv)



### Get mean and std

training_scores_mean = np.mean(training_scores, axis=1)

training_scores_std = np.std(training_scores, axis=1)

testing_scores_mean = np.mean(testing_scores, axis=1)

testing_scores_std = np.std(testing_scores, axis=1)



plt.plot(training_sizes, training_scores_mean, 'o-', color='r', label='training_score')

plt.plot(training_sizes, testing_scores_mean, 'o-', color='g', label='testing_score')

plt.fill_between(training_sizes, training_scores_mean - training_scores_std, \

                     training_scores_mean + training_scores_std, color='r', alpha=0.2)

plt.fill_between(training_sizes, testing_scores_mean - testing_scores_std, \

                     testing_scores_mean + testing_scores_std, color='g', alpha=0.2)

 

### Plot aesthetic

plt.grid(True)

plt.ylim(-0.1, 1.1)

plt.ylabel("Curve Score")

plt.xlabel("Training Points")

plt.legend(bbox_to_anchor=(1.1, 1.1), loc='best')

plt.show()
pred = clf_svc.predict(test_df)
submission = pd.Series(pred, index=test_df.index)

submission = submission.reset_index()

submission.columns=['PassengerId','Survived']

submission.Survived = submission.Survived.astype(int)
submission.to_csv('titanic_jpark_v6_SVC.csv', index=False)