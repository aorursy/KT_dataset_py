import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from matplotlib import pyplot as plt

from catboost import CatBoostClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn import metrics

from featexp import *



from fastai.imports import *

from fastai.structured import *



df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
nullAge_df = df[df['Age'].isnull()]

nullAge_df['Sex']=nullAge_df['Sex'].factorize()[0]
EDA_df = df[df['Age'].notna()]

EDA_df['Sex'] = EDA_df['Sex'].factorize()[0]

get_univariate_plots(data=EDA_df, target_col='Survived', features_list=['Age','Sex','Fare'])
get_univariate_plots(data=EDA_df, target_col='Sex', features_list=['Age','Survived','Fare','Pclass'])
EDA_df['Cabin'].unique()
X = EDA_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Age'],axis=1)

fill_na(X)

make_factorize(X)

y = EDA_df.Age

X_train, X_test, y_train, y_test = make_TTSplit(X,y)

#set_rf_samples(500)

reset_rf_samples()

rfr = RandomForestRegressor(n_jobs=-1,n_estimators=20,random_state=33,min_samples_leaf=5,max_features=0.5)

pred = rmse_train_predict_test(X_train, X_test, y_train, y_test, rfr)

new_EDA_df = X.copy()

new_EDA_df['Age'] = y
newAge = nullAge_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Age'],axis=1)

fill_na(newAge)

make_factorize(newAge)

age_pred = rfr.predict(newAge)

age_pred = np.around(age_pred)

newAge['Age'] = age_pred
df_concat = pd.concat([newAge, new_EDA_df])

X = df_concat.drop('Survived',axis=1)

y = df_concat.Survived

X_train, X_test, y_train, y_test = make_TTSplit(X,y)

rfc = RandomForestClassifier(n_jobs=-1,n_estimators=22,random_state=33,min_samples_leaf=5,max_features=0.5)

train_predict_test(X_train, X_test, y_train, y_test, rfc)

cat = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)

cat.fit(X_train,y_train,eval_set=(X_test,y_test))
df_concat.info()
test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Age'],axis=1)

test['Age'] = test_df['Age']

fill_na(test)

make_factorize(test)

#test.info()

pred = cat.predict(test)

pred = pred.astype(np.int)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':pred})

submission.to_csv('cat_fastai.csv',index=False)
submission
def make_TTSplit(X, y, testsize=0.2, stratify_choice=None):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=33, stratify=stratify_choice)

    return X_train, X_test, y_train, y_test



def fill_na(df, fillNum=-999):

    # https://stackoverflow.com/questions/36226083/how-to-find-which-columns-contain-any-nan-value-in-pandas-dataframe-python

    na_col = df.columns[df.isna().any()].tolist()

    for col in na_col:

        df[col] = df[col].fillna(fillNum)



def make_factorize(df):

    for column in df.columns[df.dtypes == 'object']:

        df[column] = df[column].factorize()[0]

    return df



def rmse_train_predict_test(X_train, X_test, y_train, y_test, Classifier):

    Classifier.fit(X_train, y_train)

    pred = Classifier.predict(X_test)

    #pred=np.around(pred)

    score = metrics.mean_squared_error(y_test, pred)

    print("RMSE:   %0.3f" % score)

    preds = np.stack([t.predict(X_test) for t in Classifier.estimators_])

    preds[:,0], np.mean(preds[:,0]), y_test

    plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(100)])

    return pred



def train_predict_test(X_train, X_test, y_train, y_test, Classifier):

    Classifier.fit(X_train, y_train)

    pred = Classifier.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)

    print("accuracy:   %0.3f" % score)

    preds = np.stack([t.predict(X_test) for t in Classifier.estimators_])

    preds[:,0], np.mean(preds[:,0]), y_test

    plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(30)])

    
(df['Pclass'].value_counts() / len(df)).plot.bar()
# Save this for 'Featexp' introduction

#df[['Survived', 'Pclass']].plot.scatter(x='Survived', y='Pclass')
df
"""

1. Make Pclass interval

2. Find mean of each interval

"""

Pclass_survived=[]

Pclass1 = df[df['Pclass'] == 1]

Pclass_survived.append(Pclass1['Survived'].mean())

Pclass2 = df[df['Pclass'] == 2]

Pclass_survived.append(Pclass2['Survived'].mean())

Pclass3 = df[df['Pclass'] == 3]

Pclass_survived.append(Pclass3['Survived'].mean())

Pclass_survived_columns = [1,2,3]

#Pclass_tuples_survived = list(zip(Pclass_survived, Pclass_survived_columns))

# Create DataFrame from multiple lists

# https://cmdlinetips.com/2018/01/how-to-create-pandas-dataframe-from-multiple-lists/

fig, ax = plt.subplots()

plt.scatter(Pclass_survived_columns, y=Pclass_survived)

plt.xticks(Pclass_survived_columns)

# Bar plot of Survived by Pclass

fig, ax = plt.subplots()

plt.bar(Pclass_survived_columns,height=Pclass_survived)

plt.xticks(Pclass_survived_columns)

# Make of Survived by Pclass & Sex

Pclass_survived_f=[]

Pclass_survived_f.append(Pclass1[Pclass1['Sex'] == 'female'].Survived.mean())

Pclass_survived_f.append(Pclass2[Pclass2['Sex'] == 'female'].Survived.mean())

Pclass_survived_f.append(Pclass3[Pclass3['Sex'] == 'female'].Survived.mean())



Pclass_survived_m=[]

Pclass_survived_m.append(Pclass1[Pclass1['Sex'] == 'male'].Survived.mean())

Pclass_survived_m.append(Pclass2[Pclass2['Sex'] == 'male'].Survived.mean())

Pclass_survived_m.append(Pclass3[Pclass3['Sex'] == 'male'].Survived.mean())



Pclass_survived_f_columns=['Class 1 Female','Class 2 Female','Class 3 Female']

Pclass_survived_m_columns=['Class 1 Male','Class 2 Male','Class 3 Male']

#Pclass_tuples_survived_sex = list(zip(Pclass_survived_sex, Pclass_survived_sex_columns))



# Bar plot Survived by Pclass & Sex

ax2 = plt.subplot(111)

ax2.barh(Pclass_survived_f_columns, Pclass_survived_f,color='g',align='center')

ax2.barh(Pclass_survived_m_columns, Pclass_survived_m,color='b',align='center')

ax2.set_ylabel('Class & Sex')

ax2.set_xlabel('Survived Ratio')

ax2.set_title("Survival Rate by PClass & Sex")



# Set Legends



# From matplotlib multiple bars - Stack Overflow

# https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars



# Siblings survival mean

#ax3 = plt.subplot(222)

#SibSp = [0,1,2,3,4,5,8]

#ax3.bar(x=SibSp,height=df['SibSp'].value_counts() / len(df),align='center')

#ax3.set_xticks(df['SibSp'].value_counts(),SibSp)

#ax3.set_ylim([0, 1])
#df['Parch'].value_counts() / len(df)
#Bin_Fare = pd.cut(x=df['Fare'],bins=9)

#Bin_Fare
#df['Embarked']
#df.groupby('SibSp').aggregate(count)
#df.groupby('SibSp').count()
def make_factorize(df):

    for column in df.columns[df.dtypes == 'object']:

        df[column] = df[column].factorize()[0]

    return df

def fill_na(df, fillNum=-999):

    # https://stackoverflow.com/questions/36226083/how-to-find-which-columns-contain-any-nan-value-in-pandas-dataframe-python

    na_col = df.columns[df.isna().any()].tolist()

    for col in na_col:

        df[col] = df[col].fillna(fillNum)

def train_predict_test(X_train, X_test, y_train, y_test, Classifier):

    Classifier.fit(X_train, y_train)

    pred = Classifier.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)

    print("accuracy:   %0.3f" % score)

def printImportances(Classifier):

    # Feature importances graph from

    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    importances = Classifier.feature_importances_

    std = np.std([Classifier.feature_importances_ for tree in Classifier.estimators_],axis=0)

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")

    for f in range(X.shape[1]):

        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")

    plt.xticks(range(X.shape[1]), indices)

    plt.xlim([-1, X.shape[1]])

    plt.show()

    



# Version 1: Random Forest + Drop(Ticket)    



#rfc = RandomForestClassifier(n_estimators=200,random_state=33,n_jobs=-1,oob_score=True)

new_df = df.drop(['PassengerId', 'Name','Ticket'], axis=1)





#fill_na(new_df)

#make_factorize(new_df)

#X = new_df[new_df.columns[1:10]]

#y = new_df.Survived

#X_train, X_test, y_train, y_test = make_TTSplit(X, y, 0.25)

#train_predict_test(X_train, X_test, y_train, y_test,rfc)

#Pipeline

# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
# Version 2: Random Forest + Drop(Ticket) + Binned(Fare)



fareLabel = [0,1,2,3,4,5,6,7,8]

new_df2 = new_df.copy()

new_df2['Fare'] = pd.cut(new_df2['Fare'],bins=9,labels=fareLabel)

new_df2['Fare']=new_df2['Fare'].astype('int')

fill_na(new_df2)

make_factorize(new_df2)

X = new_df2[new_df2.columns[1:10]]

y = new_df2.Survived

X_train, X_test, y_train, y_test = make_TTSplit(X, y, 0.25)

train_predict_test(X_train, X_test, y_train, y_test,rfc)
stats = get_trend_stats(data=new_df2, target_col='Survived',data_test=pd.concat([X_test,y_test],axis=1))

stats
# Version 3: Random Forest + Drop(Ticket) + isAlone

new_df3 = new_df.copy()

new_df3.loc[(new_df3['SibSp'] < 1) & (new_df3['Parch'] < 1), 'isAlone'] =1

fill_na(new_df3)

make_factorize(new_df3)

X = new_df3[new_df3.columns[1:11]]

y = new_df3.Survived

X_train, X_test, y_train, y_test = make_TTSplit(X, y, 0.25)

#train_predict_test(X_train, X_test, y_train, y_test,rfc)

#rfc.fit(X_train,y_train)
# Version 4 (from #3): CatBoost + Drop(Ticket) + isAlone (Best)



model = CatBoostClassifier(eval_metric='F1',use_best_model=True,random_seed=42)

model.fit(X_train,y_train,eval_set=(X_test,y_test))
# Feature Engineering





# Use FeatEXP

# Use XGBoost
stats = get_trend_stats(data=new_df3, target_col='Survived',data_test=pd.concat([X_test,y_test],axis=1))

stats
get_univariate_plots(data=new_df3, target_col='Survived',data_test=pd.concat([X_test,y_test],axis=1), features_list=['Age'])
get_univariate_plots(data=new_df3, target_col='Survived',data_test=pd.concat([X_test,y_test],axis=1), features_list=['Age'])
# Version 5: CatBoost + Drop(Ticket, Cabin,SibSp,Parch) + isAlone (Worst than version #4)

#new_df5 = new_df.copy()

#new_df5.loc[(new_df5['SibSp'] < 1) & (new_df5['Parch'] < 1), 'isAlone'] =1

#new_df5=new_df5.drop(['Cabin','SibSp','Parch'], axis=1)

#fill_na(new_df5)

#make_factorize(new_df5)

#X = new_df5[new_df5.columns[1:11]]

#y = new_df5.Survived

#X_train, X_test, y_train, y_test = make_TTSplit(X, y, 0.25)

#train_predict_test(X_train, X_test, y_train, y_test,rfc)

#model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)

#model.fit(X_train,y_train,eval_set=(X_test,y_test))
# Version 6: CatBoost + Drop(Ticket, Cabin) + isAlone

new_df6 = new_df.copy()

new_df6.loc[(new_df6['SibSp'] < 1) & (new_df6['Parch'] < 1), 'isAlone'] =1

new_df6=new_df6.drop(['Cabin'], axis=1)

fill_na(new_df6)

make_factorize(new_df6)

X = new_df6[new_df6.columns[1:11]]

y = new_df6.Survived

X_train, X_test, y_train, y_test = make_TTSplit(X, y, 0.25)

train_predict_test(X_train, X_test, y_train, y_test,rfc)



model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)

model.fit(X_train,y_train,eval_set=(X_test,y_test))
# Version 7: CatBoost + Drop(Ticket) + isAlone + Binned(Fare)

rfc = RandomForestClassifier(n_estimators=50,random_state=33,n_jobs=-1,oob_score=False)

new_df7 = new_df.copy()

new_df7.loc[(new_df7['SibSp'] < 1) & (new_df7['Parch'] < 1), 'isAlone'] =1

fareLabel = [0,1,2,3,4,5,6,7,8]

new_df7['Fare'] = pd.cut(new_df7['Fare'],bins=9,labels=fareLabel)

new_df7['Fare']=new_df7['Fare'].astype('int')

fill_na(new_df7)

make_factorize(new_df7)

X = new_df3[new_df7.columns[1:11]]

y = new_df3.Survived

X_train, X_test, y_train, y_test = make_TTSplit(X, y, 0.15)



#train_predict_test(X_train, X_test, y_train, y_test,rfc)

#model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)

#model.fit(X_train,y_train,eval_set=(X_test,y_test))
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_test), y_test),

                m.score(X_train, y_train), m.score(X_test, y_test)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)



#set_rf_samples(110)

reset_rf_samples()

%time rfc.fit(X_train,y_train)

print_score(rfc)





preds = np.stack([t.predict(X_test) for t in rfc.estimators_])

preds[:,0], np.mean(preds[:,0]), y_test

plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(50)]);
def dectree_max_depth(tree):

    children_left = tree.children_left

    children_right = tree.children_right



    def walk(node_id):

        if (children_left[node_id] != children_right[node_id]):

            left_max = 1 + walk(children_left[node_id])

            right_max = 1 + walk(children_right[node_id])

            return max(left_max, right_max)

        else: # leaf

            return 1



    root_node_id = 0

    return walk(root_node_id)



rfc = RandomForestClassifier(n_estimators=50,random_state=33,n_jobs=-1,oob_score=True)

rfc.fit(X_train,y_train)

print_score(rfc)



t=rfc.estimators_[0].tree_

dectree_max_depth(t)

preds = np.stack([t.predict(X_test) for t in rfc.estimators_])

preds[:,0], np.mean(preds[:,0]), y_test

plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(50)]);
rfc = RandomForestClassifier(n_estimators=50,random_state=33,min_samples_leaf=5,n_jobs=-1,oob_score=True)

rfc.fit(X_train,y_train)

print_score(rfc)



t=rfc.estimators_[0].tree_

dectree_max_depth(t)

preds = np.stack([t.predict(X_test) for t in rfc.estimators_])

preds[:,0], np.mean(preds[:,0]), y_test

plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(50)]);
rfc = RandomForestClassifier(n_estimators=50,random_state=33,min_samples_leaf=4,n_jobs=-1,oob_score=True)

rfc.fit(X_train,y_train)

print_score(rfc)



t=rfc.estimators_[0].tree_

dectree_max_depth(t)

preds = np.stack([t.predict(X_test) for t in rfc.estimators_])

preds[:,0], np.mean(preds[:,0]), y_test

plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(50)]);
reset_rf_samples()

rfc = RandomForestClassifier(n_estimators=50,random_state=33,min_samples_leaf=4,max_features=0.5,n_jobs=-1,oob_score=True)

rfc.fit(X_train,y_train)

print_score(rfc)



t=rfc.estimators_[0].tree_

dectree_max_depth(t)

preds = np.stack([t.predict(X_test) for t in rfc.estimators_])

preds[:,0], np.mean(preds[:,0]), y_test

plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(50)]);
test_df = pd.read_csv('../input/test.csv')



test_df.loc[(test_df['SibSp'] < 1) & (test_df['Parch'] < 1), 'isAlone'] =1

#fareLabel = [0,1,2,3,4,5,6,7,8]

#test_df['Fare'] = pd.cut(test_df['Fare'],bins=9,labels=fareLabel)

#test_df['Fare']=test_df['Fare'].astype('int')

X = test_df.drop(['PassengerId', 'Name','Ticket'], axis=1)

fill_na(X)

make_factorize(X)



pred = rfc.predict(X)

pred = pred.astype(np.int)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':pred})

submission.to_csv('rfc_fastai.csv',index=False)
# ToDo

# Plot age then find optimal value for missing age values by ['Survived', 'Sex','Pclass', 'SibSp', 'Parch', 'isAlone', 'Fare','Name']