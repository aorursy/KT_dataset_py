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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
import pandas as pd

import numpy as np

from sklearn import preprocessing 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()

  

#Check for the missing values in the columns 

fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(train_data.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
#drop the columns without any relationship with the dataset

train_data = train_data.drop(columns = ['Cabin','Name','Ticket','PassengerId'])
# filling na values with mean for age, 

train_data['Age'].fillna((train_data['Age'].mean()), inplace=True)
#plotting Survival as function of Sex

sns.barplot(x='Sex', y='Survived', data=train_data)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Sex", fontsize=16)



plt.show()

train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#plotting Survival as function of Pclass

sns.barplot(x='Pclass', y='Survived', data=train_data)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass", fontsize=16)



plt.show()

train_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#both Sex and Pclass in the same plot

sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train_data)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass and Sex")

plt.show()

#here 0 is for female and 1 is for male
#the Parch column

train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#the SibSp column

train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#pairplots for all columns

sns.pairplot(data=train_data, hue="Survived")
#Create a swarmplot to detect patterns, where is the highest survival rate

sns.swarmplot(x = 'SibSp', y = 'Parch', hue = 'Survived', data = train_data, split = True, alpha=0.8)

plt.show()
#1st model before featuring

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



yf = train_data.Survived

base_features = ['Parch','SibSp','Age', 'Fare','Pclass']



Xf = train_data[base_features]



train_X, val_X, train_y, val_y = train_test_split(Xf, yf, random_state=1)

first_model = RandomForestRegressor(n_estimators=21, random_state=1).fit(train_X, train_y)
#Explore the relationship between SipSp and Parch in the predictions for a RF Model

inter  =  pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=['SibSp', 'Parch'])



pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=['SibSp', 'Parch'], plot_type='contour')

plt.show()
#New feature FamilySize added to join the columns SibSp(sibling/spouses) & Parch(parents/children)

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] 

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).agg('mean')
#check if person travelling alone is more likely to survive

train_data['IsAlone'] = 0

train_data.loc[train_data['FamilySize'] == 0, 'IsAlone'] = 1



train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#graphs for all columns including the newly made FamilySize & IsAlone

cols = ['Survived', 'Parch', 'SibSp', 'Embarked','IsAlone', 'FamilySize']



nr_rows = 2

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        

        i = r*nr_cols+c       

        ax = axs[r][c]

        sns.countplot(train_data[cols[i]], hue=train_data["Survived"], ax=ax)

        ax.set_title(cols[i], fontsize=14, fontweight='bold')

        ax.legend(title="survived", loc='upper center') 

        

plt.tight_layout()
#changing the fare into a continous feature

#checking for unique points

feat_name = 'Fare'

pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()


train_data[["Fare", "Survived"]].groupby(['Survived'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#as females have better survival rate hence grouping higher fare with females

train_data.groupby(['Sex','Survived'])[['Fare']].agg(['min','mean','max'])
#discreting the fare in 4 states

train_data.loc[ train_data['Fare'] <= 7.22, 'Fare'] = 1

train_data.loc[(train_data['Fare'] > 7.22) & (train_data['Fare'] <= 21.96), 'Fare'] = 2

train_data.loc[(train_data['Fare'] > 21.96) & (train_data['Fare'] <= 40.82), 'Fare'] = 3

train_data.loc[ train_data['Fare'] > 40.82, 'Fare'] = 4

train_data['Fare'] = train_data['Fare'].astype(int)

#plotting the data according to the 4 states

g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Fare', bins=20)

plt.show()
#plotting bar graphs for gender and fare

sns.barplot(x='Sex', y='Survived', hue='Fare', data=train_data)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Fare and Sex")

plt.show()
#changing the age into a continous feature

#checking for unique points

feat_name = 'Age'

pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()



#Exploring the relationship between Age and Pclass for a given model preductions

inter  =  pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=['Age', 'Pclass'])



pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=['Age', 'Pclass'], plot_type='contour')

plt.show()


g = sns.FacetGrid(train_data, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Age', kde=False, bins=4, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()
#grouping age into different classes

train_data.loc[ train_data['Age'] <= 16, 'Age'] = 1

train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 2

train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 64), 'Age'] = 3

train_data.loc[ train_data['Age'] > 64, 'Age'] = 4

train_data['Age'] = train_data['Age'].astype(int)

#plotting the ages 

sns.barplot(x='Pclass', y='Survived', hue='Age', data=train_data)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Age and Sex")

plt.show()
#a new feature Age*Class is added to join Age and Pclass

train_data['Age*Class'] = train_data.Age * train_data.Pclass

train_data[["Age*Class", "Survived"]].groupby(['Age*Class'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#a crosstab giving survival for Sex with Age*Class

pd.crosstab([train_data.Survived], [train_data.Sex,train_data['Age*Class']], margins=True).style.background_gradient(cmap='autumn_r')


#a crosstab giving survival for IsAlone with Sex

pd.crosstab([train_data.Survived], [train_data.Sex,train_data['IsAlone']], margins=True).style.background_gradient(cmap='autumn_r')
#a crosstab giving survival for Fare

pd.crosstab([train_data.Survived], [train_data.Fare], margins=True).style.background_gradient(cmap='autumn_r')

#new train_data head after adding new features

train_data.head()
#training using new features

y2 = train_data.Survived



base_features2 = ['Parch','SibSp','Age', 'Fare','Pclass','Age*Class','FamilySize','IsAlone']



X2 = train_data[base_features2]

train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state=1)

second_model = RandomForestRegressor(n_estimators=21, random_state=1).fit(train_X2, train_y2)

#plotting pdp interact for age and pclass

inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['Age', 'Pclass'])

pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['Age', 'Pclass'], plot_type='contour')

plt.show()
#plotting FamilySize with Pclass

inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['FamilySize', 'Pclass'])

pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['FamilySize', 'Pclass'], plot_type='contour')

plt.show()
#As Sex and Embarked are not numerical OneHotEncoderis done

# convert Sex values and Embarked values into dummis to use a numerical classifier 

dummies_Sex = pd.get_dummies(train_data.Sex)

dummies_Embarked = pd.get_dummies(train_data.Embarked)

#join the dummies to the final dataframe

train_ready = pd.concat([train_data, dummies_Sex,dummies_Embarked], axis=1)

train_ready.head()

#drop the respective Sex and Embarked columns

train_ready = train_ready.drop(columns = ['Sex','Embarked'])

#checking for the data types to be numeric

train_ready.info()
train_ready.head(11)
#dropping Age*Class column (entropy 2,14) then the FamilySize column (entropy 1,82)

train_ready = train_ready.drop(columns = ['Age*Class'])

train_ready = train_ready.drop(columns = ['FamilySize'])
#checking entropy

from scipy import stats

for name in train_ready:

    print(name, "column entropy :", round(stats.entropy(train_ready[name].value_counts(normalize=True), base=2),2))
train_ready.head(10)
#Drop unecessary columns

test = test.drop(columns = ['Cabin','Name','Ticket','PassengerId'])

#check the test dataframe

test.head()
#filling Non valid values with mean for age, 

test['Age'].fillna((test['Age'].mean()), inplace=True)

test['Fare'].fillna((test['Fare'].mean()), inplace=True)

#discreting fare to 4 states

test.loc[ test['Fare'] <= 7.22, 'Fare'] = 0

test.loc[(test['Fare'] > 7.22) & (test['Fare'] <= 21.96), 'Fare'] = 1

test.loc[(test['Fare'] > 21.96) & (test['Fare'] <= 40.82), 'Fare'] = 2

test.loc[ test['Fare'] > 40.82, 'Fare'] = 3

#joining SibSp and Parch into FamiySize

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

#discreting Age to 4 states

test.loc[ test['Age'] <= 16, 'Age'] = 1

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 2

test.loc[(test['Age'] > 32) & (test['Age'] <= 64), 'Age'] = 3

test.loc[ test['Age'] > 64, 'Age'] = 4

#adding new feature Age*Class

test['Age*Class'] = test.Age * test.Pclass

#checking the datatypes

test.info()
#as in the train dataset, build dummis in the sex and embarked columns

test_dummies_Sex = pd.get_dummies(test.Sex)

test_dummies_Embarked = pd.get_dummies(test.Embarked)

test_ready = pd.concat([test, test_dummies_Sex,test_dummies_Embarked], axis=1)

test_ready.head()

#drop these columns, we keep only numerical values

test_ready = test_ready.drop(columns = ['Sex','Embarked'])

#dropping Age*Class column then the FamilySize column as done in train dataset

test_ready = test_ready.drop(columns = ['Age*Class'])

test_ready = test_ready.drop(columns = ['FamilySize'])
test_ready.info()

test_ready.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



# Create arrays for the features and the response variable

y = train_ready['Survived'].values

X = train_ready.drop('Survived',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=21, stratify=y)



#Importing the auxiliar and preprocessing librarys 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate

from sklearn.metrics import accuracy_score



#Models

import warnings

warnings.filterwarnings("ignore")

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
clfs = []

seed = 3

clfs.append(("LogReg", Pipeline([("Scaler", StandardScaler()), ("LogReg", LogisticRegression())]))) 

            

clfs.append(("XGBClassifier",Pipeline([("Scaler", StandardScaler()),("XGB", XGBClassifier())]))) 



clfs.append(("KNN",Pipeline([("Scaler", StandardScaler()),("KNN", KNeighborsClassifier(n_neighbors=8))])))  

                                   

clfs.append(("DecisionTreeClassifier",Pipeline([("Scaler", StandardScaler()),("DecisionTrees", DecisionTreeClassifier())])))  

             

clfs.append(("RandomForestClassifier",Pipeline([("Scaler", StandardScaler()),("RandomForest", RandomForestClassifier())])))

                                     

clfs.append(("GradientBoostingClassifier",Pipeline([("Scaler", StandardScaler()),("GradientBoosting", GradientBoostingClassifier(n_estimators=100))]))) 

                            

clfs.append(("RidgeClassifier",Pipeline([("Scaler", StandardScaler()),("RidgeClassifier", RidgeClassifier())])))                                  



clfs.append(("BaggingRidgeClassifier", Pipeline([("Scaler", StandardScaler()),("BaggingClassifier", BaggingClassifier())])))

                                  

clfs.append(("ExtraTreesClassifier",Pipeline([("Scaler", StandardScaler()),("ExtraTrees", ExtraTreesClassifier())]))) 



#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'accuracy'

n_folds = 7

results, names  = [], [] 

for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train,cv= 5, scoring=scoring, n_jobs=-1)                                 

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())

    print(msg)

    

# boxplot algorithm comparison

fig = plt.figure(figsize=(15,6))

fig.suptitle('Classifier Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names, y=results)

ax.set_xticklabels(names)

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("Accuracy of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

plt.show()
#apply Scla to train in order to standardize data 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



scaler.fit(X)

scaled_features = scaler.transform(X)

train_sc = pd.DataFrame(scaled_features) # columns=df_train_ml.columns[1::])



#apply Scla to test csv (new file)  in order to standardize data 



X_csv_test = test_ready.values  #X_csv_test the new data that is going to be test 

scaler.fit(X_csv_test)

scaled_features_test = scaler.transform(X_csv_test)

test_sc = pd.DataFrame(scaled_features_test) # , columns=df_test_ml.columns)



scaled_features_test.shape
test.head()
#Building XGBClassifier

import xgboost as xgb

from xgboost import XGBClassifier



clf = xgb.XGBClassifier(n_estimators=250, random_state=4,bagging_fraction= 0.791787170136272, colsample_bytree= 0.7150126733821065,feature_fraction= 0.6929758008695552,gamma= 0.6716290491053838,learning_rate= 0.030240003246947006,max_depth= 2,min_child_samples= 5,num_leaves= 15,reg_alpha= 0.05822089056228967,reg_lambda= 0.14016232510869098,subsample= 0.9)



clf.fit(scaled_features, y)



y_pred_xgb= clf.predict(scaled_features_test)

print(y_pred_xgb)



#Upload the test file for XGB

result_xgb = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

result_xgb['Survived'] = y_pred_xgb

result_xgb.to_csv('Titanic_xgb.csv', index=False)

result_xgb.head()