import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



import os

#print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

#drop cabin, Name and Ticket data that are not neccesary to train the model

train.head()
#Check for the missing values in the columns 

fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(train.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
#I drop those columns

train = train.drop(columns = ['Cabin','Name','Ticket','PassengerId'])
#filling Non valid values with mean for age, 

train['Age'].fillna((train['Age'].mean()), inplace=True)
sns.barplot(x='Sex', y='Survived', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Sex", fontsize=16)



plt.show()

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#This information can be displayed in the next plot too:

#sns.catplot(x='Sex', col='Survived', kind='count', data=train);
sns.barplot(x='Pclass', y='Survived', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass", fontsize=16)



plt.show()

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass and Sex")

plt.show()
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.pairplot(data=train, hue="Survived")
# I Create a swarmplot to detect patterns, where is the highest survival rate? 

sns.swarmplot(x = 'SibSp', y = 'Parch', hue = 'Survived', data = train, split = True, alpha=0.8)

plt.show()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



yf = train.Survived

base_features = ['Parch',

                 'SibSp','Age', 'Fare','Pclass']



Xf = train[base_features]



train_X, val_X, train_y, val_y = train_test_split(Xf, yf, random_state=1)

first_model = RandomForestRegressor(n_estimators=21, random_state=1).fit(train_X, train_y)
#Explore the relationship between SipSp and Parch in the predictions for a RF Model

inter  =  pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=['SibSp', 'Parch'])



pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=['SibSp', 'Parch'], plot_type='contour')

plt.show()
train['FamilySize'] = train['SibSp'] + train['Parch'] 

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).agg('mean')
train['IsAlone'] = 0

train.loc[train['FamilySize'] == 0, 'IsAlone'] = 1



train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
cols = ['Survived', 'Parch', 'SibSp', 'Embarked','IsAlone', 'FamilySize']



nr_rows = 2

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        

        i = r*nr_cols+c       

        ax = axs[r][c]

        sns.countplot(train[cols[i]], hue=train["Survived"], ax=ax)

        ax.set_title(cols[i], fontsize=14, fontweight='bold')

        ax.legend(title="survived", loc='upper center') 

        

plt.tight_layout()
feat_name = 'Fare'

pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
train[["Fare", "Survived"]].groupby(['Survived'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.groupby(['Sex','Survived'])[['Fare']].agg(['min','mean','max'])
train.loc[ train['Fare'] <= 7.22, 'Fare'] = 0

train.loc[(train['Fare'] > 7.22) & (train['Fare'] <= 21.96), 'Fare'] = 1

train.loc[(train['Fare'] > 21.96) & (train['Fare'] <= 40.82), 'Fare'] = 2

train.loc[ train['Fare'] > 40.82, 'Fare'] = 3

train['Fare'] = train['Fare'].astype(int)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Fare', bins=20)

plt.show()
sns.barplot(x='Sex', y='Survived', hue='Fare', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Fare and Sex")

plt.show()
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)

plt.show()
feat_name = 'Age'

pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()

#Exploring the relationship between Age and Pclass for a given model preductions

inter  =  pdp.pdp_interact(model=first_model, dataset=val_X, model_features=base_features, features=['Age', 'Pclass'])



pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=['Age', 'Pclass'], plot_type='contour')

plt.show()
#bins=np.arange(0, 80, 10)

g = sns.FacetGrid(train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Age', kde=False, bins=4, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()
train.loc[ train['Age'] <= 16, 'Age'] = 1

train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 2

train.loc[(train['Age'] > 32) & (train['Age'] <= 64), 'Age'] = 3

train.loc[ train['Age'] > 64, 'Age'] = 4

train['Age'] = train['Age'].astype(int)
sns.barplot(x='Pclass', y='Survived', hue='Age', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Age and Sex")

plt.show()
train['Age*Class'] = train.Age * train.Pclass
train[["Age*Class", "Survived"]].groupby(['Age*Class'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab([train.Survived], [train.Sex,train['Age*Class']], margins=True).style.background_gradient(cmap='autumn_r')
pd.crosstab([train.Survived], [train.Sex,train['IsAlone']], margins=True).style.background_gradient(cmap='autumn_r')
pd.crosstab([train.Survived], [train.Fare], margins=True).style.background_gradient(cmap='autumn_r')
train.head()
y2 = train.Survived



base_features2 = ['Parch','SibSp','Age', 'Fare','Pclass','Age*Class','FamilySize','IsAlone']



X2 = train[base_features2]

train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state=1)

second_model = RandomForestRegressor(n_estimators=21, random_state=1).fit(train_X2, train_y2)



inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['Age', 'Pclass'])

pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['Age', 'Pclass'], plot_type='contour')

plt.show()
feat_name = 'FamilySize'

pdp_dist = pdp.pdp_isolate(model=second_model, dataset=val_X2, model_features=base_features2, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['FamilySize', 'Pclass'])

pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['FamilySize', 'Pclass'], plot_type='contour')

plt.show()
feat_name = 'IsAlone'

pdp_dist = pdp.pdp_isolate(model=second_model, dataset=val_X2, model_features=base_features2, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
feat_name = 'Age*Class'

pdp_dist = pdp.pdp_isolate(model=second_model, dataset=val_X2, model_features=base_features2, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
inter2  =  pdp.pdp_interact(model=second_model, dataset=val_X2, model_features=base_features2, features=['Age*Class', 'IsAlone'])

pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=['Age*Class', 'IsAlone'], plot_type='contour')

plt.show()
# convert Sex values and Embearked values into dummis to use a numerical classifier 

dummies_Sex = pd.get_dummies(train.Sex)

dummies_Embarked = pd.get_dummies(train.Embarked)

#join the dummies to the final dataframe

train_ready = pd.concat([train, dummies_Sex,dummies_Embarked], axis=1)

train_ready.head()
#Drop the columns that are not usefull now

#train_ready = train_ready.drop(columns = ['Sex','Embarked','male','SibSp','Parch','Q'])



train_ready = train_ready.drop(columns = ['Sex','Embarked'])
#train_ready = train_ready.drop(columns = ['Age*Class'])
#train_ready = train_ready.drop(columns = ['FamilySize'])
#alst check before trainning

train_ready.info()
train_ready.head(10)
from scipy import stats

for name in train_ready:

    print(name, "column entropy :", round(stats.entropy(train_ready[name].value_counts(normalize=True), base=2),2))
#Upload the test file 

test = pd.read_csv("../input/test.csv")



#Drop unecessary columns

test = test.drop(columns = ['Cabin','Name','Ticket','PassengerId'])

#check the test dataframe

test.head()
#Check for the missing values in the columns 

fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(test.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
#filling Non valid values with mean for age, 

test['Age'].fillna((test['Age'].mean()), inplace=True)

test['Fare'].fillna((test['Fare'].mean()), inplace=True)
test.loc[ test['Fare'] <= 7.22, 'Fare'] = 0

test.loc[(test['Fare'] > 7.22) & (test['Fare'] <= 21.96), 'Fare'] = 1

test.loc[(test['Fare'] > 21.96) & (test['Fare'] <= 40.82), 'Fare'] = 2

test.loc[ test['Fare'] > 40.82, 'Fare'] = 3
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
test.loc[ test['Age'] <= 16, 'Age'] = 1

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 2

test.loc[(test['Age'] > 32) & (test['Age'] <= 64), 'Age'] = 3

test.loc[ test['Age'] > 64, 'Age'] = 4
test['Age*Class'] = test.Age * test.Pclass
test.info()
#as in the train dataset, build dummis in the sex and embarked columns

test_dummies_Sex = pd.get_dummies(test.Sex)

test_dummies_Embarked = pd.get_dummies(test.Embarked)

test_ready = pd.concat([test, test_dummies_Sex,test_dummies_Embarked], axis=1)

test_ready.head()
#drop these columns, we keep only numerical values

#train_ready = train_ready.drop(columns = ['Sex','Embarked','Survived','SibSp','Parch'])

test_ready = test_ready.drop(columns = ['Sex','Embarked'])
#test_ready = test_ready.drop(columns = ['Age*Class'])
#test_ready = test_ready.drop(columns = ['FamilySize'])
#check all is ok 

test_ready.info()
test_ready.head()
from scipy import stats

for name in test_ready:

    print(name, "column entropy :",round(stats.entropy(test_ready[name].value_counts(normalize=True), base=2),2))
## import ML

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



clfs.append(("LogReg", 

             Pipeline([("Scaler", StandardScaler()),

                       ("LogReg", LogisticRegression())])))



clfs.append(("XGBClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("XGB", XGBClassifier())]))) 

clfs.append(("KNN", 

             Pipeline([("Scaler", StandardScaler()),

                       ("KNN", KNeighborsClassifier(n_neighbors=8))]))) 



clfs.append(("DecisionTreeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("DecisionTrees", DecisionTreeClassifier())]))) 



clfs.append(("RandomForestClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RandomForest", RandomForestClassifier())]))) 



clfs.append(("GradientBoostingClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingClassifier(n_estimators=100))]))) 



clfs.append(("RidgeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RidgeClassifier", RidgeClassifier())])))



clfs.append(("BaggingRidgeClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("BaggingClassifier", BaggingClassifier())])))



clfs.append(("ExtraTreesClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreesClassifier())])))



#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'accuracy'

n_folds = 7



results, names  = [], [] 



for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, 

                                 cv= 5, scoring=scoring,

                                 n_jobs=-1)    

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
perm_xgb = PermutationImportance(XGBClassifier().fit(X_train, y_train), random_state=1).fit(X_test,y_test)

eli5.show_weights(perm_xgb, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())
perm_knn = PermutationImportance(KNeighborsClassifier(n_neighbors=8).fit(X_train, y_train), random_state=1).fit(X_test,y_test)

eli5.show_weights(perm_knn, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())
perm_gbc = PermutationImportance(GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train), random_state=1).fit(X_test,y_test)

eli5.show_weights(perm_gbc, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())
perm_gbc = PermutationImportance(RidgeClassifier().fit(X_train, y_train), random_state=1).fit(X_test,y_test)

eli5.show_weights(perm_gbc, feature_names = train_ready.drop('Survived',axis=1).columns.tolist())
#train_ready.drop('Survived',axis=1).columns
train_ready.drop('Survived',axis=1).info()
train_ready.shape
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
scaled_features.shape
# Import KNeighborsClassifier from sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier 



# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 19)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test, y_test)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')
# Import KNeighborsClassifier from sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier 



# Create a k-NN classifier with 6 neighbors: knn

knn_6 = KNeighborsClassifier(n_neighbors = 6)



# Fit the classifier to the data

knn_6.fit(scaled_features,y)



# Predict the labels for the training data X

y_pred_knn_6 = knn_6.predict(scaled_features_test)
# Import KNeighborsClassifier from sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier 



# Create a k-NN classifier with 6 neighbors: knn

knn_10 = KNeighborsClassifier(n_neighbors = 10)



# Fit the classifier to the data

knn_10.fit(scaled_features,y)



# Predict the labels for the training data X

y_pred_knn_10 = knn_10.predict(scaled_features_test)
#Upload the test file for KNN (scaled)

result_knn_6 = pd.read_csv("../input/gender_submission.csv")

result_knn_6['Survived'] = y_pred_knn_6

result_knn_6.to_csv('Titanic_knn_5.csv', index=False)
#Upload the test file for KNN (scaled)

result_knn_10 = pd.read_csv("../input/gender_submission.csv")

result_knn_10['Survived'] = y_pred_knn_10

result_knn_10.to_csv('Titanic_knn_7.csv', index=False)
logreg = LogisticRegression()

logreg.fit(scaled_features,y)

y_pred_logreg = logreg.predict(scaled_features_test)

y_pred_logreg.shape
#Upload the test file for Random Forest 

result_logreg = pd.read_csv("../input/gender_submission.csv")

result_logreg['Survived'] = y_pred_logreg

result_logreg.to_csv('Titanic_logreg.csv', index=False)
import xgboost as xgb

from xgboost import XGBClassifier



clf = xgb.XGBClassifier(n_estimators=250, random_state=4,bagging_fraction= 0.791787170136272, colsample_bytree= 0.7150126733821065,feature_fraction= 0.6929758008695552,gamma= 0.6716290491053838,learning_rate= 0.030240003246947006,max_depth= 2,min_child_samples= 5,num_leaves= 15,reg_alpha= 0.05822089056228967,reg_lambda= 0.14016232510869098,subsample= 0.9)



clf.fit(scaled_features, y)



y_pred_xgb= clf.predict(scaled_features_test)
#Upload the test file for Random Forest 

result_xgb = pd.read_csv("../input/gender_submission.csv")

result_xgb['Survived'] = y_pred_xgb

result_xgb.to_csv('Titanic_xgb.csv', index=False)
rcf= RidgeClassifier()

rcf.fit(scaled_features, y)



y_pred_rcf= rcf.predict(scaled_features_test)
#Upload the test file for  Ridge Classifier

result_rcf = pd.read_csv("../input/gender_submission.csv")

result_rcf['Survived'] = y_pred_rcf

result_rcf.to_csv('Titanic_rcf.csv', index=False)
gbc= GradientBoostingClassifier(n_estimators=100)

gbc.fit(scaled_features, y)

y_pred_gbc= gbc.predict(scaled_features_test)
#Upload the test file for Bagging Ridge Classifie

result_gbc = pd.read_csv("../input/gender_submission.csv")

result_gbc['Survived'] = y_pred_gbc

result_gbc.to_csv('Titanic_gbc.csv', index=False)