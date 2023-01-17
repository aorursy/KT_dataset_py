import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats

from scipy.stats import norm

import seaborn as sns

sns.set(context = 'paper', palette = 'winter_r', style = 'darkgrid', rc= {'figure.facecolor': 'gray',}, font_scale=1.5)
def Readcsv(data):

    return (pd.read_csv(data,index_col = 'PassengerId'))



traindf = Readcsv('../input/train.csv')

testdf  = Readcsv('../input/test.csv')   
traindf.columns, testdf.columns
traindf.head()
traindf.isnull().sum()
plt.figure(figsize = [17,6])

a = sns.distplot(traindf['Age'].dropna(),bins = range(0,81,1), rug = True, fit = norm)
f, ax = plt.subplots(4, 3,figsize=[22,12])

#sns.set_palette('Set1', 8, .75)

#sns.set_style('whitegrid')

#plt.subplot(331)

sns.violinplot(x = 'Sex', y = 'Survived', data = traindf,ax = ax[0,0])

#sns.despine(offset=10, trim=True)

#plt.subplot(332)

sns.barplot(x = 'Pclass',y ='Survived',data = traindf,hue = 'Embarked',ax = ax[0,1])

#plt.subplot(333)

sns.distplot(traindf[traindf['Survived']==1]['Age'].dropna(),norm_hist = True,bins = np.arange(0,81,1),color = 'blue',

            ax = ax[0,2])

sns.distplot(traindf[traindf['Survived']==0]['Age'].dropna(),norm_hist = True,bins = np.arange(0,81,1), color = 'red',

            ax = ax[0,2])

#plt.subplot(334)

sns.violinplot(x = 'Sex', y = 'Fare', data = traindf,ax = ax[1,0])

#plt.subplot(335)

sns.barplot(x ='Pclass', y = 'Fare',data = traindf , hue = 'Embarked', ax = ax[1,1])

#plt.subplot(336)

sns.distplot(traindf[traindf['Survived']==1]['Fare'].dropna(),bins = np.arange(0,580,10),color = 'blue',

            ax = ax[1,2])

sns.distplot(traindf[traindf['Survived']==0]['Fare'].dropna(),bins = np.arange(0,580,10),color = 'red',

            ax = ax[1,2])



#plt.subplot(337)

sns.violinplot(x = 'Sex', y ='SibSp',data = traindf,ax = ax[2,0])

#plt.subplot(338)

sns.barplot(x= 'Pclass', y = 'SibSp', data = traindf, hue = 'Embarked', ax = ax[2,1])

#plt.subplot(339)

sns.regplot(x = 'Fare', y = 'Age', data = traindf, ax = ax[2,2])

#plt.subplot(341)

sns.violinplot(x = 'Sex', y = 'Parch', data = traindf, ax = ax[3,0])

#plt.subplot(342)

sns.barplot(x = 'Pclass', y = 'Parch', data = traindf, hue = 'Embarked', ax = ax[3,1])

plt.close(12)

plt.close(13)

plt.close(14)
traindf[traindf['Fare']>500]
def get_isnull(train,test):

    return(train.isnull().sum(), test.isnull().sum())    
get_isnull(traindf,testdf)
#function for concatation so that we won't have to repeat them again and again in future.

def combine(data1,data2):

    fulldf = pd.concat([data1,data2])

    return fulldf
#Function for separation so that we won't have to repeat them again.

def saperate(data):

    data1 = data.iloc[:len(traindf)]

    data2 = data.iloc[len(traindf):]

    return data1, data2
testdf[testdf['Fare'].isnull()]
def filling(data1,data2):

    data = combine(data1,data2)

    data['Embarked'] = data['Embarked'].fillna('C')

    data['Age']      = data['Age'].fillna(data['Age'].median())

    data['Fare']     = data['Fare'].fillna(data['Fare'].median())

    data['Cabin']    = data['Cabin'].fillna('Z')

    data['Cabin']    = data['Cabin'].apply(lambda x: str(x)[0])

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = filling(traindf, testdf)

#testdf = filling(testdf)

sns.barplot(x= 'Cabin', y = 'Survived', data = traindf ,order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Z', 'T'])
def pew(data1, data2):

    data = combine(data1,data2)

    data['Cabin'] = data['Cabin'].replace(['B', 'D', 'E'], 'H')   #High

    data['Cabin'] = data['Cabin'].replace(['F', 'C'], 'M')        #Medium

    data['Cabin'] = data['Cabin'].replace(['T', 'G', 'A',], 'L')  #Low

    data['Cabin'] = data['Cabin'].replace(['Z'],'X')              #Missing

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = pew(traindf,testdf)

#testdf  = pew(testdf)
def family(data1, data2):

    data = combine(data1,data2)

    data['Family'] = data['SibSp'] + data['Parch'] + 1

    data['Alone']  = data['Family'].apply(lambda x: 1 if x == 1 else 0)

    data.drop(['SibSp','Parch'],axis = 1, inplace = True)

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = family(traindf, testdf)

#testdf  = family(testdf)

f,ax = plt.subplots(1,3,figsize=[20,7])

sns.barplot(x = 'Alone', y = 'Survived', data = traindf,orient = 'h', ax = ax[0])

sns.barplot(x = 'Survived', y = 'Family', data = traindf,orient = 'h', ax = ax[1])

sns.factorplot(y = 'Family',data = traindf, kind = 'count', orient = 'h', ax = ax[2])

plt.close(2)

plt.close(3)
traindf.head()
def FamilyGroup(data1, data2):

    data = combine(data1,data2)

    data.loc[data['Family'] > 2, 'FamilyGroup'] = 3

    data.loc[data['Family'] == 1, 'FamilyGroup'] = 1

    data.loc[data['Family'] == 2, 'FamilyGroup'] = 2

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = FamilyGroup(traindf, testdf)

#testdf  = FamilyGroup(testdf) 
def mapping(data1, data2):

    data = combine(data1,data2)

    data['Embarked'] = data['Embarked'].map({'C':1, 'S':2, 'Q':3})

    data['Sex']      = data['Sex'].map({'male': 1,'female':0})

    data['CabinGroup'] = data['Cabin'].map({'H': 0, 'M': 1, 'L': 2, 'X':3})

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = mapping(traindf, testdf)

#testdf  = mapping(testdf)
traindf['CabinGroup'].value_counts()
traindf.head()
def Titles(data1, data2):

    data = combine(data1,data2)

    data['Title'] = data['Name'].apply(lambda x: str(x).split(',')[1].split('.')[0])

    data['TitleGroup'] = 0

    data['TitleGroup'] = data['Title'].replace(['Mme','Ms','Lady','Sir','Mlle','the Countess',],0,          #High

                                            regex = True).replace(['Mrs','Miss','Master',],1,               #Medium

                                            regex = True).replace(['Dr','Major','Col','Mr'],2,              #Low

                                            regex = True).replace(['Don','Rev','Capt','Jonkheer','Dona'],4, #Least

                                            regex = True)

    #data['TitleGroup'] = data['TitleGroup'].replace({'male':1,'female':0,'Special':2})

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = Titles(traindf, testdf)

#testdf  = Titles(testdf)

##['Mrs','Miss','Mme','Ms','Lady','Mlle','Countess','Dona'],'female'

##['Sir','Don','Rev','Mr',],'male'

##['Col','Dr','Col','Capt','Major','Jonkheer','Master'],'Special'
plt.figure(figsize = [8,5])

sns.barplot(x = 'Survived', y = 'Title', data = traindf, palette = 'Blues_d',)
def surname(data1, data2):

    data = combine(data1,data2)

    data['Surname'] = data['Name'].apply(lambda x: str(x).split(' ')[0].split(',')[0])

    Shares = 0

    Shares = data.groupby('Surname').apply(lambda x: x.shape[0])

    data['SharedSurname'] = data['Surname'].map(Shares)

    traindf, testdf    = saperate(data)

    return traindf, testdf



traindf, testdf = surname(traindf, testdf)

traindf.loc[traindf['Ticket'].str.contains('113803')]
def age_distribution(data1, data2):

    data = combine(data1,data2)

    data.loc[data['Age']].round()

    data.loc[data['Age'] <= 16, 'AgeGroup'] = 1

    data.loc[(data['Age'] > 16) & (data['Age'] <= 40), 'AgeGroup'] = 2

    data.loc[(data['Age'] > 40) & (data['Age'] < 60), 'AgeGroup'] = 3

    data.loc[(data['Age'] >= 60), 'AgeGroup'] = 4

    #data['AgeGroup'].astype(int)

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = age_distribution(traindf, testdf)

#testdf = age_distribution(testdf)

plt.figure(figsize = [17,6])

sns.barplot(x = traindf['AgeGroup'], y = traindf['Survived'])#data = traindf,)#ci = 95, orient = 'v')

plt.rc('xtick',labelsize = 12)
traindf.head(2)
sns.factorplot(x = 'Pclass', y = 'Fare', col = 'Embarked', hue = 'Sex', data = traindf, margin_titles = True)
def SharedSurname(data1, data2):

    data = combine(data1,data2)

    Shares = 0

    Shares = data.groupby('Surname').apply(lambda x: x.shape[0])

    data['SharedSurname'] = data['Surname'].map(Shares)

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = SharedSurname(traindf, testdf)

#testdf  = SharedSurname(testdf)
testdf.head(3)
def fare_distribution(data1, data2):

    data = combine(data1,data2)

    payment = data.groupby('Pclass')['Fare'].mean()

    data.loc[(data['Fare'] < payment[3]),'FareGroup'] = 1

    data.loc[(data['Fare'] > payment[3]) & (data['Fare'] <  payment[2]),'FareGroup'] = 2

    data.loc[(data['Fare'] > payment[2]) & (data['Fare'] <  payment[1]), 'FareGroup'] = 3

    data.loc[(data['Fare'] > payment[1]),'FareGroup'] = 4

    #data['Fare'] = data['Fare'].map({'Very Low': 0, 'Low': 1, 'Medium':2, 'High':3})

    data['FareGroup'] = data['FareGroup'].astype(int)

    traindf, testdf    = saperate(data)

    return traindf, testdf

#payment = fulldf.groupby('Pclass')['Fare'].mean()

traindf, testdf = fare_distribution(traindf, testdf)

#testdf = fare_distribution(testdf)

#pd.get_dummies(traindf,columns = (['Pclass','Sex','Age','Fare','Embarked','Family','Alone','Title']), drop_first = True)
def Tickets(data1, data2):

    data = combine(data1,data2)

    #Creating a new column to display the freq of the tickets present

    data['SharedTicketCount'] = data.groupby('Ticket')[['Fare']].transform('count').astype(int)

    #Removing any characters other than alphabets and numbers to count the ticket length

    data['Ticket'] = data['Ticket'].str.replace('.','').str.replace('/','').str.replace(' ','')

    data['TicketLength'] = data['Ticket'].apply(lambda x: len(str(x)))

    traindf, testdf    = saperate(data)

    return traindf, testdf

traindf, testdf = Tickets(traindf, testdf)

#testdf = Tickets(testdf)
traindf.columns
traindf.head()
X = traindf.drop(['Name','Title','Surname','Survived','Cabin','Ticket',

                  'Age','Fare','Family','Alone'],axis = 1)

y = traindf['Survived']

X.shape , y.shape
X.columns
f, ax = plt.subplots(figsize = [25,16])

sns.heatmap(traindf.corr(),linewidths = .5, annot = True, cmap = 'YlGnBu', square = True)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier, plot_importance 
X.columns
npX = np.array(X).copy()

npy = np.array(y).copy()
clf_rf = RandomForestClassifier()

clf_et = ExtraTreesClassifier()

clf_bc = BaggingClassifier()

clf_ada = AdaBoostClassifier()

clf_dt = DecisionTreeClassifier()

clf_xg = XGBClassifier()

clf_lr = LogisticRegression()

clf_svm = SVC()
Classifiers = ['RandomForest','ExtraTrees','Bagging','AdaBoost','DecisionTree','XGBoost','LogisticRegression','SVM']

scores = []

models = [clf_rf, clf_et, clf_bc, clf_ada, clf_dt, clf_xg, clf_lr, clf_svm]

for model in models:

    score = cross_val_score(model, npX, npy, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()

    scores.append(score)
mode = pd.DataFrame(scores, index = Classifiers, columns = ['score']).sort_values(by = 'score',

             ascending = False)
mode
parameters_xg = {'max_depth':[3,6,7], 'learning_rate': [0.1,0.2], 'n_estimators': [300,200], 

                 'min_child_weight': [4], 'reg_alpha': [6,0], 'reg_lambda': [1,8],'max_delta_step':[2],

                 'gamma':[0],'seed':[1]}



parameters_svm = {'C':[0.9,0.01],'kernel':['rbf','linear'], 'gamma':[0,0.1,'auto'], 'probability':[True,False],

                  'random_state':[0,7,16],'decision_function_shape':['ovo','ovr'],'degree':[3,4,10]}



parameters_rf = {'n_estimators': [100,50], 'max_features': [7,'auto',None],

                 'n_jobs': [-1], 'min_samples_leaf': [2,4,], 'random_state':[1,7,], 

                 'min_samples_split':[2,6,], 'oob_score': [True,False],

                 'criterion': ['gini'], 'warm_start': [True,False]}

def grid(model,parameters):

    grid = GridSearchCV(estimator = model, param_grid = parameters, cv = 10, 

                        scoring = 'accuracy')

    grid.fit(npX,npy)

    return grid.best_score_, grid.best_estimator_.get_params()
def imp_features(model, model_name, params):

    Model = model(**params)

    Model.fit(npX,npy)

    names = X.columns

    feature = Model.feature_importances_

    important_features = pd.Series(data = feature, index = names,)

    important_features = important_features.sort_values(ascending = True)

    return important_features.plot(kind = 'barh', grid = False,title = model_name)
best_score_xg, best_params_xg = grid(clf_xg,parameters_xg)

print(best_score_xg)

imp_features(XGBClassifier, 'XGBoostClassifier', best_params_xg)
best_score_rf, best_params_rf = grid(clf_rf, parameters_rf)

print(best_score_rf)

imp_features(RandomForestClassifier,'Random Forest', best_params_rf)
best_score_svm, best_params_svm = grid(clf_svm, parameters_svm)

print(best_score_svm)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

x = StandardScaler().fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .2)

X_train_reduced = PCA(n_components = 2).fit_transform(X_train)

X_test_reduced  = PCA(n_components=  2).fit_transform(X_test)
def boundaries(model, heading, best_params):

    Model = model(**best_params)

    Model.fit(X_train_reduced, y_train)



    X_set, y_set = np.concatenate([X_train_reduced, X_test_reduced], axis = 0), np.concatenate([y_train, y_test], axis = 0)

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



    #plt.figure(figsize = [15,16])

    plt.contourf(X1, X2, Model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                     alpha = 0.5, cmap = ListedColormap(('k', 'blue')))

    

    plt.xlim(X1.min(), X1.max())

    plt.ylim(X2.min(), X2.max())



    for i, j in enumerate(np.unique(y_set)):

        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

    plt.xticks(fontsize = 3)

    plt.yticks(fontsize = 3)
fig = plt.figure(figsize=[15,7])



ax = plt.subplot(1,2,1)

plt.title('XGBClassifier')

boundaries(XGBClassifier,'eXtreme Boosting Classifier', best_params_xg)





ax = plt.subplot(1,2,2)

plt.title('RandomForest')

boundaries(RandomForestClassifier, 'Random Forest', best_params_rf)
from mlxtend.plotting import plot_decision_regions

t = np.array(y_train)

t = t.astype(np.integer)

clf_svm = SVC(**best_params_svm)

clf_svm.fit(X_train_reduced,t)

plt.figure(figsize = [15,10])

plot_decision_regions(X_train_reduced, t, clf = clf_svm, hide_spines = False, colors = 'purple,limegreen',

                      markers = ['^','v'])

plt.title('Support Vector Machines')
clf_svm = SVC(**best_params_svm)

clf_svm.fit(npX,npy)

testdf = testdf.drop(['Name','Title','Surname','Survived','Cabin','Ticket',

                  'Age','Fare','Family','Alone'],axis = 1)

nptest = np.array(testdf)

pred = clf_svm.predict(nptest)

predictions = pd.DataFrame(pred, index = testdf.index, columns = ['Survived'])

predictions.to_csv('predictions_svm_with_groups.csv')