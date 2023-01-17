import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn import cross_validation, tree, linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
import warnings
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout

warnings.filterwarnings("ignore")

print('########Methods#########')
def readData(FileLocation,FileType):
    print('   ')
    print('loading file - ' + FileLocation)
    
    if FileType=='csv':
        data = pd.read_csv(FileLocation)
    
    print("Number of Columns - " + str(len(data.columns)))
    print("DataTypes - ")
    print(data.dtypes.unique())
    print(data.dtypes)
    print('checking null - ')
    print(data.isnull().any().sum(), ' / ', len(data.columns))
    print(data.isnull().any(axis=1).sum(), ' / ', len(data))
    print ('columns having null values - ')
    print(data.columns[data.isnull().any()])
    print(data.head())
    return data
    

def DataCleaning(data, columnsToBeDropped, fillNAValues):
    if (columnsToBeDropped):
        data = data.drop(columnsToBeDropped,axis=1)
    data.dropna(thresh=0.8*len(data), axis=1)
    data.dropna(thresh=0.8*len(data))
    for value in fillNAValues:
        data[value].fillna(data[value].median(), inplace = True)
    return data
    
   
def scatterPlot(data,ColumnsToPlot,hueColumn):
    with sns.plotting_context("notebook",font_scale=2.5):
        g = sns.pairplot(data[ColumnsToPlot], hue=hueColumn, size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
        g.set(xticklabels=[]);

def PlotDataCorrelation(data,vs):
    print(data.corr().abs().unstack().sort_values()[vs])
    

def featureRankingMatrix(data,x,y):
    ranks = {}
    
    colnames = data.columns
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x,2), ranks)
        return dict(zip(names, ranks))

    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(x, y)
    ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
    lr = LinearRegression(normalize=True)
    lr.fit(x,y)
    rfe = RFE(lr, n_features_to_select=1, verbose =3 )
    rfe.fit(x,y)
    ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

    
    lr = LinearRegression(normalize=True)
    lr.fit(x,y)
    ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)


    ridge = Ridge(alpha = 7)
    ridge.fit(x,y)
    ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)


    lasso = Lasso(alpha=.05)
    lasso.fit(x,y)
    ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

    rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
    rf.fit(x,y)
    ranks["RF"] = ranking(rf.feature_importances_, colnames);

    r = {}
    for name in colnames:
        r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
    meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
    sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", size=14, aspect=1.9, palette='coolwarm')


def cleanTitle(data):
    TitlesCount=data['Title'].value_counts()
    
    Title=[]
    for i, v in TitlesCount.iteritems():
        if(v < 10):
            Title.append(i)
            
    for index, row in data.iterrows():
        if row['Title'] in Title:
            data['Title'][index]='misc'
            
    
    return data

def ModelSelection(test_data,features,label):
    MLA = [
    
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
        
    gaussian_process.GaussianProcessClassifier(),
       
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
        
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
        
    neighbors.KNeighborsClassifier(),
        
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
        
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
        
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
        
    ]
    
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Score']
    MLA_compare = pd.DataFrame(columns = MLA_columns)
    x_train,x_test,y_train,y_test = train_test_split (train_data[features],train_data[label],test_size=0.2)
    row_index = 0
    MLA_predict = train_data[label]
    for alg in MLA:

        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
        alg.fit(x_train, y_train)
        MLA_predict[MLA_name] = alg.predict(x_test)
        MLA_compare.loc[row_index, 'MLA Score']=alg.score(x_test,y_test)
        row_index+=1

    
    MLA_compare.sort_values(by = ['MLA Score'], ascending = False, inplace = True)
    return MLA_compare,x_train,x_test,y_train,y_test

print("========Loading Data========")
FileLocation="../input/train.csv"
FileType="csv"
train_data=readData(FileLocation,FileType)
FileLocation="../input/test.csv"
test_data=readData(FileLocation,FileType)
pd.set_option('display.expand_frame_repr', False)
print("========Cleaning Data========")
fillNAValues=['Age','Fare']
test_data=DataCleaning(test_data, ['Cabin', 'Ticket'], fillNAValues)
train_data=DataCleaning(train_data, ['PassengerId','Cabin', 'Ticket'], fillNAValues)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data = test_data.drop(['SibSp','Parch'],axis=1)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data = train_data.drop(['SibSp','Parch'],axis=1)

test_data['Title'] = test_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train_data['Title'] = train_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train_data = train_data.drop(['Name'],axis=1)
test_data = test_data.drop(['Name'],axis=1)
    
test_data.loc[test_data['Embarked'].isnull()] = 'S'
train_data.loc[train_data['Embarked'].isnull()] = 'S'
    
test_data=cleanTitle(test_data)
train_data=cleanTitle(train_data)

label = LabelEncoder()

x=pd.get_dummies(test_data[['Sex','Embarked','Title']])
y=test_data[['FamilySize','Age','Fare','Pclass','PassengerId']]
test_data = pd.concat([x, y], axis=1, sort=False)

x=pd.get_dummies(train_data[['Sex','Embarked','Title']])
y=train_data[['Survived','FamilySize','Age','Fare','Pclass']]
train_data = pd.concat([x, y], axis=1, sort=False)

train_data = train_data.drop(['Sex_S'],axis=1)
train_data=train_data[train_data.Survived != 'S']
train_data[['Survived']] = train_data[['Survived']].astype(int)
test_data = test_data.astype('float')

features=['Title_Mr','Title_Miss','Title_Mrs','Sex_male','Sex_female','Embarked_S','Embarked_C','FamilySize','Age','Fare','Embarked_Q','Title_misc','Title_Master','Pclass']
print("========Data Correlation========")
PlotDataCorrelation(train_data,'Survived')
MLA_compare,x_train,x_test,y_train,y_test=ModelSelection(test_data,features,'Survived')
print(MLA_compare[['MLA Name','MLA Score']].head())
import os
cls=ensemble.GradientBoostingClassifier()
predictedOutput=test_data[['PassengerId']].astype('int')
cls.fit(train_data[features], train_data[['Survived']])
predictedOutput['Survived'] = cls.predict(test_data[features])
print(predictedOutput.head())
predictedOutput.to_csv('gender_submission.csv', sep=',', index=False)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

estimators = []

model1 = linear_model.LogisticRegressionCV()
estimators.append(('LRCV', model1))

model2 = discriminant_analysis.LinearDiscriminantAnalysis()
estimators.append(('LDA', model2))

model3 = linear_model.RidgeClassifierCV()
estimators.append(('RCCV', model3))

model4 = GradientBoostingClassifier()
estimators.append(('GBC', model4))

ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, x_train, y_train, cv=kfold)
print(results.mean())
import os
cls=ensemble
predictedOutput=test_data[['PassengerId']].astype('int')
cls.fit(train_data[features], train_data[['Survived']])
predictedOutput['Survived'] = cls.predict(test_data[features])
print(predictedOutput.head())
predictedOutput.to_csv('gender_submission.csv', sep=',', index=False)
x = x_train.values
y = y_train.values
x_val = x_test.values
y_val = y_test.values

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=14))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x, y,epochs=50,batch_size=32)
score = model.evaluate(x_val, y_val, batch_size=32)
print(score[1])