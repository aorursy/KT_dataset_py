import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

input_train_df = pd.read_csv('../input/titanic/train.csv')
input_test_df = pd.read_csv('../input/titanic/test.csv')
input_train_df.head()
#Cleaning - select required features
cleaned_train_df = input_train_df.drop(['Ticket'],axis=1)
cleaned_train_df.head()
#Fill missing data
print(cleaned_train_df.isnull().sum())
cleaned_train_df['Embarked'] = cleaned_train_df['Embarked'].fillna(cleaned_train_df.Embarked.dropna().mode()[0]) 
cleaned_train_df['Age'] = cleaned_train_df['Age'].fillna(cleaned_train_df.Age.dropna().mode()[0]) 
cleaned_train_df['Cabin'] = cleaned_train_df['Cabin'].fillna('NONE')
print(cleaned_train_df.isnull().sum())
#remove outliers
selectedFetaureForOutliers = ['Pclass','Age','Fare','Parch'] #numeric feilds
dfSelected = cleaned_train_df[selectedFetaureForOutliers]
Q1 = dfSelected.quantile(0.25)
Q3 = dfSelected.quantile(0.75)
IQR = Q3-Q1
print(IQR)
print("Before Remove outlier shape : "+str(dfSelected.shape))
noOutlier_df = dfSelected[~((dfSelected < (Q1 - 1.5 * IQR)) |(dfSelected > (Q3 + 1.5 * IQR))).any(axis=1)]
print("After Remove outlier shape : "+str(noOutlier_df.shape))
noOutlier_df.head()
#Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
noOutlier_df[['Age','Fare']] = scaler.fit_transform(noOutlier_df[['Age','Fare']])
noOutlier_df.head()
cleaned_train_df = cleaned_train_df.iloc[noOutlier_df.index,:]
cleaned_train_df.shape
print(cleaned_train_df['Cabin'].unique())
cleaned_train_df['CabinGroup'] = cleaned_train_df['Cabin'].str.slice(0,1)

cleaned_train_df['Title'] = cleaned_train_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
cleaned_train_df = cleaned_train_df.drop('Name',axis=1)
cleaned_train_df.head()
# Construct feature AgeCat
cleaned_train_df['AgeCat']=0
cleaned_train_df.loc[cleaned_train_df['Age']<=16,'AgeCat']=0
cleaned_train_df.loc[(cleaned_train_df['Age']>16)&(cleaned_train_df['Age']<=32),'AgeCat']=1
cleaned_train_df.loc[(cleaned_train_df['Age']>32)&(cleaned_train_df['Age']<=48),'AgeCat']=2
cleaned_train_df.loc[(cleaned_train_df['Age']>48)&(cleaned_train_df['Age']<=64),'AgeCat']=3
cleaned_train_df.loc[cleaned_train_df['Age']>64,'AgeCat']=4

input_test_df['AgeCat']=0
input_test_df.loc[input_test_df['Age']<=16,'AgeCat']=0
input_test_df.loc[(input_test_df['Age']>16)&(input_test_df['Age']<=32),'AgeCat']=1
input_test_df.loc[(input_test_df['Age']>32)&(input_test_df['Age']<=48),'AgeCat']=2
input_test_df.loc[(input_test_df['Age']>48)&(input_test_df['Age']<=64),'AgeCat']=3
input_test_df.loc[input_test_df['Age']>64,'AgeCat']=4

# Construct feature FareCat
cleaned_train_df['FareCat']=0
cleaned_train_df.loc[cleaned_train_df['Fare']<=7.775,'FareCat']=0
cleaned_train_df.loc[(cleaned_train_df['Fare']>7.775)&(cleaned_train_df['Fare']<=8.662),'FareCat']=1
cleaned_train_df.loc[(cleaned_train_df['Fare']>8.662)&(cleaned_train_df['Fare']<=14.454),'FareCat']=2
cleaned_train_df.loc[(cleaned_train_df['Fare']>14.454)&(cleaned_train_df['Fare']<=26.0),'FareCat']=3
cleaned_train_df.loc[(cleaned_train_df['Fare']>26.0)&(cleaned_train_df['Fare']<=52.369),'FareCat']=4
cleaned_train_df.loc[cleaned_train_df['Fare']>52.369,'FareCat']=5

input_test_df['FareCat']=0
input_test_df.loc[input_test_df['Fare']<=7.775,'FareCat']=0
input_test_df.loc[(input_test_df['Fare']>7.775)&(input_test_df['Fare']<=8.662),'FareCat']=1
input_test_df.loc[(input_test_df['Fare']>8.662)&(input_test_df['Fare']<=14.454),'FareCat']=2
input_test_df.loc[(input_test_df['Fare']>14.454)&(input_test_df['Fare']<=26.0),'FareCat']=3
input_test_df.loc[(input_test_df['Fare']>26.0)&(input_test_df['Fare']<=52.369),'FareCat']=4
input_test_df.loc[input_test_df['Fare']>52.369,'FareCat']=5

# Construct feature FamilySize
cleaned_train_df['FamilySize'] = cleaned_train_df['Parch'] + cleaned_train_df['SibSp']
input_test_df['FamilySize'] = input_test_df['Parch'] + input_test_df['SibSp']

cleaned_train_df.head(2)
import matplotlib.pyplot as plt
%matplotlib inline
fig,axes = plt.subplots(nrows=4,ncols=2)
fig.tight_layout(pad=1.0)
def plot_stat(data,column,ax_pos):
    grouped = data.groupby([column]).agg({'Survived':['count','sum']})
    grouped.plot(kind='bar',title=column+' Total Vs Survived',ax=ax_pos,figsize=(15,10))
    grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
    stat = pd.DataFrame({'Survival_rate':grouped['Survived_sum'].divide(grouped['Survived_count'])*100})
    stat = stat.set_index(column+'_'+stat.index.astype(str))
    return stat

sexBaseSurviveRate = plot_stat(cleaned_train_df,'Sex',axes[0,0])
pclassBaseSurviveRate = plot_stat(cleaned_train_df,'Pclass',axes[0,1])
parchBaseSurviveRate = plot_stat(cleaned_train_df,'Parch',axes[1,0])
embarkedBaseSurviveRate = plot_stat(cleaned_train_df,'Embarked',axes[1,1])
sibSpBaseSurviveRate = plot_stat(cleaned_train_df,'SibSp',axes[2,0])
cabinGroupBaseSurviveRate = plot_stat(cleaned_train_df,'CabinGroup',axes[2,1])
titleBaseSurviveRate = plot_stat(cleaned_train_df,'Title',axes[3,0])
AgeCatBaseSurviveRate = plot_stat(cleaned_train_df,'AgeCat',axes[3,1])

survival_rate_df=sexBaseSurviveRate.append(pclassBaseSurviveRate).append(parchBaseSurviveRate).append(embarkedBaseSurviveRate).append(sibSpBaseSurviveRate).append(cabinGroupBaseSurviveRate).append(titleBaseSurviveRate)
survival_rate_df.sort_values(by=['Survival_rate'],ascending=False)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X = cleaned_train_df.drop(['Survived','PassengerId','Cabin',"Age","Fare","Parch","SibSp"],axis=1)
X['CabinGroup'] = le.fit_transform(X['CabinGroup'])
X['Embarked'] = le.fit_transform(X['Embarked'])
X['Sex'] = le.fit_transform(X['Sex'])
X['Title'] = le.fit_transform(X['Title'])

Y = cleaned_train_df['Survived']

test_altered = input_test_df.drop(['PassengerId','Ticket',"Parch","SibSp"],axis=1)
test_altered['Cabin'] = test_altered['Cabin'].fillna('NONE')
test_altered['CabinGroup'] = test_altered['Cabin'].str.slice(0,1)
test_altered['CabinGroup'] = le.fit_transform(test_altered['CabinGroup'])
test_altered['Embarked'] = le.fit_transform(test_altered['Embarked'])
test_altered['Sex'] = le.fit_transform(test_altered['Sex'])
test_altered['Title'] = test_altered['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
test_altered['Title'] = le.fit_transform(test_altered['Title'])
test_altered = test_altered.drop('Name',axis=1)
X_predict_test = test_altered.drop(['Cabin','Age','Fare'],axis=1)

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neural_network
from xgboost import XGBClassifier

random_state = 20
algorithums = [
    linear_model.LogisticRegressionCV(max_iter = 50000,random_state=random_state),
    linear_model.PassiveAggressiveClassifier(random_state=random_state),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(random_state=random_state),
    linear_model.Perceptron(random_state=random_state),
    
    svm.SVC(max_iter = 500000,probability=True,kernel='linear',C=0.025),
    svm.NuSVC(max_iter = 500000,probability=True),
    svm.LinearSVC(max_iter = 500000),
    
    ensemble.AdaBoostClassifier(random_state=random_state,n_estimators=500,learning_rate=0.75),
    ensemble.BaggingClassifier(random_state=random_state),
    ensemble.ExtraTreesClassifier(random_state=random_state,max_depth=6,min_samples_leaf=2),
    ensemble.GradientBoostingClassifier(random_state=random_state,n_estimators=500,max_depth=6,min_samples_leaf=2),
    ensemble.RandomForestClassifier(random_state=random_state,n_estimators=500,warm_start=True,max_depth=6,min_samples_leaf=2,max_features='sqrt'),
    
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    neighbors.KNeighborsClassifier(),
    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    gaussian_process.GaussianProcessClassifier(),
    
    XGBClassifier(),
    
    neural_network.MLPClassifier(random_state=random_state)
]    
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_validate,ShuffleSplit

result_table_columns = ['Algo','Train Accuracy','Test Accuracy','Fit Time','Score Time']

features = ['Pclass', 'Sex','Embarked', 'AgeCat','FareCat', 'FamilySize', 'CabinGroup','Title']

results = pd.DataFrame(columns=result_table_columns)

def model_stats(algo,X,Y,features,row_index):    
    cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0 ) 
    cv_results = cross_validate(algo,X[features],Y,cv=cv_split,return_train_score=True)
    
    algo.fit(X[features],Y)
    
    results.loc[row_index,'Algo'] = algo.__class__.__name__
    results.loc[row_index,'Fit Time'] = cv_results['fit_time'].mean()
    results.loc[row_index,'Score Time'] = cv_results['score_time'].mean()
    results.loc[row_index,'Train Accuracy'] = cv_results['train_score'].mean() *100
    results.loc[row_index,'Test Accuracy'] = cv_results['test_score'].mean()  *100  
    
row_index = 0
for algo in algorithums:
    model_stats(algo,X,Y,features,row_index)
    row_index += 1

results.sort_values(by='Test Accuracy',ascending=False)
vote_est = [
    ('bnb', naive_bayes.BernoulliNB()),
    ('gbc', ensemble.GradientBoostingClassifier(random_state=random_state,n_estimators=500,max_depth=6,min_samples_leaf=2)),
    ('xgb', XGBClassifier()),
    ('ada', ensemble.AdaBoostClassifier(random_state=random_state,n_estimators=500,learning_rate=0.75)),
    ('bc', ensemble.BaggingClassifier(random_state=random_state)),
    ('etc',ensemble.ExtraTreesClassifier(random_state=random_state,max_depth=6,min_samples_leaf=2)), 
    ('rfc', ensemble.RandomForestClassifier(random_state=random_state,n_estimators=500,warm_start=True,max_depth=6,min_samples_leaf=2,max_features='sqrt')),
    ('gpc', gaussian_process.GaussianProcessClassifier()),  
    ('lr', linear_model.LogisticRegressionCV(max_iter = 50000)),  
    ('gnb', naive_bayes.GaussianNB()), 
    ('knn', neighbors.KNeighborsClassifier()), 
    ('svc', svm.SVC(probability=True))
]

#Hard Vote
cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0 ) 
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = cross_validate(vote_hard, X[features],Y, cv  = cv_split,return_train_score=True)
vote_hard.fit(X[features],Y)

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)


#Soft Vote
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = cross_validate(vote_soft, X[features],Y, cv=cv_split,return_train_score=True)
vote_soft.fit(X[features],Y)

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)
# bestAlgo = ensemble.RandomForestClassifier(random_state=random_state,n_estimators=500,warm_start=True,max_depth=6,min_samples_leaf=2,max_features='sqrt')
# bestAlgo.fit(X[features],Y)
bestAlgo = vote_hard

Y_trainPredict = bestAlgo.predict(X[features])

Y_predict = bestAlgo.predict(X_predict_test)

print(bestAlgo.score(X[features],Y))

confusion_matrix(Y,Y_trainPredict)
submission = pd.DataFrame({
        "PassengerId": input_test_df["PassengerId"],
        "Survived": Y_predict
    })
#submission.to_csv('submissionVotingHard.csv', index=False)
submission.head()