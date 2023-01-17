import pandas as pd
data_train = pd.read_csv("train.csv")
data_test  = pd.read_csv("test.csv")
data_all   = pd.concat([data_train,data_test]).reset_index(drop=True)
print('Train Data Size =',np.shape(data_train))
print('Test Data Size  =',np.shape(data_test))
print('Full Data Size  =',np.shape(data_all))
print( data_all.isnull().sum()) # Age, Cabin & Embarked are NULL
data_all.head(3)
new1= pd.DataFrame(data_all.Name.str.split(',').tolist(),columns=list('AB'))
new1 = pd.DataFrame(new1.B.str.split().tolist())
data_all ['Title']= new1.iloc[:, 0]
data_all.head(3)
print( data_all.isnull().sum()) 
data_all['Age']   = data_all.groupby(['Pclass','Sex','SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))
data_all['Fare']  = data_all.groupby(['Pclass','Sex','SibSp'])['Fare'].apply(lambda x: x.fillna(x.median()))
data_all['Deck']  = data_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
data_all.loc[data_all['Embarked'].isnull() ,'Embarked']='M'
print('=========================')
print( data_all.isnull().sum())
data_all['AgeCat'] = ''
data_all['AgeCat'].loc[(data_all['Age'] < 16)                    ] = 'young'
data_all['AgeCat'].loc[(data_all['Age'] >= 16) & (data_all['Age'] < 50)] = 'mature'
data_all['AgeCat'].loc[(data_all['Age'] >= 50)                   ] = 'senior'

data_all['FamilySize'] = ''
data_all['FamilySize'].loc[(data_all['SibSp'] <= 2)                     ] = 'small'
data_all['FamilySize'].loc[(data_all['SibSp'] > 2) & (data_all['SibSp'] <= 5 )] = 'medium'
data_all['FamilySize'].loc[(data_all['SibSp'] > 5)                      ] = 'large'

data_all['IsAlone'] = ''
data_all['IsAlone'].loc[((data_all['SibSp'] + data_all['Parch']) > 0)] = '0'
data_all['IsAlone'].loc[((data_all['SibSp'] + data_all['Parch']) == 0)] = '1'

data_all['SexCat'] = ''
data_all['SexCat'].loc[(data_all['Sex'] == 'male'  ) & (data_all['Age'] <  50)] = 'youngmale'
data_all['SexCat'].loc[(data_all['Sex'] == 'male'  ) & (data_all['Age'] >= 50)] = 'seniormale'
data_all['SexCat'].loc[(data_all['Sex'] == 'female') & (data_all['Age'] <  50)] = 'youngfemale'
data_all['SexCat'].loc[(data_all['Sex'] == 'female') & (data_all['Age'] >= 50)] = 'seniorfemale'

#categorical_columns = ['Pclass', 'Sex', 'Embarked', 'AgeCat', 'FamilySize', 'IsAlone', 'SexCat']
#for col in categorical_columns:
#    data_all[col] = data_all[col].astype('category')
data_all.drop(['SibSp','Parch','Age', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data_all.head(3)
from sklearn.preprocessing import LabelEncoder
le_Sex               = LabelEncoder()
le_Embarked          = LabelEncoder()
le_SexCat            = LabelEncoder()
le_FamilySize        = LabelEncoder()
le_AgeCat            = LabelEncoder()
le_Title             = LabelEncoder()
le_Deck              = LabelEncoder()
data_all['le_Deck']        = le_Sex.fit_transform(data_all.Deck)
data_all['le_Title']       = le_Sex.fit_transform(data_all.Title)
data_all['le_Sex']         = le_Sex.fit_transform(data_all.Sex)
data_all['le_Embarked' ]   = le_Embarked.fit_transform(data_all.Embarked)
data_all['le_SexCat' ]     = le_SexCat.fit_transform(data_all.SexCat)
data_all['le_FamilySize' ] = le_FamilySize.fit_transform(data_all.FamilySize)
data_all['le_AgeCat' ]     = le_AgeCat.fit_transform(data_all.AgeCat)
data_all.drop(['Deck', 'Title', 'Sex','Embarked','SexCat','FamilySize','AgeCat'], axis=1, inplace=True)
data_all.head(3)
data_train=data_all.loc[:890]
data_train.drop(['PassengerId'], axis=1, inplace=True)
data_test=data_all.loc[891:]
print('Train Data Size =',np.shape(data_train))
print('Test Data Size  =',np.shape(data_test))
data_train.head(3)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

X, y = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_train_norm = mms.fit_transform(X_train)
X_test_norm  = mms.transform(X_test)
# SBS Algorithm 
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            dim = X_train.shape[1]
            self.indices_ = tuple(range(dim))
            self.subsets_ = [self.indices_]
            score = self._calc_score(X_train, y_train,
            X_test, y_test, self.indices_)
            self.scores_ = [score]
            while dim > self.k_features:
                scores = []
                subsets = []
                for p in combinations(self.indices_, r=dim - 1):
                    score = self._calc_score(X_train, y_train, X_test, y_test, p)
                    scores.append(score)
                    subsets.append(p)
                    best = np.argmax(scores)
                    self.indices_ = subsets[best]
                    self.subsets_.append(self.indices_)
                    dim -= 1
                    self.scores_.append(scores[best])
                    self.k_score_ = self.scores_[-1]
            return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test,indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
#LOGISTIC REGRESSION
lr = LogisticRegression(penalty='l2', C=100.0, solver='liblinear')
lr.fit(X_train_norm, y_train)
y_train_pred = lr.predict(X_train_norm)
y_test_pred = lr.predict(X_test_norm)
lr_train = accuracy_score(y_train, y_train_pred)
lr_test = accuracy_score(y_test, y_test_pred)
# DECISION TREE CLASSIFIER
tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)
tree = tree.fit(X_train_norm, y_train)
y_train_pred = tree.predict(X_train_norm)
y_test_pred = tree.predict(X_test_norm)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
# BAGGING CLASSIFIER
bag = BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,bootstrap=True,
                        bootstrap_features=False,n_jobs=1,random_state=1)
bag = bag.fit(X_train_norm, y_train)
y_train_pred = bag.predict(X_train_norm)
y_test_pred = bag.predict(X_test_norm)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
#ADABOOST CLASSIFIER
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)
ada = ada.fit(X_train_norm, y_train)
y_train_pred = ada.predict(X_train_norm)
y_test_pred = ada.predict(X_test_norm)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
#KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
sbs = SBS(knn, k_features=1).fit(X_train_norm, y_train)
sbs.fit(X_train_norm, y_train)
k3 = list(sbs.subsets_[8])
knn.fit(X_train_norm[:, k3], y_train)
knn_train=knn.score(X_train_norm[:, k3], y_train)
knn_test=knn.score(X_test_norm[:, k3], y_test)
#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1, criterion='entropy')
forest.fit(X_train_norm, y_train)
forest_train=forest.score(X_train_norm, y_train)
forest_test=forest.score(X_test_norm, y_test)
print('                                            Train Accuracy  Test Accuracy')
print('Logistic Regression                       : %.3f            %.3f' % (lr_train, lr_test))
print('KNN train/test accuracies                 : %.3f            %.3f' % (knn_train, knn_test))
print('AdaBoost train/test accuracies            : %.3f            %.3f' % (ada_train, ada_test))
print('Bagging train/test accuracies             : %.3f            %.3f' % (bag_train, bag_test))
print('Decision Tree train/test accuracies       : %.3f            %.3f' % (tree_train, tree_test))
print('Random Forest train/test accuracies       : %.3f            %.3f' % (forest_train, forest_test))
print('Test Data Size  =',np.shape(data_test))
data_test.head()
#from sklearn.preprocessing import MinMaxScaler
Xt = data_test.iloc[:, 2:].values
mms = MinMaxScaler()
Xt_test_norm = mms.fit_transform(Xt)
print('Test Data Size  =',np.shape(Xt))
lr_y_test_pred = lr.predict(Xt_test_norm)
dt_y_test_pred = tree.predict(Xt_test_norm)
bag_y_test_pred = bag.predict(Xt_test_norm)
ada_y_test_pred = ada.predict(Xt_test_norm)
forest_test_pred=forest.predict(Xt_test_norm)
data_test['Survived'] = forest_test_pred
data_test.drop(['Pclass','Fare','IsAlone','le_Deck','le_Title','le_Sex','le_Embarked','le_SexCat','le_FamilySize','le_AgeCat'], axis=1, inplace=True)
data_test.reset_index(drop=True)
final_data = data_test.to_csv('Titanic_Result.csv', index = True)
data_test.head()