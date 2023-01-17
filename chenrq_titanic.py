import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
data_train = pd.read_csv('../input/train.csv')
data_train
data_train.info()
data_train.describe()
sns.boxplot(x='Survived',y='Age',data=data_train)
sns.boxplot(x='Survived',y='Fare',data=data_train)
sns.barplot(x='Sex',y='Survived',data=data_train)
sns.barplot(x='Embarked',y='Survived',hue='Sex',data=data_train)
sns.barplot(x='Pclass',y='Survived',data=data_train)
sns.barplot(x='SibSp',y='Survived',data=data_train)
sns.barplot(x='Parch',y='Survived',data=data_train)
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
data_train['Name_length'] = data_train['Name'].apply(len)
name_length = data_train[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
print(name_length)
sns.barplot(x='Name_length', y='Survived', data=name_length)
corrmat = data_train.corr() #correlation matrix
sns.heatmap(corrmat, vmax=.8, annot=True, square=True);
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
data_train = data_train.drop((missing_data[missing_data['Total'] > 177]).index,1)
data_train
data_train.loc[data_train['Embarked'].isnull()].index
data_train.loc[data_train.Embarked.isnull(), 'Embarked'] = 'S'
data_train['FamilySize'] = data_train.Parch + data_train.SibSp
data_train
data_train.drop('Name', axis=1, inplace=True)
data_train
data_train.drop(['Parch','SibSp'], axis=1, inplace=True)
data_train
data_train.drop(['Ticket'], axis=1, inplace=True)
data_train
from sklearn.ensemble import RandomForestRegressor
# 把已有的数值型特征取出来丢进Random Forest Regressor中
age_df = data_train[['Age','Fare', 'FamilySize', 'Pclass', 'Name_length']]

# 乘客分成已知年龄和未知年龄两部分
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()

# y即目标年龄
y = known_age[:, 0]

# X即特征属性值
X = known_age[:, 1:]

# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X, y)

# 用得到的模型进行未知年龄结果预测
predictedAges = rfr.predict(unknown_age[:, 1:])

data_train.loc[(data_train.Age.isnull()),'Age'] = predictedAges
data_train
data_train
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

data_train = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data_train.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
data_train
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(data_train['Age'].values.reshape(-1, 1))
data_train['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(data_train['Fare'].values.reshape(-1, 1))
data_train['Fare_scaled'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1, 1), fare_scale_param)
namelength_scale_param = scaler.fit(data_train['Name_length'].values.reshape(-1, 1))
data_train['Name_length_scaled'] = scaler.fit_transform(data_train['Name_length'].values.reshape(-1, 1), namelength_scale_param)
familysize_scale_param = scaler.fit(data_train['FamilySize'].values.reshape(-1, 1))
data_train['FamilySize_scaled'] = scaler.fit_transform(data_train['FamilySize'].values.reshape(-1, 1), familysize_scale_param)
data_train.drop(['Age', 'Fare', 'Name_length', 'FamilySize'], axis=1, inplace=True)
data_train
def classifier_cv(model,X,y):
    classifier_loss = cross_validation.cross_val_score(model, X, y, cv=10)
    return classifier_loss
from sklearn import linear_model, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age_.*|Fare_.*|Name_.*|FamilySize_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

models = [linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-1, max_iter=100), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=25), GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=200), SVC(C=1, degree=2, kernel='rbf', probability=True)]
names = ["LR", "KNN", "RF", "GBC", "SVC"]
for name, model in zip(names, models):
    clf = model.fit(X, y)
    score = classifier_cv(clf, X, y)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X,y)
        print(grid_search.best_params_, grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params','std_test_score','mean_test_score']])
grid(linear_model.LogisticRegression()).grid_get(X, y, {'penalty':['l1', 'l2'], 'tol':[1e-1, 1e-2, 1e-3], 'max_iter':[100, 1000, 10000]})
grid(KNeighborsClassifier()).grid_get(X, y, {'n_neighbors':[3, 4, 5, 6, 7]})
grid(RandomForestClassifier()).grid_get(X, y, {'n_estimators':[5, 10, 15, 20, 25, 30]})
grid(GradientBoostingClassifier()).grid_get(X, y, {'n_estimators':[150, 200, 250], 'learning_rate':[1, 1e-1, 1e-2, 1e-3], 'max_depth':[2, 3, 4, 5]})
grid(SVC(probability=True)).grid_get(X, y, {'C':[0.5, 0.75, 1, 1.25], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'degree':[2, 3, 4]})
data_test = pd.read_csv('../input/test.csv')
data_test.loc[data_test.Embarked.isnull(), 'Embarked'] = 'S'
data_test['FamilySize'] = data_test.Parch + data_test.SibSp
data_test['Name_length'] = data_test['Name'].apply(len)
data_test.drop(['Cabin', 'Name', 'Parch', 'SibSp', 'Ticket'], axis=1, inplace=True)
tmp_df = data_test[['Age','Fare', 'FamilySize', 'Pclass', 'Name_length']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
Z = null_age[:, 1:]
predictedAges = rfr.predict(Z)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
data_test = pd.concat([data_test, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data_test.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
data_test['Age_scaled'] = scaler.transform(data_test['Age'].values.reshape(-1, 1), age_scale_param)
data_test['Fare_scaled'] = scaler.transform(data_test['Fare'].values.reshape(-1, 1), fare_scale_param)
data_test['Name_length_scaled'] = scaler.transform(data_test['Name_length'].values.reshape(-1, 1), namelength_scale_param)
data_test['FamilySize_scaled'] = scaler.transform(data_test['FamilySize'].values.reshape(-1, 1), familysize_scale_param)
data_test.loc[data_test.Fare_scaled.isnull(), 'Fare_scaled'] = data_train.Fare_scaled.mean()
data_test.drop(['Age', 'Fare', 'Name_length', 'FamilySize'], axis=1, inplace=True)
data_test
test_df = data_test.filter(regex='Age_.*|Fare_.*|Name_.*|FamilySize_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test_df.as_matrix()

models = [linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-1, max_iter=100), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=25), GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=200), SVC(C=1, degree=2, kernel='rbf', probability=True)]
names = ["LR", "KNN", "RF", "GBC", "SVC"]
for name, model in zip(names, models):
    clf = model.fit(X, y)
    score = classifier_cv(clf, X, y)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))
    predictions = clf.predict(test_np)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("./"+name+"_predictions.csv", index=False)

'''
predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("./predictions.csv", index=False)
'''
# pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone, BaseEstimator
from sklearn.metrics import classification_report
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w
len(models)
w=[0.1, 0.2, 0.1, 0.2, 0.4]
weight_avg = AverageWeight(mod = models, weight = w)
cross_validation.cross_val_score(weight_avg, X, y, cv=2)
from sklearn.preprocessing import Imputer
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
stack_model = stacking(mod=models,meta_model=SVC(C=1, degree=2, kernel='rbf', probability=True))
stack_model
stack_score = cross_validation.cross_val_score(stack_model, X, y, cv=5)
np.mean(stack_score)
X_train_stack, X_test_stack = stack_model.get_oof(X, y, test_np)
X_train_stack.shape
X_test_stack.shape
X_train_add = np.hstack((X, X_train_stack))
X_train_add.shape
X_test_add = np.hstack((test_np,X_test_stack))
X_test_add.shape
m_stack_score = cross_validation.cross_val_score(stack_model, X_train_add, y, cv=5)
np.mean(m_stack_score)