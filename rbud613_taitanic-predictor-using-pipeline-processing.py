import numpy as np 

import pandas as pd 

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import roc_auc_score

import random

random.seed(123)

sns.set_style("darkgrid")

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

len_train = len(train)

train_y = train["Survived"]
plt.figure(figsize=(20,6))

train_group = train.groupby(['Pclass','Sex'])['Fare'].mean()

plt.subplot(121)

sns.heatmap(train_group.unstack("Pclass"), cmap='Blues', annot=True)

plt.title("Fare (Pclass vs Sex)")

train_group = train.groupby(['Pclass','Sex'])['Age'].mean()

plt.subplot(122)

sns.heatmap(train_group.unstack("Pclass"), cmap='Blues', annot=True)

plt.title("Age (Pclass vs Sex)")
from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm

lm = ols('Age~ C(Pclass) + C(Sex) * C(Embarked) + C(SibSp) + C(Parch)', pd.concat([train,test],axis=0)).fit()

anova_lm(lm,typ=2)
lm = ols('Fare~ C(Pclass) + C(Sex) + C(Embarked) + C(SibSp) + C(Parch)', pd.concat([train,test],axis=0)).fit()

anova_lm(lm,typ=2)
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'the Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]



class FirstTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, name):

        self.name = name

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        value = [i[0][0] for i in X[self.name]]

        X[self.name] = value

        return X

    

class ValueImputer(BaseEstimator, TransformerMixin):

    def __init__(self, name, value):

        self.value = value

        self.name = name

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X[self.name].fillna(self.value,inplace=True)

        return X



class GroupbyImputer(BaseEstimator, TransformerMixin):

    def __init__(self, name, group_by):

        self.name = name

        self.group_by = group_by

    def fit(self, X, y=None):

        self.value = X.groupby(self.group_by)[self.name]

        return self

    def transform(self, X):

        X[self.name] = self.value.apply(lambda x: x.fillna(x.median()))

        i=1

        while(X[self.name].isnull().sum()>0):

            k = X.groupby(self.group_by[:-i])[self.name]

            X[self.name] = (k.apply(lambda x: x.fillna(x.median())))

            i+=1

            if i == len(self.group_by):

                break

        return X



class MostTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, names,mapping):

        self.names = names

        self.mapping = mapping

    def fit(self, X, y=None):

        self.freq = X[self.names].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

        return self

    def transform(self, X, y=None):

        X[self.names + "_map"] = self.freq

        X.replace({self.names+"_map": self.mapping}, inplace=True)

        return X.drop("Name",axis=1)





class FrequentTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, names):

        self.names = names

    def fit(self, X, y=None):

        self.freq =  X.groupby(self.names)[self.names].transform('count')

        return self

    def transform(self, X, y=None):

        X[self.names + "_freq"] = self.freq

        return X.drop(self.names, axis=1)



imputer_pipeline = Pipeline([

        ('Cabin_imputer', ValueImputer(name = "Cabin",value="None")),

        ('Embarked_imputer', ValueImputer(name= "Embarked",value="S")),

        ('Fare_imputer', GroupbyImputer(name= "Fare",group_by = ['Pclass', 'Parch','SibSp'])),

        ('Age_imputer', GroupbyImputer(name= "Age",group_by = ['Pclass','Parch','Embarked'])),

    ])



transform_pipeline = Pipeline([

        ('Cabin', FirstTransformer(name= "Cabin")),

        ('Name_transformer', MostTransformer(names = "Name", mapping = mapping)),

        ('Ticket_transformer', FrequentTransformer(names= "Ticket")),

    ])
class TaitanicProcessing(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.y = y

        return self

    def transform(self, X):

        X["Sex"] = X["Sex"].map({"male":1, "female":0})

        X["Family"] = X["SibSp"] + X["Parch"] + 1

        X["Family"] = X["Family"].astype(str)

        X['Family'] = X['Family'].replace("1", 'Alone')

        X['Family'] = X['Family'].replace(["2","3","4"], 'Normal')

        X['Family'] = X['Family'].replace(["5","6"], 'Mid')

        X['Family'] = X['Family'].replace(["7","8","11"], 'Big')

        X["Fare"] = np.log1p(X["Fare"])

        X["Fare"] = MinMaxScaler().fit_transform(X['Fare'].values.reshape(-1, 1))

        X['Cabin'] = X['Cabin'].replace(['A', 'B', 'C','T'], 'ABCT')

        X['Cabin'] = X['Cabin'].replace(['D', 'E'], 'DE')

        X['Cabin'] = X['Cabin'].replace(['F', 'G'], 'FG')

        X['Age'] = pd.qcut(X['Age'], 10)

        X['Age'] = LabelEncoder().fit_transform(X['Age'])

        return X



class DummyCategory(BaseEstimator, TransformerMixin):

    def __init__(self, names):

        self.names = names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        encoded_fea = []

        for c in self.names:

            encoded = OneHotEncoder().fit_transform(X[c].values.reshape(-1, 1)).toarray()

            n = X[c].nunique()

            cols = ['{}_{}'.format(c, n) for n in range(1, n + 1)]

            encoded_df = pd.DataFrame(encoded, columns=cols)

            encoded_df.index = X.index

            encoded_fea.append(encoded_df)

            

        return pd.concat([X, *encoded_fea[:5]], axis=1)



process_pipeline = Pipeline([

        ("process", TaitanicProcessing()),

        ("cat_pipeline", DummyCategory(["Embarked", "Name_map","Cabin","Family"])),

    ])



full_pipeline =  Pipeline([

        ("imputer", imputer_pipeline),

        ("transform", transform_pipeline),

        ("process", process_pipeline),

    ])
df = pd.concat([train.drop("Survived",axis=1),test],axis=0,sort=False)

df.set_index("PassengerId",drop=True, inplace=True)

df = full_pipeline.fit_transform(df,train_y)

df.reset_index(drop=True,inplace=True)

df.info()
df.drop(["Embarked", "Name_map","Cabin","Family",'SibSp','Parch'],axis=1,inplace=True)

df.head()
train = df.iloc[:len_train,:].reset_index(drop=True)

test = df.iloc[len_train:,:].reset_index(drop=True)
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, ccp_alpha, x_data=None, y_data=None, n_splits=5, output='score'):

    score = 0

    kf = StratifiedKFold(n_splits=n_splits, random_state=5, shuffle=True)

    models = []

    for train_index, valid_index in kf.split(x_data, y_data):

        x_train, y_train = x_data.iloc[train_index], y_data[train_index]

        x_valid, y_valid = x_data.iloc[valid_index], y_data[valid_index]

                                           

        model = RandomForestClassifier(

            criterion='gini',

            max_features='auto',

            n_estimators = int(n_estimators), 

            max_depth = int(max_depth),

            min_samples_split = int(min_samples_split),

            min_samples_leaf = int(min_samples_leaf),

            ccp_alpha = ccp_alpha,

            random_state = 123,

            oob_score=True,

            n_jobs=-1

        )

        

        model.fit(x_train, y_train)

        models.append(model)

        

        pred = model.predict_proba(x_valid)[:, 1]

        true = y_valid

        score += roc_auc_score(true, pred)/n_splits

    

    if output == 'score':

        return score

    if output == 'model':

        return models



from functools import partial 

from bayes_opt import BayesianOptimization

func_fixed = partial(rf_cv, x_data=train, y_data=train_y, n_splits=5, output='score')

rf_ba = BayesianOptimization(

    func_fixed, 

    {

        'n_estimators': (1000, 2000),                        

        'max_depth' : (5,13),

        'min_samples_split' : (4,8),

        'min_samples_leaf' : (4,8),

        'ccp_alpha' : (0.0001, 0.01)

    }, 

    random_state=4321            

)

rf_ba.maximize(init_points=5, n_iter=20)
params = rf_ba.max['params']

rf_model = rf_cv(

    params['n_estimators'],

    params['max_depth'],

    params['min_samples_split'],

    params['min_samples_leaf'],

    params['ccp_alpha'],

    x_data=train, y_data=train_y, n_splits=5, output='model') 



importances = pd.DataFrame(np.zeros((train.shape[1], 5)), columns=['Fold_{}'.format(i) for i in range(1, 6)], index=train.columns)



preds = []



for i, model in enumerate(rf_model):

    importances.iloc[:, i] = model.feature_importances_

    pred = model.predict(test)

    preds.append(pred)

pred = np.mean(preds, axis=0)
importances['Mean_Importance'] = importances.mean(axis=1)

importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)



plt.figure(figsize=(15, 15))

s = sns.barplot(x=importances.index, y='Mean_Importance', data=importances)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.title('Random Forest Classifier Feature Importance', size=15)
y_pred = pred

y_pred[y_pred >= 0.5] = 1

y_pred = y_pred.astype(int)

submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

submission["Survived"] = y_pred

submission.to_csv("submission.csv",index = False)
rfc_model = RandomForestClassifier(criterion='gini',n_estimators=1800,max_depth=7, min_samples_split=6,min_samples_leaf=6,

                                           max_features='auto', oob_score=True, random_state=123,n_jobs=-1) 



oob = 0

probs = pd.DataFrame(np.zeros((len(test),10)), columns=['Fold_{}_Sur_{}'.format(i, j) for i in range(1, 6) for j in range(2)])





kf = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)



for fold, (trn_idx, val_idx) in enumerate(kf.split(train, train_y), 1):

    rfc_model.fit(train.iloc[trn_idx,:], train_y[trn_idx])

    probs.loc[:, ['Fold_{}_Sur_0'.format(fold),'Fold_{}_Sur_1'.format(fold)]] = rfc_model.predict_proba(test)

    oob += rfc_model.oob_score_ / 5

    print('Fold {} OOB : {}'.format(fold, rfc_model.oob_score_))

print('Average Score: {}'.format(oob))
survived =[col for col in probs.columns if col.endswith('Sur_1')]

probs['survived'] = probs[survived].mean(axis=1)

probs['unsurvived'] = probs.drop(columns=survived).mean(axis=1)

probs['pred'] = 0

sub = probs[probs['survived'] >= 0.5].index

probs.loc[sub, 'pred'] = 1



y_pred = probs['pred'].astype(int)
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

submission["Survived"] = y_pred

submission.to_csv("submission.csv",index = False)
submission.head()