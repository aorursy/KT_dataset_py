import pandas as pd
import numpy as np
from sklearn import svm, preprocessing

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.layouts import row, widgetbox
output_notebook()
def DataFirstLoad(filename):
    df=pd.DataFrame.from_csv(filename, header=0)
    
    df['Age']=df.apply(lambda s: -1 if np.isnan(s['Age']) else s['Age'], axis=1)
    df['Parch']=df.apply(lambda s: -1 if np.isnan(s['Parch']) else s['Parch'], axis=1)
    df['SibSp']=df.apply(lambda s: -1 if np.isnan(s['SibSp']) else s['SibSp'], axis=1)
    df['Fare']=df.apply(lambda s: -1 if np.isnan(s['Fare']) else s['Fare'], axis=1)
    
    df['AgeRange']=pd.cut(df['Age'], 4)
    df['FamilySize']=df.apply(lambda s: s['Parch']+s['SibSp']+1, axis=1)
    df['HasSibSp']=df.apply(lambda s: 1 if s['SibSp']>0 else 0, axis=1)
    df['HasParch']=df.apply(lambda s: 1 if s['Parch']>0 else 0, axis=1)
    
    return df

df_train=DataFirstLoad('../input/train.csv')

print(df_train['Sex'].value_counts(dropna=False))
print(df_train['Embarked'].value_counts(dropna=False))
print(df_train[['Age']].apply(lambda s: 1 if s['Age']>=0 else s['Age'], axis=1).value_counts(dropna=False))
print(df_train[['Fare']].apply(lambda s: 1 if s['Fare']>=0 else s['Fare'], axis=1).value_counts(dropna=False))
f = {'Age':['min','max'], 'Survived':['count']}
df_train[['Survived','Age','AgeRange']].groupby(['AgeRange'], as_index=False).agg(f)
age_ref=(df_train[df_train['Age'] != -1])[['Pclass', 'Sex', 'Embarked', 'HasSibSp', 'HasParch', 'Age']].groupby(['Pclass', 'Sex', 'Embarked', 'HasSibSp', 'HasParch'], as_index=False).mean()


def DefineMissedAge(pclass, sex, embarked, hassibsp, hasparch):
    return age_ref.loc[(age_ref['Pclass'] == pclass) 
                     & (age_ref['Sex'] == sex)
                     & (age_ref['Embarked'] == embarked)
                     & (age_ref['HasSibSp'] == hassibsp)
                     & (age_ref['HasParch'] == hasparch)
                    ].Age.mean()


DefineMissedAge(1, 'male', 'S', 1, 1)
def DataPrep(filename):
    df=pd.DataFrame.from_csv(filename, header=0)
    
    #Define Age
    df['HasSibSp']=df.apply(lambda s: 1 if s['SibSp']>0 else 0, axis=1)
    df['HasParch']=df.apply(lambda s: 1 if s['Parch']>0 else 0, axis=1)
    
    df['Age']=df.apply(lambda s:
        DefineMissedAge(s['Pclass'], s['Sex'], s['Embarked'], s['HasSibSp'], s['HasParch']) 
                       if np.isnan(s['Age']) else s['Age'], axis=1)
    

    df['Parch']=df.apply(lambda s: -1 if np.isnan(s['Parch']) else s['Parch'], axis=1)
    df['SibSp']=df.apply(lambda s: -1 if np.isnan(s['SibSp']) else s['SibSp'], axis=1)
    df['Fare']=df.apply(lambda s: -1 if np.isnan(s['Fare']) else s['Fare'], axis=1)
    
    
    df['Gender']=df.apply(lambda s: 1 if s['Sex']=='male' else 0, axis=1)
    df['FamilySize']=df.apply(lambda s: s['Parch']+s['SibSp']+1, axis=1)
    df['isAlone']=df.apply(lambda s: 1 if s['Parch']+s['SibSp']+1==1 else 0, axis=1)
    df['EmbCode']=df.apply(lambda s: 2 if s['Embarked']=='S' else 1 if s['Embarked']=='C' else 0 if s['Embarked']=='Q' else 2, axis=1)
    df['FareRange']=pd.qcut(df['Fare'], 10, labels=[0, 1, 2, 3,4,5,6,7,8,9])
    
    df['AgeRange']=pd.cut(df['Age'], 10, labels=[0, 1, 2, 3,4,5,6,7,8,9])

    df_prepared=df[['Pclass','Gender','FamilySize','isAlone', 'Age', 'EmbCode', 'AgeRange','FareRange','HasSibSp','HasParch']].apply(pd.to_numeric, axis=1)
    x = df_prepared.as_matrix()#.astype(np.integer)
    x = preprocessing.scale(x)#without this we have about 60% right answers, after 72%
    return [x,df,df_prepared]

[x, df_train, df_prepared]=DataPrep('../input/train.csv')

f = {'Age':['min','max'], 'Survived':['count']}
df_train[['Survived','Age','AgeRange']].groupby(['AgeRange'], as_index=False).agg(f)
y=df_train[['Survived']].apply(pd.to_numeric)
y = y.as_matrix().astype(np.integer).ravel()


# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001, C=40)

# We learn the digits on the first half of the digits
classifier.fit(x, y)
def SirvivePosibility(groupvar):
    ds=ColumnDataSource(df_train[['Survived',groupvar]].groupby([groupvar], as_index=False).mean())
    # create a new plot with a title and axis labels
    p = figure(title=groupvar, x_axis_label=groupvar, y_axis_label='Survived %', plot_width=150, plot_height=150)
    
    # add a line renderer with legend and line thickness
    p.line(source=ds, x=groupvar, y='Survived',  line_width=2)
    p.toolbar.logo = None
    p.toolbar_location = None
    return p

# show the results
show(row(SirvivePosibility('Pclass'), 
         SirvivePosibility('Gender'),
         SirvivePosibility('FamilySize'),
         SirvivePosibility('isAlone'),
         SirvivePosibility('EmbCode'),
         SirvivePosibility('AgeRange'),
         SirvivePosibility('FareRange'),
        ))
[x_test, df_test, df_prepared]=DataPrep('../input/test.csv')
#x_test = DataPrep('test.csv')[0]

predicted = classifier.predict(x_test)
df2csv=pd.DataFrame(data={'PassengerId': df_test.index, 'Survived': predicted})

df2csv.to_csv('tested.csv', sep=',', index=False)
from sklearn.grid_search import GridSearchCV
from statistics import mean, median

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series(dict(params.items() | d.items()))
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                for k in self.keys
                for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC

models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = { 
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32, 64], 'learning_rate': [0.6, 0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [40, 50, 70, 80, 100, 150], 'gamma': [0.01, 0.001, 0.0001]},
    ]
}
helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(x, y, scoring='f1', n_jobs=-1)
helper1.score_summary(sort_by='min_score')



