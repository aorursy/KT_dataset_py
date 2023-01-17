import pandas as pd
import numpy as np
import re

# Plotly
from plotly.offline import init_notebook_mode, iplot
import cufflinks as cf

init_notebook_mode()
cf.go_offline()
df = pd.read_csv('../input/train.csv')
df.info()
from plotly.tools import FigureFactory as FF
iplot(FF.create_scatterplotmatrix(df[['Survived', 'Pclass', 'Age','Fare','Parch','SibSp']],width=800,height=700));        
def feature_extract(df):
    
    df['Age'] = df.Age.fillna(df.Age.median())
    df['Fare'] = df.Fare.fillna(df.Fare.median())
    df['Embarked'] = df['Embarked'].fillna('S')   
    df['Embarked'] = df.Embarked.map({'C':1,'Q':2,'S':3}).astype(int)
    df['isAlone'] = (df.Parch + df.SibSp).apply(lambda x : True if x == 0 else False)
    df['familySize'] = (df.Parch + df.SibSp + 1)
          
    
    # Get passenger title from name.
    df['Title'] = df['Name'].apply(lambda x : re.search(' ([A-Za-z]+)\.', x).group(1))
    df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss','Master':'Mr'})
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
                                      ,'Special')

    df = df.join(pd.get_dummies(df['Title'], prefix='Title', drop_first=True))
    df = df.join(pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True))
    df = df.join(pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True))
#     df = df.join(pd.get_dummies(df['cabin_'], prefix='cabin_', drop_first=True))
#     df = df.join(pd.get_dummies(df['Pclass'], prefix='Pclass', drop_first=True))
 
    col_drop = ['PassengerId','Name', 'Sex','Ticket', 'Cabin','SibSp','Parch','Embarked','Title']
    
    return df.drop(col_drop, axis=1)
feature_extract(pd.read_csv('../input/train.csv')).head()
%%time
X = feature_extract(pd.read_csv('../input/train.csv').drop('Survived', axis = 1))
y = pd.read_csv('../input/train.csv')['Survived']

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
param_grid = { 
    'n_estimators': np.arange(10,50,10),
    'max_depth': np.arange(10,50,10),
    "min_samples_leaf" : np.arange(2,10,1),
    'max_features': ['auto', 'sqrt', 'log2']
}
CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5 , scoring='accuracy')
CV_rfc.fit(X, y)

print (CV_rfc.best_params_)
print (CV_rfc.best_score_)
df_test = pd.read_csv('../input/test.csv')
# feature_extract(df_test).head()
Y_pred = CV_rfc.predict(feature_extract(df_test))
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].head()
#df_test[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)