import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



train['dataset'] = 'train'

test['dataset'] = 'test'



data = pd.concat([train,test])
data
ds = data.groupby(['Sex', 'dataset'])['PassengerId'].count().reset_index()



ds.columns = ['Sex', 'dataset', 'count']
fig = px.bar(

    ds,

    x='Sex',

    y='count',

    color='dataset',

    barmode='group',

    orientation='v',

    title='Sex train/test count',

    width=600,

    height=500

)



fig.show()
ds = data.groupby(['Pclass', 'dataset'])['PassengerId'].count().reset_index()



ds.columns = ['Pclass', 'dataset', 'count']



fig = px.bar(

    ds,

    x='Pclass',

    y='count',

    color='dataset',

    barmode='group',

    orientation='v',

    title='Pclass train/test count',

    width=600,

    height=500

)



fig.show()
ds = data[data.dataset == 'train'].groupby(['Pclass','Survived'])['PassengerId'].count().reset_index()



ds.columns = ['Pclass', 'Survived', 'count']



fig = px.bar(

   ds,

    x='Pclass',

    y='count',

    color='Survived',

    orientation='v',

    title='Pclass Survived count',

    width=600,

    height=500

)



fig.show()
fig = px.histogram(data, x='Age', color='dataset')



fig.show()
fig = px.histogram(data[data.dataset == 'train'], x='Age', color='Survived', facet_col='Pclass', facet_row='Sex',

                  category_orders={'Pclass': [1,2,3]})



fig.show()
fig = px.histogram(data[data.dataset == 'train'], x='Fare', color='Survived', facet_col='Sex',

                  category_orders={'Pclass': [1,2,3]})



fig.show()




fig = px.box(data[data.dataset == 'train'], x='Sex', y='Age', points='all', color='Survived')



fig.show()
data
data.isna().sum()
data['Age'].fillna(value=round(data.Age.mean()), inplace=True)
data.Cabin.fillna(value='U', inplace=True)
data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
normal_titles = ['Mr','Miss','Mrs','Master']



data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



data['Title'] = [row.replace(row,'Rare') if row not in normal_titles else row for row in data['Title']]
data.head()
data.Fare.fillna(value=round(data.Fare.mean()), inplace=True)
data.dropna(subset=['Embarked'], how='any', inplace=True)
y = data[data.dataset== 'train']['Survived']
cat_features = ['Sex', 'SibSp', 'Parch', 'Cabin', 'Title', 'Embarked', 'Pclass']

num_features = ['Age', 'Fare']



#y = train['Survived']

final_ft = pd.get_dummies(data[num_features + cat_features])



X = final_ft[:len(y)]

test = final_ft[len(y):]
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import StackingClassifier







clfs = [CatBoostClassifier(verbose=False), XGBClassifier(), LGBMClassifier(), RandomForestClassifier()]



for c in clfs:

    c.fit(X,y)
from sklearn.metrics import accuracy_score, auc,roc_auc_score, recall_score, log_loss, roc_curve, f1_score,precision_score





def get_scores(clfs):

    

    metrics = pd.DataFrame([], columns=['Name','Accuracy','AUC Score','Precision','Recall','F1-Score','Logloss'])

    



    for cls in clfs:

        stats = {}

        prediction = cls.predict(X)

        fpr, tpr, thresholds = roc_curve(y, prediction, pos_label=1)

        stats.update({'Accuracy': accuracy_score(y, prediction),

                  'Name': type(cls).__name__ ,

                 'Recall' : recall_score(y, prediction),

                 'F1-Score': f1_score(y, prediction),

                 'AUC Score': roc_auc_score(y, prediction),

                 'Logloss': log_loss(y,prediction),

                 'Precision': precision_score(y,prediction)})

        metrics = metrics.append(stats, ignore_index=True)

    return metrics



get_scores(clfs)
from sklearn.model_selection import KFold

from mlxtend.classifier import StackingCVClassifier



kfolds = KFold(n_splits=10, shuffle=True, random_state=42)





stack = StackingCVClassifier(classifiers=clfs,

                            shuffle=False,

                            use_probas=False,

                            cv=kfolds,

                            meta_classifier=clfs[3])

stack.fit(X,y)
clfs.append(stack)



get_scores(clfs)
from sklearn.metrics import plot_confusion_matrix



def confusion_matrix(clfs):

    plt.style.use('default')

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))



    for cls, ax in zip(clfs, axes.flatten()):

        plot_confusion_matrix(cls, 

                          X, 

                          y, 

                          ax=ax, 

                          cmap='Reds',

                         display_labels=y)

    ax.title.set_text(type(cls).__name__)

    plt.tight_layout()  

    plt.show()



    

confusion_matrix(clfs)
sub = stack.predict(test)
output_stacl = pd.DataFrame({'PassengerId': data[data.dataset == 'test']['PassengerId'],

                            'Survived': sub})
output_stacl['Survived'] = output_stacl['Survived'].astype('int32')
output_stacl.to_csv('output_2.csv', index=False)