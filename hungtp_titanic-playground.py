import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRFClassifier, XGBClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
X,y = train.drop('Survived',axis=1), train['Survived']
titanic = pd.concat([X,test]).reset_index(drop=True)
titanic.head()
titanic.info()
# Age, Fare, Embarked, Cabin have missing values

# I will drop Cabin becasuse it has too much missing values

titanic.drop('Cabin',axis=1,inplace=True)
# I will fill missing values of Age , Fare based on their meidan value

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
# I will fill Embarked missing values by the value which has the highest frequency counts

most_emb_fre_val = titanic['Embarked'].value_counts().index[0]

titanic['Embarked'] = titanic['Embarked'].fillna(most_emb_fre_val)
# I will not use ['PassengerId', 'Name', 'Ticket'] to train model because I think they are not useful

# I will drop these columns

titanic.drop(['PassengerId', 'Name', 'Ticket'],axis=1,inplace=True)
titanic.head()
# SibSp and Parch have the same meaning because they point to the number of family members

# So, I will make a column name Family = SibSp + Parch + 1 to represent the number of family members

# Then, I will drop SibSp and Parch

titanic['Family'] = titanic['SibSp'] + titanic['Parch'] + 1

titanic.drop(['SibSp','Parch'],axis=1,inplace=True)
titanic.head()
titanic_train = titanic[:len(train)]

findsomething_data = pd.concat([titanic_train,y],axis=1).reset_index(drop=True)
findsomething_data.head()
sns.countplot(x='Sex',hue='Survived',data=findsomething_data);
sns.distplot(findsomething_data['Age']);
sns.countplot(x='Embarked',data=findsomething_data,hue='Survived');
sns.heatmap(findsomething_data.corr(),annot=True);
class ColumnsSelection(BaseEstimator,TransformerMixin):

    def __init__(self,attribute_names):

        self.attribute_names = attribute_names

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        return X[self.attribute_names].values
titanic.info()
pip_discretizer = Pipeline([('family',ColumnsSelection(['Family'])),

                            ('discretizer',KBinsDiscretizer(n_bins=3,encode='onehot'))])
pip_text_onehot = Pipeline([('onehot_cols',ColumnsSelection(['Embarked','Pclass','Sex'])),

                            ('onehot',OneHotEncoder())])
pip_std_scale = Pipeline([('num_cols',ColumnsSelection(['Age','Fare'])),

                          ('std_scale',StandardScaler())])
pip_preprocess = FeatureUnion(transformer_list=[('discre',pip_discretizer),

                                                ('onehot',pip_text_onehot),

                                                ('stdscale',pip_std_scale)])
titanic_processed = pip_preprocess.fit_transform(titanic)

train_pre , test_pre = titanic_processed[:len(train)], titanic_processed[len(train):]
log = LogisticRegression()
ests = [LogisticRegression(n_jobs=-1),DecisionTreeClassifier(),

       RandomForestClassifier(n_jobs=-1),SVC(),

       XGBClassifier(n_jobs=-1),XGBRFClassifier(n_jobs=-1)]

estt_name = [e.__class__.__name__ for e in ests]
data = []

for est in ests:

    print("========== train with: ", est.__class__.__name__ , " =============")

    scores = cross_val_score(est,train_pre,y,cv=10,scoring='accuracy')

    mean_score = scores.mean()

    data.append(np.append(scores,mean_score))

    print("========== end train: ", est.__class__.__name__ , " =============")
df = pd.DataFrame(data=data,columns=[*['Fold'+str(i) for i in range(10)],'Mean'], index=estt_name)
df
xgb = XGBClassifier(n_jobs=-1)
xgb.fit(train_pre,y)
pre = xgb.predict(test_pre)
sample = pd.read_csv('../input/titanic/gender_submission.csv')
sample['Survived'] = pre
sample.to_csv('submit_playground_1.csv',index=False)