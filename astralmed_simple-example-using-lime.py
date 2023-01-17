import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# LIME

import lime.lime_tabular

from lime.explanation import Explanation



import warnings

warnings.filterwarnings('ignore')
# Load Titanic DataSet

df = pd.read_csv('../input/train.csv')

df.isnull().sum()  # check missing data.
# Remove Useless Attributes

df.drop(['PassengerId', 'Name', 'Pclass', 'SibSp',

         'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)



# Handle Missing Data

age_train_mean = df.groupby('Sex').Age.mean()

df.loc[df['Age'].isnull() & (df['Sex'] == 'male'),

       'Age'] = age_train_mean['male']

df.loc[df['Age'].isnull() & (df['Sex'] == 'female'),

       'Age'] = age_train_mean['female']



df.dropna(subset=['Embarked'], axis=0, inplace=True)



print(df.isnull().sum())
df.head()
sns.catplot(data=df, kind='violin', hue='Survived',

            x='Embarked', y='Age', col='Sex')
X_train = df.drop(['Survived'],  axis=1, inplace=False)

y_train = df.Survived
X_train_lbenc = X_train.copy()

cats = ['Sex','Embarked'] # not yet specified label encoded attributes.



cat_dic = {}  # also be used at LimeTabularExplainer's parameter.

cat_list = [] # also be used at OneHotEncoder, LimeTabularExplainer's parameter.



le = LabelEncoder()

for s in cats:

    i = X_train_lbenc.columns.get_loc(s)

    X_train_lbenc.loc[:,s] = le.fit_transform(X_train_lbenc[s])

    cat_dic[i] = le.classes_

    cat_list.append(i)



X_train_lbenc.head()
print(cat_list, '\n',  cat_dic) # check
# Non-categorical features are always stacked to the right of the matrix.

oe = OneHotEncoder(sparse=False, categorical_features=cat_list)

oe_fit = oe.fit(X_train_lbenc)

X_train_ohenc = oe_fit.transform(X_train_lbenc)

X_train_ohenc[:5, :]  # show 5 samples.
parameters = {

    'C': np.logspace(-5, 5, 10),

    'random_state': [0]

}



gs = GridSearchCV(

    LogisticRegression(),

    parameters,

    cv=5

)

gs.fit(X_train_ohenc, y_train)



print(gs.best_score_)

print(gs.best_params_)



model = LogisticRegression(**gs.best_params_)

model.fit(X_train_ohenc, y_train)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_lbenc.values,  # Label Encoded Numpy Format

                                                   feature_names = X_train_lbenc.columns,

                                                   class_names = [

                                                       'dead', 'survive' ], # 0,1,...

                                                   categorical_features = cat_list,

                                                   categorical_names = cat_dic,

                                                   mode = 'classification'

                                                   )
def pred_fn(x):

    return model.predict_proba(oe_fit.transform(x)).astype(float)
exp = explainer.explain_instance(X_train_lbenc.values[2, :],

                                 pred_fn,

                                 num_features=len(X_train_lbenc.columns)

                                 )

exp.show_in_notebook(show_all=False)