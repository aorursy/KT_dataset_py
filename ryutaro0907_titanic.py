# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame

% matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.info()
train.columns
sns.countplot("Sex", data = train)
sns.countplot("Sex", data = train, hue ="Pclass")
sex = train['Sex'] 

age = train['Age']

survived = train['Survived']

Pclass = train["Pclass"]

change = {"male": 0, "female": 1}

sex = sex.map(change)

Embarked = train['Embarked']



ax=sns.swarmplot(x='Survived', y='Age', data = train)

ax.set_xticklabels(ax.get_xticklabels())
sns.distplot(age)
age.mean()
age.std()
Pclass.value_counts()



# People in 3rd class was more likely to die.



pd.crosstab(survived, Pclass)

# Visualize tendency of survival across passenger classes.

sns.countplot(x = Pclass, hue = survived)


sns.countplot(x = sex, hue = survived)



fig = sns.FacetGrid(train, hue = "Pclass", aspect = 4)

fig.map(sns.kdeplot, "Age", shade = True)

oldest = age.max()

fig.set(xlim=(0,oldest))

fig.add_legend()
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()



X = train[['Pclass', 'Age']]

y = survived



X[['Pclass', 'Age']] = scale.fit_transform(X[['Pclass', 'Age']].values)



print (X)



y.groupby(train.Pclass).mean()

#1st class people is more likely to survive.
y.groupby(train.Sex).mean()

#Women is more likely to survive. What is a ratio of men and women?

sex.value_counts()
#Again, we can see this plot and we can conclude the reason why female have such a high percentage of survival is that

#There are significant amount of dead case of male in 3rd class.

sns.countplot(x = Pclass, hue = sex)
Embarked.value_counts()
sns.countplot(x = Embarked, hue = Pclass)
pd.crosstab(survived, Embarked)







y.groupby(train.Embarked).mean()

train.head()
deck = train["Cabin"].dropna()

deck.head()
levels = []



for level in deck:

    levels.append(level[0])

 

levels
cabin_df = DataFrame(levels)

cabin_df.columns = ["Cabin"]

cabin_df

sns.countplot("Cabin", data = cabin_df, order = sorted(set(levels)))
train["Alone"] = train.Parch + train.SibSp

train["Alone"]
train["Alone"].loc[train["Alone"]> 0] = "With family"

train["Alone"].loc[train["Alone"]==0] = "Alone"

train.head()
sns.countplot("Alone", data = train)
train['Survivor'] = train.Survived.map({0:"No", 1:"Yes"})

sns.countplot("Survivor", data = train)

sns.factorplot("Pclass", "Survived", data = train, order =[1,2,3])
sns.countplot("Survivor", data = train, hue = 'Pclass')
generations = [10, 20,40,60,80]

sns.lmplot("Age", "Survived", hue = "Pclass",  data = train, hue_order =[1,2,3]

          ,x_bins = generations)

sns.lmplot("Age", "Survived", hue = "Sex",  data = train

          ,x_bins = generations)
train.tail()
d = {'male': 1, 'female': 0}

train['Sex'] = train['Sex'].map(d)
test.head()
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

import xgboost as xgb
from sklearn.linear_model import LogisticRegression

import sklearn as sk

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()



all_features = train[['Pclass', 'Sex', 'Age','SibSp','Parch']].values

all_classes = train['Survived'].values



feature_names =['Pclass', 'Sex','Age', 'SibSp','Parch']



X

from sklearn import preprocessing



scaler = preprocessing.StandardScaler()

all_features_scaled = scaler.fit_transform(all_features)

all_features_scaled
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential



def create_model():

    model = Sequential()

    #4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)

    model.add(Dense(4, input_dim=5, kernel_initializer='normal', activation='relu'))



    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))

    # Output layer with a binary classification (benign or malignant)

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model; rmsprop seemed to work best

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from sklearn.model_selection import cross_val_score

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



# Wrap our Keras model in an estimator compatible with scikit_learn

estimator = KerasClassifier(build_fn=create_model, epochs=1000, verbose=0)

# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others

cv_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=100)

cv_scores.mean()
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_all_features = my_imputer.fit_transform(all_features_scaled)

lr = LogisticRegression()

lr.fit(imputed_X_train , all_classes)

lr.score(imputed_all_features , all_classes)
