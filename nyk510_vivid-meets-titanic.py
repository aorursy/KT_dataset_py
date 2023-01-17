# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install git+https://gitlab.com/nyker510/vivid
INPUT_DIR = '/kaggle/input/titanic/'

train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))



y = train_df.pop('Survived')
import re

from sklearn.preprocessing import StandardScaler

import numpy as np



from vivid.featureset import AbstractAtom

from vivid.featureset.molecules import create_molecule, MoleculeFeature
def get_title(name):

    if re.search(' ([A-Za-z]+)\.', name):

        return re.search(' ([A-Za-z]+)\.', name).group(1)

    return ""



class TitanicBasicAtom(AbstractAtom):

    def call(self, input_df, y=None):

        output_df = input_df.copy()

        output_df['Cabin'] = output_df['Cabin'].apply(lambda x: 1 if type(x) == str else 0)



        output_df['Age'] = output_df['Age'].fillna(-1).astype(int)



        output_df['Fare'] = output_df['Fare'].fillna(-1).astype(int)



        output_df['Sex'] = output_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



        output_df['Title'] = output_df['Name'].apply(get_title)

        output_df['Title'] = output_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        output_df['Title'] = output_df['Title'].replace('Mlle', 'Miss')

        output_df['Title'] = output_df['Title'].replace('Ms', 'Miss')

        output_df['Title'] = output_df['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        output_df['Title'] = output_df['Title'].map(title_mapping)

        output_df['Title'] = output_df['Title'].fillna(-1)



        output_df['Embarked'] = output_df['Embarked'].fillna('S')

        output_df['Embarked'] = output_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



        output_df.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)



        return output_df
TitanicBasicAtom().generate(train_df, y)
# create new molecule

basic_molecule = create_molecule(atoms=[TitanicBasicAtom()], name='basic')



# create feature

basic_feature = MoleculeFeature(basic_molecule, root_dir='/kaggle/working/')



# feature has fit method.

basic_feature.fit(train_df, y)
from vivid.out_of_fold.boosting import XGBoostClassifierOutOfFold, LGBMClassifierOutOfFold

from vivid.out_of_fold.boosting.block import create_boosting_seed_blocks

from vivid.out_of_fold.linear import LogisticOutOfFold

from vivid.out_of_fold.ensumble import RFClassifierFeatureOutOfFold

from vivid.out_of_fold.base import BaseOutOfFoldFeature

from vivid.core import MergeFeature, EnsembleFeature
class KaggleKernelMixin:

    def save_best_models(self, best_models):

        pass



class CustomLGBM(KaggleKernelMixin, LGBMClassifierOutOfFold):

    initial_params = {

        'n_estimators': 10000,

        'objective': 'binary',

        'feature_fraction': .9,

        'learning_rate': .05,

        'max_depth': 5,

        'num_leaves': 17

    }
class LogisticOptuna(KaggleKernelMixin, LogisticOutOfFold):

    initial_params = {

        'input_scaling': 'standard'

    }



class XGB(KaggleKernelMixin, XGBoostClassifierOutOfFold):

    pass



class SimpleLGBM(KaggleKernelMixin, LGBMClassifierOutOfFold):

    initial_params = {

        'n_estimators': 10000,

        'learning_rate': .05,

        'reg_lambda': 1.,

        'reg_alpha': 1.,

        'feature_fraction': .7,

        'max_depth': 3,

    }

    

class RF(KaggleKernelMixin, RFClassifierFeatureOutOfFold):

    initial_params = {'n_estimators': 125, 'max_features': 0.2, 'max_depth': 25, 'min_samples_leaf': 4, 'n_jobs': -1}
from sklearn.ensemble import ExtraTreesClassifier

from rgf.sklearn import RGFClassifier

from sklearn.linear_model import LogisticRegression
class ExtraTree(KaggleKernelMixin, BaseOutOfFoldFeature):

    model_class = ExtraTreesClassifier

    initial_params = {'n_estimators': 100, 'max_features': 0.5, 'max_depth': 18, 'min_samples_leaf': 4, 'n_jobs': -1}

    

class RGF(KaggleKernelMixin, BaseOutOfFoldFeature):

    model_class = RGFClassifier

    initial_params = {'algorithm': 'RGF_Sib', 'loss': 'Log'}

    

class Logistic(KaggleKernelMixin, BaseOutOfFoldFeature):

    model_class = LogisticRegression

    init_params = { 'input_scaling': 'standard' }
from keras import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from copy import deepcopy

from sklearn.base import ClassifierMixin
class NN(ClassifierMixin, KerasClassifier):

    def __call__(self, n_input):

        clf = Sequential()

        clf.add(Dense(12, input_dim=n_input, activation='relu'))

        clf.add(Dense(6, activation='relu'))

        clf.add(Dense(1, activation='sigmoid'))

        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return clf



    def fit(self, X, y):

        self.sk_params['n_input'] = X.shape[1]

        super().fit(X, y)

        return self
class KerasNN(KaggleKernelMixin, BaseOutOfFoldFeature):        

    model_class = NN

    initial_params = {

        'epochs': 100,

        'batch_size': 32,

    }
single_models = [

    Logistic(parent=basic_feature, name='logistic', add_init_param={ 'input_scaling': 'standard' }),

    XGB(parent=basic_feature, name='xgb'),

    SimpleLGBM(parent=basic_feature, name='lgbm'),

    ExtraTree(parent=basic_feature, name='extra'),

    RGF(parent=basic_feature, name='rgf'),

    RF(parent=basic_feature, name='rf'),

    KerasNN(parent=basic_feature, name='keras_nn')

]



ens = EnsembleFeature(single_models[:], name='ensumble', root_dir=basic_feature.root_dir)

single_models += [ens]



merged = MergeFeature([*single_models, basic_feature], name='merged', root_dir=basic_feature.root_dir)



stacking_models = [

    Logistic(parent=merged,name='logistic_stacked', add_init_param={ 'input_scaling': 'standard' }),

    SimpleLGBM(parent=merged, name='lgbm_stacked')

]
models = [

    *single_models,

    *stacking_models

]
oof_df = pd.DataFrame()

for m in models:

    df_i = m.fit(train_df, y)

    oof_df = pd.concat([oof_df, df_i], axis=1)
oof_df
from vivid.metrics import binary_metrics



score_df = None

for c in oof_df.columns:

    score = binary_metrics(y, oof_df[c])

    if score_df is None:

        score_df = score.rename(columns={ 'score': c })

    else:

        score_df[c] = score.values[:, 0]
score_df.T.sort_values('auc', ascending=False)
import seaborn as sns
sns.clustermap(data=oof_df)
from vivid.utils import timer
OUTPUT_DIR = '/kaggle/working/'
for m in models:

    with timer(logger=m.logger, format_str='{:.3f}[s]'):

        pred = m.predict(test_df)

    

    sub_df = pd.DataFrame()

    sub_df['Survived'] = np.round(pred.values[:, 0]).astype(np.int)

    sub_df['PassengerId'] = test_df['PassengerId']

    sub_df.to_csv(os.path.join(OUTPUT_DIR, m.name + '.csv'), index=False)