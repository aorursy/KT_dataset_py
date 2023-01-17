import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bird.csv')

df.fillna(0, inplace=True)

df.info()
sns.countplot(df.type)
from imblearn.under_sampling import AllKNN, RandomUnderSampler

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler 

from imblearn.pipeline import make_pipeline





from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, f1_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

from tqdm import tqdm



from sklearn.decomposition import PCA, KernelPCA



import warnings
warnings.filterwarnings("ignore")

x, y = df.drop(['id', 'type'], axis=1), df['type']





class Vanilla:

    def fit(self, x, y):

        return self

    def sample(self, x, y):

        return x, y

    def fit_sample(self, x, y):

        return x, y

    def __str__(self):

        return 'Vanilla'

print('Generating pipelines')

samplers = [Vanilla(), AllKNN(), RandomUnderSampler(), RandomOverSampler(), SMOTE()]

estimators = [GridSearchCV(estimator=RandomForestClassifier(),

                           param_grid={'max_depth':[2,3,4,5]}),

              #GaussianNB(),

              GridSearchCV(estimator=SVC(kernel='poly'),

                           param_grid={'C': [0.1, 0.5, 1, 5, 10]}),

              GridSearchCV(estimator=LogisticRegression(),

                           param_grid={'C': [0.1, 0.5, 1, 5, 10]}

                          )]

names = ['allknn', 'rus', 'ros', 'smote', 'adasyn']



pipelines = [make_pipeline(sampler, estimator) for estimator in estimators

             for sampler in samplers ]

print('Running Multiple CVs to generate performance data')

data = []

for run in range(10):

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    for pipe in tqdm(pipelines):

        pipe.fit(X_train, y_train)

        s, e = list(map(lambda x: x[0], pipe.steps))

        data.append((s, e, f1_score(y_test,

                                    pipe.predict(X_test),

                                    average='weighted')))
data =pd.DataFrame(data, columns=['Sampler', 'Estimator', 'f1'])

data.info()
plt.figure(figsize=(10, 7))

sns.barplot(x='Sampler', y='f1', data=data, ci=0)