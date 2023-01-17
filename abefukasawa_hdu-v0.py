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
df.columns
df = pd.read_csv(

    '/kaggle/input/heart-disease-uci/heart.csv',

    dtype={

        'sex':'category'

        ''

    })
df.sex.value_counts()
df.info()
df.head()
df.describe().T
df.target.value_counts()
import seaborn as sns 

from matplotlib import pyplot as plt

%matplotlib inline
sns.pairplot(df, hue='target', diag_kind='KDE')
df.age.unique()
dt = pd.read_csv(

    '/kaggle/input/heart-disease-uci/heart.csv')

dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',

              'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 

              'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 

              'num_major_vessels', 'thalassemia', 'target']



dt['sex'][dt['sex'] == 0] = 'female'

dt['sex'][dt['sex'] == 1] = 'male'



dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'

dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'

dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'

dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'



dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'

dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'

dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'



dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 0] = 'no'

dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 1] = 'yes'



dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'

dt['st_slope'][dt['st_slope'] == 2] = 'flat'

dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'



dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'

dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'

dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'



dt['sex'] = dt['sex'].astype('object')

dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')

dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')

dt['rest_ecg'] = dt['rest_ecg'].astype('object')

dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')

dt['st_slope'] = dt['st_slope'].astype('object')

dt['thalassemia'] = dt['thalassemia'].astype('object')

dt.info()
plt.scatter(dt.age, dt.target)
sns.violinplot(x="target", y="age", hue="target",

                    data=dt, split=True)
sns.violinplot(x="target", y="resting_blood_pressure", hue="target",

                    data=dt, split=True)
sns.violinplot(x="target", y="cholesterol", hue="target",

                    data=dt, split=True)
sns.violinplot(x="target", y="max_heart_rate_achieved", hue="target",

                    data=dt, split=True)
sns.violinplot(x="target", y="st_depression", hue="target",

                    data=dt, split=True)
sns.violinplot(x="target", y="num_major_vessels", hue="target",

                    data=dt, split=True)
#sns.catplot(x="chest_pain_type", y="target", hue="chest_pain_type", kind="swarm", data=dt);
sns.catplot(x="sex", y="target", hue="sex", kind="bar", data=dt);
sns.catplot(x="chest_pain_type", y="target", hue="chest_pain_type", kind="bar", data=dt);
sns.catplot(x="fasting_blood_sugar", y="target", hue="chest_pain_type", kind="bar", data=dt);
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

X, y, = df.drop('target', 1), df['target']

model = RandomForestClassifier(max_depth=8)

model.fit(X, y)
estimator = model.estimators_[1]

feature_names = [i for i in X.columns]



y_str = y.astype('str')

y_str[y_str == '0'] = 'no disease'

y_str[y_str == '1'] = 'disease'

y_str = y_str.values
from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform

from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,

                              random_state=0)



distributions = dict(C=uniform(loc=0, scale=4),

                     penalty=['l2', 'l1'])



clf = RandomizedSearchCV(logistic, distributions, random_state=42)

search = clf.fit(X, y)

search.best_params_
model = RandomForestClassifier(n_jobs=-1)



distributions = dict(n_estimators=np.arange(20,150,10),

                     max_depth=np.arange(5,15,2),

                     min_samples_split=np.arange(5, 20, 2))



clf = RandomizedSearchCV(model, distributions, random_state=42)

search = clf.fit(X, y)

print(search.best_params_)

print(search.best_score_)
test_df = df.sample(frac=.1, )