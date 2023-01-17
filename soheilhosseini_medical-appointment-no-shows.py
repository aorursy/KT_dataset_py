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
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



from scipy import stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression



from sklearn.linear_model import Ridge



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier







from sklearn.preprocessing import OneHotEncoder



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import cross_validate

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC



from imblearn.over_sampling import SMOTE

from collections import Counter

noshow = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
noshow_2 = noshow.copy()
#04 - DataMarcacaoConsulta

# The day of the actuall appointment, when they have to visit the doctor.

# 05 - DataAgendamento

# The day someone called or registered the appointment, this is before appointment of course.



noshow.head()
#There is no missing value.

noshow.info()
noshow.describe()
noshow.describe(include='O')
#Remove negative Age from Dataset

noshow = noshow[noshow['Age']>0]
# Let’s check if there are duplicated data



np.sum(noshow.duplicated())
#the distribution of “No-Show” and “Shop-up” cases

sns.countplot(x='No-show', data=noshow)

plt.show()
plt.figure(figsize=(6,6))

sns.boxplot(x="No-show", y="Age", data=noshow, palette='plasma')

plt.show()
#??? check this part 



plt.figure(figsize=(7,7))

sns.countplot(noshow['Age'],color='gray')

plt.xticks(rotation=90)



plt.show()
sns.countplot(x='Gender', data=noshow, palette = 'plasma')

plt.show()
ct = pd.crosstab(noshow.Gender, noshow['No-show'])

ct.plot.bar(stacked=True)

plt.show()
noshow['Scheduled_Day'] = pd.to_datetime(noshow['ScheduledDay'])

noshow['appointment'] = pd.to_datetime(noshow['AppointmentDay'])

noshow['day_of_week'] = noshow['appointment'].dt.day_name()
sns.countplot(x='day_of_week', data=noshow, palette = 'plasma')

plt.show()
noshow.head()
noshow.columns
noshow.head()
#Lag between Scheduled day and AppointmentDay



noshow['Lag'] = abs((noshow['appointment'] - noshow['Scheduled_Day']).dt.days)

noshow.head()


noshow.drop(columns = ['PatientId', 'AppointmentID', 'ScheduledDay','AppointmentDay','appointment','Scheduled_Day'], inplace=True)
#Correlation Matrix

fig = plt.figure(figsize=[8,8])

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(noshow.corr(), annot = True, square=True,linecolor='white',cmap='jet' )

plt.show()
y = noshow['No-show']

y.replace('No',0,inplace=True)

y.replace('Yes',1,inplace=True)
noshow.head()
noshow_dummy = pd.get_dummies(data=noshow.drop(columns='No-show'), columns= ['Neighbourhood','Gender','day_of_week'], drop_first=True)
noshow_dummy.head()
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

selector = SelectFromModel(estimator=RandomForestRegressor(n_jobs=-1, n_estimators=100)).fit(noshow_dummy, y)

noshow_dummy_robust = selector.transform(noshow_dummy)

noshow_dummy_robust = RobustScaler().fit_transform(noshow_dummy_robust)

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

noshow_dummy_robust = minmax.fit_transform(noshow_dummy_robust)
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(noshow_dummy_robust, y)

print('Resampled dataset shape %s' % Counter(y_res))

model ={'Decision Tree':DecisionTreeClassifier(max_depth=2,min_samples_leaf =10, random_state=0),

       'Random Forest Classifier': RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0),

       'Logistic Regression': LogisticRegression(random_state=0),

       'Gaussian Naive': GaussianNB(),

       'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, max_depth=2, random_state=0)}



for keys, items in model.items():

    cv_results = cross_validate(items, X_res, y_res, cv=5, scoring=('r2', 'f1','precision','recall','roc_auc'))

    print("keys" + "  " + str(keys),"\n")

    print("Recall:  ", cv_results['test_recall'])

    print("Precision:  ", cv_results['test_precision'],"\n")

    print("AUC:  ", cv_results['test_roc_auc'])

    print("max AUC:  ", max(cv_results['test_roc_auc']),"\n")

    print("average AUC:  ", np.mean(cv_results['test_roc_auc']))


