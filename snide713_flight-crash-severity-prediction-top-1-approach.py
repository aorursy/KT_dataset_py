from IPython.display import Image

Image(filename= "../input/feature-decp-img/Feature_Description.png", height=700, width=700)
%load_ext autoreload

%autoreload 2



%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

import matplotlib as mpl



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, learning_curve, ShuffleSplit

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import VotingClassifier, IsolationForest

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix



plt.style.use('seaborn-whitegrid')

plt.rcParams['lines.linewidth'] = 2

plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['text.color'] = 'black'

plt.rcParams['axes.labelcolor']= 'black'

plt.rcParams['xtick.color'] = 'black'

plt.rcParams['ytick.color'] = 'black'

plt.rcParams['font.size']=12
train = pd.read_csv('../input/data-for-flight-crash-severity-prediction/train.csv')

test = pd.read_csv('../input/data-for-flight-crash-severity-prediction/test.csv')



train.head(10)
train.info()
train.describe(include='all')
X = train.drop(columns=['Accident_ID', 'Severity'])

y = train.Severity

y.map({'Minor_Damage_And_Injuries' : 0, 'Significant_Damage_And_Serious_Injuries' : 1, 

      'Significant_Damage_And_Fatalities' : 2,  'Highly_Fatal_And_Damaging' : 3})
print('Percentage of each class in Target Variable \n')

print((train.Severity.value_counts()/len(train))*100)
rfc = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=10, max_features=0.5 , bootstrap=False)

cross_val_score(rfc, X, y, cv=4, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
train_sizes_abs, train_scores, val_scores = learning_curve(rfc, X, y, train_sizes=np.arange(0.3, 1, 0.05), n_jobs=-1, cv=4, 

                                                           scoring='f1_weighted', random_state=2, verbose=1)

fig, axes = plt.subplots(2, 2, figsize=(15,15))



axes[0, 0].plot(train_sizes_abs, train_scores[:, 0], label='Train Score')

axes[0, 0].plot(train_sizes_abs, val_scores[:, 0], label='Val Score')

axes[0, 1].plot(train_sizes_abs, train_scores[:, 1], label='Train Score')

axes[0, 1].plot(train_sizes_abs, val_scores[:, 1], label='Val Score')

axes[1, 0].plot(train_sizes_abs, train_scores[:, 2], label='Train Score')

axes[1, 0].plot(train_sizes_abs, val_scores[:, 2], label='Val Score')

axes[1, 1].plot(train_sizes_abs, train_scores[:, 3], label='Train Score')

axes[1, 1].plot(train_sizes_abs, val_scores[:, 3], label='Val Score')



axes[0, 0].legend(), axes[1, 0].legend(), axes[0, 1].legend(), axes[1, 1].legend()

axes[0, 0].set_xlabel("No.of Examples Used"), axes[0, 0].set_ylabel("Score")

axes[0, 1].set_xlabel("No.of Examples Used"), axes[0, 1].set_ylabel("Score")

axes[1, 0].set_xlabel("No.of Examples Used"), axes[1, 0].set_ylabel("Score")

axes[1, 1].set_xlabel("No.of Examples Used"), axes[1, 1].set_ylabel("Score")
f, ax = plt.subplots(5, 2, figsize=(15,15))



for i in range(5):

    sns.distplot(X.iloc[:, i], color="b", bins=20, hist_kws=dict(alpha=1), hist=False, label='Train', ax=ax[i, 0] )

    sns.distplot(test.iloc[:, i], color="r", bins=20, hist_kws=dict(alpha=1), hist=False, label='Test', ax=ax[i, 0])

    plt.legend()

for i in range(5):

    sns.distplot(X.iloc[:, i+5], color="b", bins=20, hist_kws=dict(alpha=1), hist=False, label='Train', ax=ax[i, 1] )

    sns.distplot(test.iloc[:, i+5], color="r", bins=20, hist_kws=dict(alpha=1), hist=False, label='Test', ax=ax[i, 1])

    plt.legend()
f, ax = plt.subplots(1, 1, figsize=(11,11))

sns.barplot(X.Violations, y, ax=ax)

plt.legend()
sns.pairplot(train, hue="Severity")
y = train.Severity

y = y.str.replace('Minor_Damage_And_Injuries', '0').str.replace('Significant_Damage_And_Serious_Injuries', '1').str.replace('Significant_Damage_And_Fatalities', '2').str.replace('Highly_Fatal_And_Damaging', '3').astype(int)

corr = pd.concat(objs=[X, y], axis=1).corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
f,ax=plt.subplots(1,2,figsize=(18,8))

train['Accident_Type_Code'].value_counts().plot.bar(color=['grey','orange','blue'],ax=ax[0])

ax[0].set_title('Accident_Type_Code')

ax[0].set_ylabel('Count')

sns.countplot('Accident_Type_Code',hue='Severity',data=train,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
X_test = test.drop(columns=['Accident_ID'])

concat = pd.concat(objs=[X, X_test], axis=0)

concat['Type_7_acc'] = (concat.Accident_Type_Code == 7)

concat['More_than_2_violations'] = (concat.Violations>2)

concat['Safety_score/Days_since_inspection'] = concat.Safety_Score/(concat.Days_Since_Inspection)

concat.drop(columns=['Violations', 'Max_Elevation', 'Adverse_Weather_Metric', 

                     'Turbulence_In_gforces', 'Cabin_Temperature', 'Accident_Type_Code'], inplace=True)



ss = StandardScaler()

concat = ss.fit_transform(concat)



X, X_test = concat[:len(train), :], concat[len(train):, :]
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=10, max_features=0.5 , bootstrap=False)

cross_val_score(rfc, X, y, cv=4, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
nnc = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=300,activation = 'relu',solver='adam', 

                           random_state=1, batch_size=50)

cross_val_score(nnc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
knc = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

cross_val_score(knc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
lgbmc = LGBMClassifier(random_state=2, n_estimators=100, colsample_bytree=0.6, 

                       max_depth=10, learning_rate=0.5, boosting_type='gbdt')

cross_val_score(lgbmc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
xgbc = XGBClassifier(seed=7, n_jobs=-1, n_estimators=900, random_state=0, max_depth=7, learning_rate=0.7)

cross_val_score(xgbc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[('rfc', rfc), ('xgbc', xgbc), ('knc', knc),

                                        ('nnc', nnc), ('lgbmc', lgbmc)],

                                         voting='soft', n_jobs=-1)

cross_val_score(ensemble, X, y, cv=5, n_jobs=-1, verbose=1, scoring='f1_weighted').mean()
ensemble.fit(X, y)

y_pred = ensemble.predict(X_test)

y_pred = pd.Series(y_pred).astype(str).str.replace('0', 'Minor_Damage_And_Injuries').str.replace('1', 'Significant_Damage_And_Serious_Injuries').str.replace('2', 'Significant_Damage_And_Fatalities').str.replace('3', 'Highly_Fatal_And_Damaging')

sub = pd.DataFrame(data={'Accident_ID' : test.Accident_ID, 'Severity' : y_pred})

sub.to_csv('final_pred.csv', index=False)