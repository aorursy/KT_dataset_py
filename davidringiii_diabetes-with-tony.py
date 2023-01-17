# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fname = os.path.join(dirname, filename)

# Load dataframe

df = pd.read_csv(fname)
df.head()
df.duplicated().sum()

df.drop_duplicates(inplace=True)
len(df)
df['Outcome'].value_counts()
# train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# Split the data

X = df.drop('Outcome', axis=1)

y = df['Outcome']



# Setup random seed

np.random.seed(42)



# Train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Instantiate classifier

clf = RandomForestClassifier(n_jobs=1)



# Fit the data

clf.fit(X_train, y_train)



# Score the data

clf_score1 = clf.score(X_test, y_test)

print(f"The score is: {clf_score1:.6f}%")
np.random.seed(42)



# Cross_val_score mean

cr_val_score = np.mean(cross_val_score(clf, X_test, y_test))

cr_val_score
y_preds_proba = clf.predict_proba(X_test)
proba_df = pd.DataFrame(y_preds_proba,

                       columns=['0', '1'])

proba_df
y_test[:5]
grid = {

       'n_estimators': list(range(530,571,10)),

       'criterion': ['gini', 'entropy'],

       #'max_depth': [None, 3, 10, 20],

       'min_samples_split': [3],

       #'min_samples_leaf': [1, 4, 8],

       #'max_features': ['auto', 'sqrt'],

       }



clf_2 = GridSearchCV(clf, grid)

clf_2.fit(X_train, y_train)
gs_mean_scores = clf_2.cv_results_['mean_test_score']

gs_mean_scores
gs_params = clf_2.cv_results_['params']

param_df = pd.DataFrame(gs_params)

param_df['Mean Score'] = gs_mean_scores
param_df
from pandas.plotting import parallel_coordinates



plt.style.use("seaborn")

fig = plt.figure(figsize=(10, 6))

title = fig.suptitle("Parallel Coordinates", fontsize=18)

fig.subplots_adjust(top=0.93, wspace=0)



pc = parallel_coordinates(param_df, 'Mean Score', color=('skyblue', 'firebrick'))
clf = RandomForestClassifier(n_estimators=500, n_jobs=1)



clf.fit(X_train, y_train)



clf_score = clf.score(X_test, y_test)

print(f"The best score possible is:  {clf_score:.6f}%")
#plt.style.available

print(param_df.columns)

import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline



#sns.kdeplot(param_df)

sns.set()



# for col in ['min_samples_split', 'n_estimators']:

#     #sns.kdeplot(param_df[col], shade=True)

#     plt.plot(param_df[col],param_df['Mean Score'] )

    

# plt.legend('ABCDEF', ncol=2, loc='upper left');







#sns.kdeplot(param_df);



# with sns.axes_style('white'):

#     sns.jointplot("Mean Score", param_df, kind='hex')



print(param_df.head())

#mappy = sns.load_dataset('flights')

mappy = param_df.copy()

mappy = mappy.drop('min_samples_split', axis=1)

print(mappy.head())

mappy = mappy.pivot("criterion", "n_estimators", "Mean Score")

#sns.heatmap(param_df)

ax = sns.heatmap(mappy, cmap='Blues', annot=True)
sns.palplot(sns.light_palette("navy", reverse=True))
# Testing RandomForestClassifiers over a range of n_estimators



max = 0

for est in range(100,700,100):

    clf = RandomForestClassifier(n_estimators=est, n_jobs=1)



    clf = RandomForestClassifier(

                      # min_samples_leaf=2,

                      min_samples_split = 4,        

                      n_estimators=est,

                      bootstrap=True,

                      n_jobs=1,

                      random_state=42,

                      max_features='auto')

    

    clf.fit(X_train, y_train)



    clf_score = clf.score(X_test, y_test)

    if clf_score > max:

        max = clf_score

        print(f"New max with estimators={est}, the best score possible is:  {clf_score:.6f}%")

    else:

        print('.', end="")

print(f'\nFinished.')

        
# Testing XGBoost over a range of n_estimators



import xgboost as xgb

from sklearn.metrics import mean_squared_error



lowest = 100

for est in range(100, 700, 100):

    xg_reg = xgb.XGBClassifier(max_depth=10, learning_rate=0.15, n_estimators=est, verbosity=0,

                              silent=True, objective='binary:logistic', booster='gbtree')



    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    

    if rmse < lowest:

        lowest = rmse

        print(f"\nNew low with estimators={est}, rmse:  {rmse:.6f}%")

        

        preds[preds>0.5] = "1"

        pred_y = preds==y_test

        print(f'Percent Correct: {100*pred_y.mean()}%')        

        

    else:

        print('.', end="")

print(f'\nFinished.')    

    
df.head()
df[df['Pregnancies'] == 3].count()
g = sns.distplot(df['Pregnancies'], color='purple', bins=30)

g.set(xlim=(0, 17));
# Split the data into neg and pos diabetes

diabetes_neg = df[df['Outcome'] == 0]

diabetes_pos = df[df['Outcome'] == 1]
# Plot the neg and pos diabetes in a stacked bar chart

fig, ax = plt.subplots()



ax.bar(diabetes_neg['Pregnancies'], diabetes_neg['Outcome'], label='Negative', width=0.35)

#ax.bar(diabetes_pos['Outcome'], diabetes_pos['Pregnancies'], label='Positive')

ax.legend()

plt.show()
df[:5]
df['Outcome'].value_counts()
491 / (491 + 253)
for i, col in enumerate(df.columns):

    if col == 'Outcome':

        continue

    plt.figure(i)

    positives = df[col][df['Outcome']==1]

    negatives = df[col][df['Outcome']==0]

    sns.distplot(positives, hist = False, color="red", label="Positive")

    sns.distplot(negatives, hist = False, color="skyblue", label="Negative")
## All in one cell



np.random.seed(42)



# Split data

X = df.drop('Outcome', axis=1)

y = df['Outcome']



# Train and test splits

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Instantiate the model

clf = RandomForestClassifier(n_jobs=-1)



# Grid for RandomSearchCV



grid = {

       'n_estimators': list(range(100, 801, 100)),

       'criterion': ['gini', 'entropy'],

       'max_depth': [None, 3, 10, 20],

       'min_samples_split': [3],

       'min_samples_leaf': [1, 4, 8],

       'max_features': ['auto', 'sqrt'],

       }



rs_clf = GridSearchCV(clf, grid)

rs_clf.fit(X_train, y_train)

rs_clf.score(X_test, y_test)
rs_clf.best_params_