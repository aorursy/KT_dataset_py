import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization advanced
df_train = pd.read_csv('../input/da-coaching-human-activity/train.csv')

df_test = pd.read_csv('../input/da-coaching-human-activity/test.csv')
df_train.head()
df_test.head()
df_train.shape
df_test.shape
df_train.shape
df_train.dtypes.value_counts()
df_train.select_dtypes('int64').head(3)
df_train.select_dtypes('object').head(3)
import re
columns = df_train.columns.to_numpy()
time_feats = []

time_func = set()

freq_feats = []

freq_func = set()

other_feats = []



n_time = 0

n_freq = 0

n_other = 0



regex_func = re.compile('-([a-z]+)')

regex_axis = re.compile('-([A-Z])')



for i in range(564):

    if np.char.startswith(columns[i],'t'):

        time_feats.append(columns[i])

        time_func.add(regex_func.findall(columns[i])[0])

        n_time += 1

    elif np.char.startswith(columns[i],'f'):

        freq_feats.append(columns[i])

        freq_func.add(regex_func.findall(columns[i])[0])

        n_freq += 1

    else:

        other_feats.append(columns[i])

        n_other += 1
print('Time freatures:',sorted(time_func))

print('Frequency freatures:',sorted(freq_func))
print('Other features:',sorted(other_feats))
n_time, n_freq, n_other, n_time + n_freq + n_other
df_train['Activity'].value_counts()
chart = sns.countplot(df_train['Activity'])

t = chart.set_xticklabels(chart.get_xticklabels(),rotation=25)
act_map = {'STANDING':0, 'SITTING':1, 'LAYING':2, 'WALKING':3, 'WALKING_DOWNSTAIRS':4, 'WALKING_UPSTAIRS':5}

df_train['activity_code'] = df_train['Activity'].map(act_map)
df_train['Activity'].value_counts()
df_train['activity_code'].value_counts()
fig, ax = plt.subplots(1,1, figsize=(15,15))

sns.heatmap(df_train[time_feats+['activity_code']].corr(), 

            cmap=sns.diverging_palette(240, 10, n=25), 

            cbar=True,ax=ax)
fig, ax = plt.subplots(1,1, figsize=(15,15))

sns.heatmap(df_train[freq_feats+['activity_code']].corr(), 

            cmap=sns.diverging_palette(240, 10, n=25), 

            cbar=True,ax=ax)
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
df_train.drop(columns=['Id', 'subject', 'Activity'], inplace=True)
y = df_train.pop('activity_code')

X = df_train
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



from sklearn.metrics import classification_report
std_scaler = StandardScaler()

X_prep_train = std_scaler.fit_transform(X_train)

X_prep_test = std_scaler.transform(X_test)
LR_clf = LogisticRegression()

LR_clf.fit(X_prep_train, y_train)



y_pred = LR_clf.predict(X_prep_train)

print(classification_report(y_train, y_pred))
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=5)
scores = cross_val_score(LR_clf, X_prep_train, y_train, scoring='accuracy', cv=kfold)

print('Scores:',scores)

print('Mean:',np.mean(scores))

print('Std:',np.std(scores))
y_pred = LR_clf.predict(X_prep_test)



print(classification_report(y_test, y_pred))
df_test.drop(columns=['Id', 'subject'], inplace=True)
X_final_test = df_test
X_final_prep_test = std_scaler.transform(X_final_test)
y_final_pred = LR_clf.predict(X_final_prep_test)
y_final_pred
rev_act_map = {0:'STANDING', 1:'SITTING', 2:'LAYING', 3:'WALKING', 4:'WALKING_DOWNSTAIRS', 5:'WALKING_UPSTAIRS'}

y_final = [rev_act_map[code] for code in y_final_pred]
submission = pd.DataFrame({

        "Id": range(1,len(y_final)+1),

        "Activity": y_final

    })



submission.to_csv('lr_sub.csv',index=False)
submission.head()