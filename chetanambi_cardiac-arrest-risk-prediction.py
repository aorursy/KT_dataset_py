import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/cardiacarrestriskprediction/Cardiac_Arrest_Participants_Data/Train.csv')

test = pd.read_csv('/kaggle/input/cardiacarrestriskprediction/Cardiac_Arrest_Participants_Data/Test.csv')

sample_sub = pd.read_excel('/kaggle/input/cardiacarrestriskprediction/Cardiac_Arrest_Participants_Data/Sample_Submission.xlsx')
train.shape, test.shape, sample_sub.shape
train.head(3)
train.nunique()
test.nunique()
train['Gender'].value_counts()
train['UnderRisk'].value_counts()
train['UnderRisk'] = train['UnderRisk'].map({'yes':1, 'no':0})

train['UnderRisk'].value_counts()
train.describe()
plt.figure(figsize=(12,8))

sns.heatmap(train.corr(), annot=True, cmap='Blues');
bin_cols = ['Chain_smoker', 'Consumes_other_tobacco_products', 'HighBP', 'Obese', 'Diabetes', 'Metabolic_syndrome']



import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20)) # size of figure



total = len(train)



# loop to get column and the count of plots

for n, col in enumerate(train[bin_cols]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=train, hue='UnderRisk', palette='hls') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by UnderRisk', fontsize=14) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

    sizes=[] # Get highest values in y

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=12) 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

    

plt.show()
bin_cols = ['Use_of_stimulant_drugs', 'Family_history', 'History_of_preeclampsia', 'CABG_history', 'Respiratory_illness']



import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20)) # size of figure



total = len(train)



# loop to get column and the count of plots

for n, col in enumerate(train[bin_cols]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=train, hue='UnderRisk', palette='hls') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by UnderRisk', fontsize=14) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

    sizes=[] # Get highest values in y

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=12) 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

    

plt.show()
cols = ['Gender', 'Chain_smoker', 'Consumes_other_tobacco_products', 'HighBP',

       'Obese', 'Diabetes', 'Metabolic_syndrome', 'Use_of_stimulant_drugs',

       'Family_history', 'History_of_preeclampsia', 'CABG_history',

       'Respiratory_illness']
train = pd.get_dummies(train, columns=cols, drop_first=True)

test = pd.get_dummies(test, columns=cols, drop_first=True)
train.head(3)
X = train.drop(labels=['UnderRisk'], axis=1)

y = train['UnderRisk'].values



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
Xtest = test
err_lgb = []

y_pred_tot_lgb = []

fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2)

for train_index, test_index in fold.split(X, y):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    #clf = XGBClassifier(random_state=2, learning_rate=0.05, subsample=0.8, max_depth=4, gamma=0.1)

    clf = LogisticRegression(random_state=2, C=10, max_iter=100)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)         

    print("Log Loss:", log_loss(y_test, y_pred))

    err_lgb.append(log_loss(y_test, y_pred))

    p = clf.predict_proba(Xtest)

    y_pred_tot_lgb.append(p)
np.mean(err_lgb, 0)
y_pred = np.mean(y_pred_tot_lgb, 0)
sub = pd.DataFrame(y_pred)

sub.columns = ['no','yes']

sub.head(5)
sub.to_excel('Output.xlsx', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)