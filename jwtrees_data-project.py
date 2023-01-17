import pandas as pd

import numpy as np

import random as rnd

from tabulate import tabulate



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import plotly as py

import plotly.express as px

import plotly.graph_objects as go



%matplotlib inline



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn import svm

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB,CategoricalNB,MultinomialNB

from sklearn.linear_model import Perceptron, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, accuracy_score

from xgboost import XGBClassifier

from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss



train_df = pd.read_csv("/kaggle/input/dataproject/train.csv")

test_df = pd.read_csv("/kaggle/input/dataproject/test.csv")

complete_df = [train_df, test_df]



drop_column = ['Gameplay']

for datadf in complete_df:    

    datadf.drop(drop_column, axis=1, inplace = True)

    

# Ok, time to visualize the data!

import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

print('x' in np.arange(5))
plt.figure(figsize=(12,8))



custom = [Line2D([], [], marker='o', color='#023EFF', linestyle='None'),

          Line2D([], [], marker='o', color='#FF7C00', linestyle='None')]

sns.swarmplot(x="blueBaronKills", y="blueDragonKills", hue="blueWins", data=train_df, palette="bright")

ax = sns.boxplot(x="blueBaronKills", y="blueDragonKills", data=train_df, color='white')



for i,box in enumerate(ax.artists):

    box.set_edgecolor('black')

    box.set_facecolor('white')

    for j in range(6*i,6*(i+1)):

         ax.lines[j].set_color('black')

                    

plt.legend(custom, ['Lose', 'Win'], loc='upper right')

plt.rc('font', size=20)

plt.rc('axes', titlesize=20)
plt.figure(figsize=(12,8))



custom = [Line2D([], [], marker='o', color='#023EFF', linestyle='None'),

          Line2D([], [], marker='o', color='#FF7C00', linestyle='None')]

sns.swarmplot(x="blueInhibitorKills", y="blueTowerKills", hue="blueWins", data=train_df, palette="bright")

ax = sns.boxplot(x="blueInhibitorKills", y="blueTowerKills", data=train_df, color='white')



for i,box in enumerate(ax.artists):

    box.set_edgecolor('black')

    box.set_facecolor('white')

    for j in range(6*i,6*(i+1)):

         ax.lines[j].set_color('black')

                    

plt.legend(custom, ['Lose', 'Win'], loc='upper right')

plt.rc('font', size=20)

plt.rc('axes', titlesize=20)
train_df["blueWins"] = train_df["blueWins"].astype(str)

fig = px.scatter(train_df, x="blueBaronKills", y="blueDragonKills", color="blueWins", marginal_x="box", marginal_y="box" ,title="Box plots of Dragon and Baron")

fig.update_layout(legend_title_text='blueWins? 0=No, 1=Yes')

fig
X = train_df

y = train_df['blueWins'].values



X.drop(['blueWins'],axis = 1, inplace = True)



X_submit = test_df



X = preprocessing.StandardScaler().fit(X).transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
DT_model = DecisionTreeClassifier(criterion="entropy")

DT_model.fit(X_train,y_train)



DT_yhat = DT_model.predict(X_test)



print("DT accuracy: %.2f" % accuracy_score(y_test, DT_yhat))

print("DT Jaccard index: %.2f" % jaccard_score(y_test, DT_yhat,pos_label='1'))

print("DT F1-score: %.2f" % f1_score(y_test, DT_yhat, average='weighted') )
RanFor_model = RandomForestClassifier(n_estimators=10,random_state=1).fit(X_train,y_train)

RanFor_yhat = RanFor_model.predict(X_test)



print("Random Forest accuracy: %.2f" % accuracy_score(y_test, RanFor_yhat))

print("Random Forest Jaccard index: %.2f" % jaccard_score(y_test, RanFor_yhat,pos_label='1'))

print("Random Forest F1-score: %.2f" % f1_score(y_test, RanFor_yhat, average='weighted') )
# From sklearn import svm.SVC()

SVM_model = SVC()

SVM_model.fit(X_train,y_train)



SVM_yhat = SVM_model.predict(X_test)



print("SVM accuracy: %.2f" % accuracy_score(y_test, SVM_yhat))

print("SVM Jaccard index: %.2f" % jaccard_score(y_test, SVM_yhat,pos_label='1'))

print("SVM F1-score: %.2f" % f1_score(y_test, SVM_yhat, average='weighted') )
LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)

LR_yhat = LR_model.predict(X_test)



print("LR accuracy: %.2f" % accuracy_score(y_test, LR_yhat))

print("LR Jaccard index: %.2f" % jaccard_score(y_test, LR_yhat,pos_label='1'))

print("LR F1-score: %.2f" % f1_score(y_test, LR_yhat, average='weighted') )
NB_model = BernoulliNB(2).fit(X_train,y_train)

NB_yhat = NB_model.predict(X_test)



print("NB accuracy: %.2f" % accuracy_score(y_test, NB_yhat))

print("NB Jaccard index: %.2f" % jaccard_score(y_test, NB_yhat,pos_label='1'))

print("NB F1-score: %.2f" % f1_score(y_test, NB_yhat, average='weighted') )
XGB_model=XGBClassifier(max_depth=3).fit(X_train,y_train)

XGB_pred=XGB_model.predict(X_test)

    

print("XGB accuracy: %.2f" % accuracy_score(y_test, XGB_pred))

print("XGB Jaccard index: %.2f" % jaccard_score(y_test, XGB_pred,pos_label='1'))

print("XGB F1-score: %.2f" % f1_score(y_test, XGB_pred, average='weighted') )    

    
from tabulate import tabulate

data = [['Decision Tree', accuracy_score(y_test, DT_yhat), jaccard_score(y_test, DT_yhat,pos_label='1'), f1_score(y_test, DT_yhat, average='weighted')],

['Random Forest Classifier', accuracy_score(y_test, RanFor_yhat), jaccard_score(y_test, RanFor_yhat,pos_label='1'), f1_score(y_test, RanFor_yhat, average='weighted')],

['Support Vector Machine', accuracy_score(y_test, SVM_yhat), jaccard_score(y_test, SVM_yhat,pos_label='1'), f1_score(y_test, SVM_yhat, average='weighted')],

['Logistic Regression', accuracy_score(y_test, LR_yhat), jaccard_score(y_test, LR_yhat,pos_label='1'), f1_score(y_test, LR_yhat, average='weighted')],

['Bernoulli Naive_Bayes', accuracy_score(y_test, NB_yhat), jaccard_score(y_test, NB_yhat,pos_label='1'), f1_score(y_test, NB_yhat, average='weighted')],

['XGB Classifier', accuracy_score(y_test, XGB_pred), jaccard_score(y_test, XGB_pred,pos_label='1'), f1_score(y_test, XGB_pred, average='weighted')]]

print (tabulate(data, headers=["Model", "Accuracy", "Jaccard score", "F1-Score"]))
prediction_submit = SVM_model.predict(X_submit)





testdata = pd.read_csv("/kaggle/input/dataproject/submission.csv")

output = pd.DataFrame({'Gameplay': testdata.Gameplay, 'blueWins': prediction_submit})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")