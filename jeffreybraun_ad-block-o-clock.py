# Imports
import numpy as np
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df_main = pd.read_excel('/kaggle/input/ad-blockers-to-use-or-not-to-use/Research data.xlsx')
df_main['used_ad_block'] = df_main['Current users of ad blockers'] | df_main['Past users of ad blockers']
print(df_main.head())
print(df_main.info())
fig, ax = plt.subplots()

plt.style.use('fivethirtyeight')

current = df_main['Current users of ad blockers'].value_counts()[1]
past = df_main['Past users of ad blockers'].value_counts()[1]
never = df_main['used_ad_block'].value_counts()[0]

size = 0.7
vals = np.array([[current, past], [never, 0]])
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(2)*4)
inner_colors = cmap(np.array([1, 3, 5]))

wedges,_ = ax.pie(vals.sum(axis=1), radius=2, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'))
leg1 = ax.legend(wedges, ['Used an Ad Blocker', 'Never Used an Ad Blocker'],
          title='Outer',
          loc="center left",
          bbox_to_anchor=(-1.4, 0, 0.5, 1))

wedges,_ = ax.pie(vals.flatten(), radius=2-size, colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w'))
leg2 = ax.legend(wedges, ['Currently Uses an Ad Blocker', 'Used an Ad Blocker, but not now', 'Never Used an Ad Blocker'],
          title='Inner',
          loc="center right",
          bbox_to_anchor=(2.2, 0, 0.5, 1))

ax.set(aspect='equal')
ax.add_artist(leg1)
plt.title('Respondents Ad Blocker Usage', pad = 100)
plt.show()


plt.figure(figsize=(20,10))
sns.distplot(df_main['Age'], kde=False)
plt.title('Age Distribution')
plt.ylabel('Number of Respondents')
plt.show()

def assign_age_bin(row):
    age = row.Age
    if age <= 17:
        return "17 and Under"
    elif age >= 18 and age <= 24:
        return "18 to 24"
    elif age >= 25 and age <= 34:
        return "25 to 34"
    elif age >= 35 and age <= 44:
        return "35 to 44"
    elif age >= 45 and age <= 54:
        return "45 to 54"
    elif age >= 55 and age <= 64:
        return "55 to 64"
    elif age >= 65:
        return "65+"

    
df_main['age_bin'] = df_main.apply(lambda row: assign_age_bin(row), axis=1)
col_order = ['17 and Under', '18 to 24', '25 to 34', '35 to 44', '45 to 54', '55 to 64', '65+']

ax = sns.catplot(x='used_ad_block', y=None, col="age_bin", data=df_main,saturation=.5,col_order=col_order,kind='count', ci=None, aspect=.6)
plt.show()

prob = []
for age_bin_lab in col_order:
    vc = df_main[df_main.age_bin == age_bin_lab]['used_ad_block'].value_counts()
    prob.append(vc[1]/(vc[0] + vc[1]))

plt.figure(figsize=(20,10))
sns.lineplot(x = col_order, y = prob)
sns.scatterplot(x = col_order, y = prob, s = 200)
plt.title('Age Group vs. Ad Blocker Usage')
plt.xlabel('Age Group')
plt.ylabel('Percentage of Respondents who have used Ad Blockers')
plt.show()
ax = sns.catplot(x='used_ad_block', y=None, col="Sex", data=df_main,saturation=.5,col_order=['female','male'],kind='count', ci=None, aspect=.6)
plt.show()

cols = ['female', 'male']
prob = []
for sex in cols:
    vc = df_main[df_main.Sex == sex]['used_ad_block'].value_counts()
    prob.append(vc[1]/(vc[0] + vc[1]))
    
plt.figure(figsize=(6,6))
sns.barplot(x = cols, y = prob)
plt.title('Sex vs. Ad Blocker Usage')
plt.ylim((0.25, 0.5))
plt.xlabel('Sex')
plt.ylabel('Percentage of Respondents who have used Ad Blockers')
plt.show()

df_female = df_main[df_main["Sex"] == 'female']
df_male = df_main[df_main["Sex"] == 'male']
col_order = ['17 and Under', '18 to 24', '25 to 34', '35 to 44', '45 to 54', '55 to 64', '65+']

ax = sns.catplot(x='used_ad_block', y=None, col="age_bin", data=df_female,saturation=.5,col_order=col_order,kind='count', ci=None, aspect=.6)
ax.fig.suptitle('Female Respondents', y=1.1)
plt.show()

ax = sns.catplot(x='used_ad_block', y=None, col="age_bin", data=df_male,saturation=.5,col_order=col_order,kind='count', ci=None, aspect=.6)
ax.fig.suptitle('Male Respondents', y =1.1)
plt.show()

prob_female = []
prob_male = []
for age_bin_lab in col_order:
    vc = df_female[df_female.age_bin == age_bin_lab]['used_ad_block'].value_counts()
    prob_female.append(vc[1]/(vc[0] + vc[1]))
    vc = df_male[df_male.age_bin == age_bin_lab]['used_ad_block'].value_counts()
    prob_male.append(vc[1]/(vc[0] + vc[1]))
    
plt.figure(figsize=(20,10))
sns.lineplot(x = col_order, y = prob_female, color='m')
sns.scatterplot(x = col_order, y = prob_female, s = 200, color='m', label='female')
sns.lineplot(x = col_order, y = prob_male, color='g')
sns.scatterplot(x = col_order, y = prob_male, s = 200,color = 'g', label='male')
plt.title('Age Group and Sex vs. Ad Blocker Usage')
plt.xlabel('Age Group')
plt.ylabel('Percentage of Respondents who have used Ad Blockers')
plt.legend()
plt.show()
df_main['Sex'] = df_main['Sex'].astype('category')
df_main['Sex_Cat'] = df_main['Sex'].cat.codes
df_data = df_main.copy()
target = df_data["used_ad_block"]
df_data = df_data.drop(columns=['Sex','Current users of ad blockers', 'Past users of ad blockers', 'used_ad_block'])
import xgboost as xgb
from xgboost import XGBClassifier


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  f1_score
from sklearn import preprocessing ,decomposition, model_selection,metrics,pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, plot_confusion_matrix
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import random

X_train, X_test, y_train, y_test = train_test_split(df_data, target, random_state = 1, shuffle = True)
def show_cm(classifier, X_test, y_test):
    plt.style.use('default')
    class_names = ['Never Used Ad Blocker', 'Has Used Ad Blocker']
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 xticks_rotation = 30)
        plt.title(title)
        plt.show()
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train ,y_train)
lr_pred = lr.predict(X_test)
lr_score = accuracy_score(y_test, lr_pred)
print('LogisticRegression Score: ', lr_score)
show_cm(lr, df_data, target)

# XGBoost
xgb_classifier=xgb.XGBClassifier(objective='binary:logistic',learning_rate = 0.1,gamma=0.01,max_depth = 10,booster="gbtree")
xgb_classifier.fit(X_train ,y_train)
xgb_pred = xgb_classifier.predict(X_test)
xgb_score = accuracy_score(y_test,xgb_pred)
print('XGBoost Score: ',xgb_score)
show_cm(xgb_classifier, df_data, target)

# Decision Tree
dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(X_train ,y_train)
dt_pred = dt_clf.predict(X_test)
dt_score = accuracy_score(y_test,dt_pred)
print('Decision Tree Score: ',dt_score)
show_cm(dt_clf, df_data, target)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(n_estimators=1500,learning_rate=1,algorithm='SAMME.R')
ada_clf.fit(X_train ,y_train)
ada_pred = ada_clf.predict(X_test)
ada_score = accuracy_score(y_test,ada_pred)
print('AdaBoost Decision Tree Score: ',ada_score)
show_cm(ada_clf, df_data, target)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth=10, random_state=0)
rf_clf.fit(X_train ,y_train)
rf_pred = rf_clf.predict(X_test)
rf_score = accuracy_score(y_test,rf_pred)
print('Random Forest Score: ',rf_score)
show_cm(rf_clf, df_data, target)

# Linear Support Vector Classification
lsvc = LinearSVC()
lsvc.fit(X_train ,y_train)
lsvc_pred = lsvc.predict(X_test)
lsvc_score = accuracy_score(y_test,lsvc_pred)
print('LinearSVC Score: ', lsvc_score)
show_cm(lsvc, df_data, target)

# Support Vector Classification
svc = SVC()
svc.fit(X_train ,y_train)
svc_pred = svc.predict(X_test)
svc_score = accuracy_score(y_test,svc_pred)
print('SVC Score: ', svc_score)
show_cm(svc, df_data, target)

# Nu-Support Vector Classification
nusvc = NuSVC()
nusvc.fit(X_train ,y_train)
nusvc_pred = count_nusvc.predict(X_test)
nusvc_score = accuracy_score(y_test, nusvc_pred)
print('NuSVC Score: ', nusvc_score)
show_cm(nusvc, df_data, target)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train ,y_train)
sgd_pred = count_sgd.predict(X_test)
sgd_score = accuracy_score(y_test,sgd_pred)
print('SGD Score: ', sgd_score)
show_cm(sgd, df_data, target)


models = pd.DataFrame({
    'Model': ['Linear Support Vector Classification', 'Support vector Classification', 'Nu-Support Vector Classification', 
              'Stochastic Gradient Decent', 'Logistic Regression','XGBoost','AdaBoost', 'Decision Tree', 'Random Forest'],
    'Score': [ 
              lsvc_score, svc_score, nusvc_score, 
              sgd_score, lr_score, xgb_score, ada_score, dt_score, rf_score]})
models = models.sort_values(by="Score",ascending=False)
print(models.to_string())
