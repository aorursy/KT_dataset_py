

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import warnings

import seaborn as sns

from colorama import Fore, Back, Style 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from mlxtend.plotting import plot_confusion_matrix

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.formula.api import ols

import plotly.graph_objs as gobj



init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")

import plotly.figure_factory as ff



%matplotlib inline



import xgboost

import lightgbm

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
heart_data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heart_data.head()
# age distribution



hist_data =[heart_data["age"].values]

group_labels = ['age'] 



fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(title_text='Age Distribution plot')



fig.show()
fig = px.box(heart_data, x='sex', y='age', points="all")

fig.update_layout(

    title_text="Gender wise Age Spread - Male = 1 Female =0")

fig.show()
male = heart_data[heart_data["sex"]==1]

female = heart_data[heart_data["sex"]==0]



male_survi = male[heart_data["DEATH_EVENT"]==0]

male_not = male[heart_data["DEATH_EVENT"]==1]

female_survi = female[heart_data["DEATH_EVENT"]==0]

female_not = female[heart_data["DEATH_EVENT"]==1]



labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]

values = [len(male[heart_data["DEATH_EVENT"]==0]),len(male[heart_data["DEATH_EVENT"]==1]),

         len(female[heart_data["DEATH_EVENT"]==0]),len(female[heart_data["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - Gender")

fig.show()
surv = heart_data[heart_data["DEATH_EVENT"]==0]["age"]

not_surv = heart_data[heart_data["DEATH_EVENT"]==1]["age"]

hist_data = [surv,not_surv]

group_labels = ['Survived', 'Not Survived']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(

    title_text="Analysis in Age on Survival Status")

fig.show()
fig = px.violin(heart_data, y="age", x="sex", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)

fig.update_layout(title_text="Analysis in Age and Gender on Survival Status")

fig.show()
fig = px.violin(heart_data, y="age", x="smoking", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)

fig.update_layout(title_text="Analysis in Age and Smoking on Survival Status")

fig.show()
fig = px.violin(heart_data, y="age", x="diabetes", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)

fig.update_layout(title_text="Analysis in Age and Diabetes on Survival Status")

fig.show()
fig = px.histogram(heart_data, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)

fig.show()
fig = px.histogram(heart_data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)

fig.show()
fig = px.histogram(heart_data, x="platelets", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)

fig.show()
fig = px.histogram(heart_data, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=heart_data.columns)

fig.show()
fig = px.histogram(heart_data, x="serum_sodium", color="DEATH_EVENT", marginal="violin",hover_data=heart_data.columns)

fig.show()
surv = heart_data[heart_data['DEATH_EVENT']==0]['serum_sodium']

not_surv = heart_data[heart_data['DEATH_EVENT']==1]['serum_sodium']

hist_data = [surv,not_surv]

group_labels = ['Survived', 'Not Survived']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(

    title_text="Analysis in Serum Sodium on Survival Status")

fig.show()
surv = heart_data[heart_data['DEATH_EVENT']==0]['serum_creatinine']

not_surv = heart_data[heart_data['DEATH_EVENT']==1]['serum_creatinine']

hist_data = [surv,not_surv]

group_labels = ['Survived', 'Not Survived']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(

    title_text="Analysis in Serum Creatinine on Survival Status")

fig.show()
surv = heart_data[heart_data['DEATH_EVENT']==0]['ejection_fraction']

not_surv = heart_data[heart_data['DEATH_EVENT']==1]['ejection_fraction']

hist_data = [surv,not_surv]

group_labels = ['Survived', 'Not Survived']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(

    title_text="Analysis in Ejaction Fraction on Survival Status")

fig.show()
labels = ['No Diabetes','Diabetes']

diabetes_yes = heart_data[heart_data['diabetes']==1]

diabetes_no = heart_data[heart_data['diabetes']==0]

values = [len(diabetes_no), len(diabetes_yes)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Diabetes")

fig.show()
fig = px.pie(heart_data, values='diabetes',names='DEATH_EVENT', title='Diabetes Death Event Ratio')

fig.show()
diabetes_yes_survi = diabetes_yes[heart_data["DEATH_EVENT"]==0]

diabetes_yes_not_survi = diabetes_yes[heart_data["DEATH_EVENT"]==1]

diabetes_no_survi = diabetes_no[heart_data["DEATH_EVENT"]==0]

diabetes__no_not_survi = diabetes_no[heart_data["DEATH_EVENT"]==1]



labels = ['Diabetes Yes - Survived','Diabetes Yes - Not Survived', 'Diabetes NO - Survived', 'Diabetes NO - Not Survived']

values = [len(diabetes_yes[heart_data["DEATH_EVENT"]==0]),len(diabetes_yes[heart_data["DEATH_EVENT"]==1]),

         len(diabetes_no[heart_data["DEATH_EVENT"]==0]),len(diabetes_no[heart_data["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - Diabetes")

fig.show()
anaemia_yes = heart_data[heart_data['anaemia']==1]

anaemia_no = heart_data[heart_data['anaemia']==0]



labels = ['No Anaemia', 'Anaemia']

values = [len(anaemia_no), len(anaemia_yes)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on - Anaemia")

fig.show()
fig = px.pie(heart_data, values='anaemia',names='DEATH_EVENT', title='Anaemia Death Event Ration')

fig.show()
anaemia_yes_survi = anaemia_yes[heart_data["DEATH_EVENT"]==0]

anaemia_yes_not_survi = anaemia_yes[heart_data["DEATH_EVENT"]==1]

anaemia_no_survi = anaemia_no[heart_data["DEATH_EVENT"]==0]

anaemia_no_not_survi = anaemia_no[heart_data["DEATH_EVENT"]==1]



labels = ['Anaemia Yes - Survived','Anaemia Yes - Not Survived', 'Anaemia No - Survived', 'Anaemia NO - Not Survived']

values = [len(anaemia_yes[heart_data["DEATH_EVENT"]==0]),len(anaemia_yes[heart_data["DEATH_EVENT"]==1]),

         len(anaemia_no[heart_data["DEATH_EVENT"]==0]),len(anaemia_no[heart_data["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - Anaemia")

fig.show()
hbp_yes = heart_data[heart_data['high_blood_pressure']==1]

hbp_no = heart_data[heart_data['high_blood_pressure']==0]



labels = ["No High BP","High BP"]

values = [len(hbp_no), len(hbp_yes)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on - High Blood Pressure")

fig.show()
fig = px.pie(heart_data, values='high_blood_pressure',names='DEATH_EVENT', title='High Blood Pressure Death Event Ratio')

fig.show()
hbp_yes_survi = hbp_yes[heart_data["DEATH_EVENT"]==0]

hbp_yes_not_survi = hbp_yes[heart_data["DEATH_EVENT"]==1]

hbp_no_survi = hbp_no[heart_data["DEATH_EVENT"]==0]

hbp_no_not_survi = hbp_no[heart_data["DEATH_EVENT"]==1]



labels = ['HBP Yes - Survived','HBP Yes - Not Survived', 'HBP No - Survived', 'HBP NO - Not Survived']

values = [len(hbp_yes[heart_data["DEATH_EVENT"]==0]),len(hbp_yes[heart_data["DEATH_EVENT"]==1]),

         len(hbp_no[heart_data["DEATH_EVENT"]==0]),len(hbp_no[heart_data["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - HBP(high blood pressure)")

fig.show()
smoking_yes = heart_data[heart_data['smoking']==1]

smoking_no = heart_data[heart_data['smoking']==0]



labels = ['No Smoking','Smoking']

values = [len(smoking_no), len(smoking_yes)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on - Smoking")

fig.show()
fig = px.pie(heart_data, values='smoking',names='DEATH_EVENT', title='Smoking Death Event Ratio')

fig.show()
smoking_yes_survi = smoking_yes[heart_data["DEATH_EVENT"]==0]

smoking_yes_not_survi = smoking_yes[heart_data["DEATH_EVENT"]==1]

smoking_no_survi = smoking_no[heart_data["DEATH_EVENT"]==0]

smoking_no_not_survi = smoking_no[heart_data["DEATH_EVENT"]==1]



labels = ['Smoking Yes - Survived','Smoking Yes - Not Survived', 'Smoking No - Survived', 'Smoking NO- Not Survived']

values = [len(smoking_yes[heart_data["DEATH_EVENT"]==0]),len(smoking_yes[heart_data["DEATH_EVENT"]==1]),

         len(smoking_no[heart_data["DEATH_EVENT"]==0]),len(smoking_no[heart_data["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - Smoking")

fig.show()
plt.figure(figsize=(10,10))

sns.heatmap(heart_data.corr(), vmin=-1, cmap='coolwarm', annot=True);
Features = ['time','ejection_fraction','serum_creatinine']

x = heart_data[Features]

y = heart_data["DEATH_EVENT"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)
accuracy_list = []
# logistic regression



log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

log_reg_pred = log_reg.predict(x_test)

log_reg_acc = accuracy_score(y_test, log_reg_pred)

accuracy_list.append(100*log_reg_acc)
print(Fore.GREEN + "Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))
cm = confusion_matrix(y_test, log_reg_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Logistic Regression Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# svc



sv_clf = SVC()

sv_clf.fit(x_train, y_train)

sv_clf_pred = sv_clf.predict(x_test)

sv_clf_acc = accuracy_score(y_test, sv_clf_pred)

accuracy_list.append(100* sv_clf_acc)
print(Fore.GREEN + "Accuracy of SVC is : ", "{:.2f}%".format(100* sv_clf_acc))
cm = confusion_matrix(y_test, sv_clf_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("SVC Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# K Neighbors Classifier



kn_clf = KNeighborsClassifier(n_neighbors=6)

kn_clf.fit(x_train, y_train)

kn_pred = kn_clf.predict(x_test)

kn_acc = accuracy_score(y_test, kn_pred)

accuracy_list.append(100*kn_acc)
print(Fore.GREEN + "Accuracy of K Neighbors Classifier is : ", "{:.2f}%".format(100* kn_acc))
cm = confusion_matrix(y_test, kn_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("K Neighbors Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# Decision Tree Classifier



dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')

dt_clf.fit(x_train, y_train)

dt_pred = dt_clf.predict(x_test)

dt_acc = accuracy_score(y_test, dt_pred)

accuracy_list.append(100*dt_acc)
print(Fore.GREEN + "Accuracy of Decision Tree Classifier is : ", "{:.2f}%".format(100* dt_acc))
cm = confusion_matrix(y_test, dt_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Decision Tree Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# RandomForestClassifier



r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)

r_clf.fit(x_train, y_train)

r_pred = r_clf.predict(x_test)

r_acc = accuracy_score(y_test, r_pred)

accuracy_list.append(100*r_acc)
print(Fore.GREEN + "Accuracy of Random Forest Classifier is : ", "{:.2f}%".format(100* r_acc))
cm = confusion_matrix(y_test, r_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Random Forest Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# GradientBoostingClassifier



gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)

gradientboost_clf.fit(x_train,y_train)

gradientboost_pred = gradientboost_clf.predict(x_test)

gradientboost_acc = accuracy_score(y_test, gradientboost_pred)

accuracy_list.append(100*gradientboost_acc)
print(Fore.GREEN + "Accuracy of Gradient Boosting is : ", "{:.2f}%".format(100* gradientboost_acc))
cm = confusion_matrix(y_test, gradientboost_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Gredient Boosting Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# xgbrf classifier



xgb_clf = xgboost.XGBRFClassifier(max_depth=3, random_state=1)

xgb_clf.fit(x_train,y_train)

xgb_pred = xgb_clf.predict(x_test)

xgb_acc = accuracy_score(y_test, xgb_pred)

accuracy_list.append(100*xgb_acc)
print(Fore.GREEN + "Accuracy of XGBRFClassifier is : ", "{:.2f}%".format(100* xgb_acc))
cm = confusion_matrix(y_test, xgb_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("XGBRFClassifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
#  lightgbm 



lgb_clf = lightgbm.LGBMClassifier(max_depth=2, random_state=4)

lgb_clf.fit(x_train,y_train)

lgb_pred = lgb_clf.predict(x_test)

lgb_acc = accuracy_score(y_test, lgb_pred)

accuracy_list.append(100*lgb_acc)
print(Fore.GREEN + "Accuracy of LGBMClassifier is : ","{:.2f}%".format(100* lgb_acc))
cm = confusion_matrix(y_test, lgb_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("LGBMClassifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# CatBoostClassifier



cat_clf = CatBoostClassifier()

cat_clf.fit(x_train,y_train)

cat_pred = cat_clf.predict(x_test)

cat_acc = accuracy_score(y_test, cat_pred)

accuracy_list.append(100*cat_acc)
print(Fore.GREEN + "Accuracy of CatBoostClassifier is : ","{:.2f}%".format(100* cat_acc))
cm = confusion_matrix(y_test, cat_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Cat Boost Classifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
model_list = ['Logistic Regression', 'SVC','KNearestNeighbours', 'DecisionTree', 'RandomForest',

              'GradientBooster', 'XGBRF','LGBM', 'CatBoostClassifier']
plt.rcParams['figure.figsize']=20,8

sns.set_style('darkgrid')

ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)

plt.xlabel('Classifier Models', fontsize = 20 )

plt.ylabel('% of Accuracy', fontsize = 20)

plt.title('Accuracy of different Classifier Models', fontsize = 20)

plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)

plt.yticks(fontsize = 12)

for i in ax.patches:

    width, height = i.get_width(), i.get_height()

    x, y = i.get_xy() 

    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')

plt.show()