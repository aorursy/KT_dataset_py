import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import plotly.express as px 

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import plotly.figure_factory as ff





from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

import lightgbm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

import xgboost



df= pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.describe()
correlation = df.corr()

plt.figure(figsize=(16,12))

plt.title('Correlation Heatmap of Death event')

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
fig = px.histogram(x=df['age'], nbins=50, histnorm='density')

fig.update_layout(title='Age distribution:',

                 xaxis_title='age', yaxis_title='Count')
g = sns.catplot(x="age", kind="box",data=df)
fig = px.box(df, x='DEATH_EVENT', y='age',color='sex')

fig.update_layout(

    title_text="Gender wise Age Spread and Death - Male = 1 Female =0")

fig.show()
ax = sns.barplot(x=df.sex, y= df.age,hue=df.DEATH_EVENT, data=df)

ax.set_title("Death age for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x=df.sex, hue=df.DEATH_EVENT, data=df)

ax.set_title("Death case for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()
gender_smoking= df.groupby('sex')['smoking'].sum().sort_values(ascending=False)

fig = px.bar(df,gender_smoking.index, gender_smoking)

fig.update_layout(title='Relationship between gender and smoking - Male = 1 Female =0')

fig.show()
gender_smoking= df.groupby('sex')['high_blood_pressure'].sum().sort_values(ascending=False)

fig = px.bar(df,gender_smoking.index, gender_smoking)

fig.update_layout(title='high blood_pressure for both gender - Male = 1 Female =0')

fig.show()
gender_smoking= df.groupby('sex')['diabetes'].sum().sort_values(ascending=False)

fig = px.bar(df,gender_smoking.index, gender_smoking)

fig.update_layout(title='diabetes for both gender - Male = 1 Female =0')

fig.show()
gender_smoking= df.groupby('sex')['creatinine_phosphokinase'].sum().sort_values(ascending=False)

fig = px.bar(df,gender_smoking.index, gender_smoking)

fig.update_layout(title='high blood pressure for both gender')

fig.show()
male = df[df["sex"]==1]

female = df[df["sex"]==0]



male_survi = male[df["DEATH_EVENT"]==0]

male_not = male[df["DEATH_EVENT"]==1]

female_survi = female[df["DEATH_EVENT"]==0]

female_not = female[df["DEATH_EVENT"]==1]



labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]

values = [len(male[df["DEATH_EVENT"]==0]),len(male[df["DEATH_EVENT"]==1]),

         len(female[df["DEATH_EVENT"]==0]),len(female[df["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - Gender")

fig.show()




ax = sns.barplot(x=df.sex, y= df.platelets,hue=df.DEATH_EVENT, data=df)

ax.set_title("Death age for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()




fig = px.scatter(df, x="creatinine_phosphokinase", y="age", color="DEATH_EVENT", marginal_y="violin",

           marginal_x="box", trendline="ols", template="simple_white")

fig.show()

fig = px.box(df, x='DEATH_EVENT', y='creatinine_phosphokinase')

fig.update_layout(

    title_text="Gender wise Age Spread and Death - Male = 1 Female =0")

fig.show()




ax = sns.barplot(x=df.sex, y= df.creatinine_phosphokinase,hue=df.DEATH_EVENT, data=df)

ax.set_title("Death age for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()








fig = px.scatter(df, x="serum_creatinine", y="age", color="DEATH_EVENT", marginal_y="violin",

           marginal_x="box", trendline="ols", template="simple_white")

fig.show()



ax = sns.barplot(x=df.sex, y= df.serum_creatinine,hue=df.DEATH_EVENT, data=df)

ax.set_title("Death age for Females and Males")

x_ticks_labels=['Female', 'Male']

ax.set_xticklabels(x_ticks_labels)

plt.show()
folowUp = df.groupby('time').sum()['DEATH_EVENT']

folowUp.plot(figsize=(20,10),title='trends on death times as folow up increased')


x=df.drop(columns=('DEATH_EVENT'))

y=df['DEATH_EVENT']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)


clf = RandomForestClassifier(max_depth=2, random_state=1)

clf.fit(x_train, y_train)

pred=clf.predict(x_test)

clf.score(x_test,y_test)
cm = confusion_matrix(y_test, pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Random Forest Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)

gradientboost_clf.fit(x_train,y_train)

gradientboost_pred = gradientboost_clf.predict(x_test)
gradientboost_clf.score(x_test,y_test)
cm = confusion_matrix(y_test, gradientboost_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Gredient Boosting Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# xgbrf classifier



xgb_clf = xgboost.XGBRFClassifier(max_depth=2, random_state=1)

xgb_clf.fit(x_train,y_train)

xgb_pred = xgb_clf.predict(x_test)

xgb_clf.score(x_test,y_test)
cm = confusion_matrix(y_test, xgb_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("XGBRFClassifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
lgb_clf = lightgbm.LGBMClassifier(max_depth=2, random_state=1)

lgb_clf.fit(x_train,y_train)

lgb_pred = lgb_clf.predict(x_test)

lgb_clf.score(x_test,y_test)
cm = confusion_matrix(y_test, lgb_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("LGBMClassifier Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()