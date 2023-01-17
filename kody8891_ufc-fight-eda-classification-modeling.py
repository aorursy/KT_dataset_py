import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math

import pylab

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline

import xgboost

import shap

from xgboost import XGBRegressor

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, make_scorer

import re

from xgboost import XGBClassifier



UFC=pd.read_csv('../input/ufcdata/data.csv')

Fighter_df=pd.read_csv('../input/ufcdata/raw_fighter_details.csv')

df=pd.read_csv('../input/ufcdata/preprocessed_data.csv')

UFC.head()
Fighter_df.info()

UFC.info()



#null variables

UFC.isnull().sum()
#drop null variables

UFC.dropna(inplace=True)

UFC.isnull().sum()
# data types

UFC.dtypes
# change object variables to float

UFC['no_of_rounds'] = pd.to_numeric(UFC['no_of_rounds'])

UFC['B_current_lose_streak'] = pd.to_numeric(UFC['B_current_lose_streak'])

UFC['B_current_win_streak'] = pd.to_numeric(UFC['B_current_win_streak'])

UFC['B_draw'] = pd.to_numeric(UFC['B_draw'])

UFC['B_avg_BODY_att'] = pd.to_numeric(UFC['B_avg_BODY_att'])

UFC['B_avg_BODY_landed'] = pd.to_numeric(UFC['B_avg_BODY_landed'])

UFC['B_avg_CLINCH_att'] = pd.to_numeric(UFC['B_avg_CLINCH_att'])

UFC['B_avg_CLINCH_landed'] = pd.to_numeric(UFC['B_avg_CLINCH_landed'])

UFC['B_avg_DISTANCE_att'] = pd.to_numeric(UFC['B_avg_DISTANCE_att'])

UFC['B_avg_DISTANCE_landed'] = pd.to_numeric(UFC['B_avg_DISTANCE_landed'])

UFC['B_avg_DISTANCE_att'] = pd.to_numeric(UFC['B_avg_DISTANCE_att'])

UFC['B_avg_GROUND_att'] = pd.to_numeric(UFC['B_avg_GROUND_att'])

UFC['B_avg_DISTANCE_att'] = pd.to_numeric(UFC['B_avg_DISTANCE_att'])

UFC['B_avg_GROUND_landed'] = pd.to_numeric(UFC['B_avg_GROUND_landed'])

UFC['B_avg_HEAD_att'] = pd.to_numeric(UFC['B_avg_HEAD_att'])

UFC['B_avg_HEAD_landed'] = pd.to_numeric(UFC['B_avg_HEAD_landed'])

UFC['B_avg_KD'] = pd.to_numeric(UFC['B_avg_KD'])

UFC['B_avg_LEG_att'] = pd.to_numeric(UFC['B_avg_LEG_att'])

UFC['B_avg_LEG_landed'] = pd.to_numeric(UFC['B_avg_LEG_landed'])

UFC['B_avg_PASS'] = pd.to_numeric(UFC['B_avg_PASS'])

UFC['B_avg_REV'] = pd.to_numeric(UFC['B_avg_REV'])

UFC['B_avg_SIG_STR_att'] = pd.to_numeric(UFC['B_avg_SIG_STR_att'])

UFC['B_avg_SIG_STR_landed'] = pd.to_numeric(UFC['B_avg_SIG_STR_landed'])

UFC['B_avg_SIG_STR_pct'] = pd.to_numeric(UFC['B_avg_SIG_STR_pct'])

UFC['B_Height_cms'] = pd.to_numeric(UFC['B_Height_cms'])

UFC['B_age'] = pd.to_numeric(UFC['B_age'])

UFC['B_total_time_fought(seconds)'] = pd.to_numeric(UFC['B_total_time_fought(seconds)'])





UFC['no_of_rounds'] = pd.to_numeric(UFC['no_of_rounds'])

UFC['R_current_lose_streak'] = pd.to_numeric(UFC['R_current_lose_streak'])

UFC['R_current_win_streak'] = pd.to_numeric(UFC['R_current_win_streak'])

UFC['R_draw'] = pd.to_numeric(UFC['R_draw'])

UFC['R_avg_BODY_att'] = pd.to_numeric(UFC['R_avg_BODY_att'])

UFC['R_avg_BODY_landed'] = pd.to_numeric(UFC['R_avg_BODY_landed'])

UFC['R_avg_CLINCH_att'] = pd.to_numeric(UFC['R_avg_CLINCH_att'])

UFC['R_avg_CLINCH_landed'] = pd.to_numeric(UFC['R_avg_CLINCH_landed'])

UFC['R_avg_DISTANCE_att'] = pd.to_numeric(UFC['R_avg_DISTANCE_att'])

UFC['R_avg_DISTANCE_landed'] = pd.to_numeric(UFC['R_avg_DISTANCE_landed'])

UFC['R_avg_DISTANCE_att'] = pd.to_numeric(UFC['R_avg_DISTANCE_att'])

UFC['R_avg_GROUND_att'] = pd.to_numeric(UFC['R_avg_GROUND_att'])

UFC['R_avg_DISTANCE_att'] = pd.to_numeric(UFC['R_avg_DISTANCE_att'])

UFC['R_avg_GROUND_landed'] = pd.to_numeric(UFC['R_avg_GROUND_landed'])

UFC['R_avg_HEAD_att'] = pd.to_numeric(UFC['R_avg_HEAD_att'])

UFC['R_avg_HEAD_landed'] = pd.to_numeric(UFC['R_avg_HEAD_landed'])

UFC['R_avg_KD'] = pd.to_numeric(UFC['R_avg_KD'])

UFC['R_avg_LEG_att'] = pd.to_numeric(UFC['R_avg_LEG_att'])

UFC['R_avg_LEG_landed'] = pd.to_numeric(UFC['R_avg_LEG_landed'])

UFC['R_avg_PASS'] = pd.to_numeric(UFC['R_avg_PASS'])

UFC['R_avg_REV'] = pd.to_numeric(UFC['R_avg_REV'])

UFC['R_avg_SIG_STR_att'] = pd.to_numeric(UFC['R_avg_SIG_STR_att'])

UFC['R_avg_SIG_STR_landed'] = pd.to_numeric(UFC['R_avg_SIG_STR_landed'])

UFC['R_Height_cms'] = pd.to_numeric(UFC['R_Height_cms'])

UFC['R_age'] = pd.to_numeric(UFC['R_age'])

UFC['R_total_time_fought(seconds)'] = pd.to_numeric(UFC['R_total_time_fought(seconds)'])

UFC['R_win_by_KO/TKO'] = pd.to_numeric(UFC['R_win_by_KO/TKO'])

UFC['R_win_by_Submission'] = pd.to_numeric(UFC['R_win_by_Submission'])

UFC.dtypes
import seaborn as sns

import matplotlib.pyplot as plt

from numpy import mean

time=sns.barplot(x="weight_class",y='R_total_time_fought(seconds)', data=UFC,estimator=mean,order=['Flyweight','Featherweight','Bantamweight','Lightweight','Welterweight','Middleweight','Light Heavyweight','Heavyweight'])

plt.xlabel('Weight Class')

plt.ylabel('Fight time in Seconds')

plt.title('Average Fight time by weight Class')

plt.xticks(rotation=60)

plt.tight_layout()

time
from numpy import mean

ko = sns.barplot(x="weight_class",y='R_win_by_KO/TKO', data=UFC,estimator=mean,order=['Heavyweight','Light Heavyweight','Middleweight','Welterweight','Lightweight','Featherweight','Bantamweight','Flyweight'])

plt.xlabel('Weight Class')

plt.ylabel('Total Knockouts')

plt.title('Average Knockouts by Weight Class')

plt.xticks(rotation=60)

plt.tight_layout()

ko
sub = sns.barplot(x="weight_class",y='R_win_by_Submission', data=UFC,estimator=mean,order=['Lightweight','Welterweight','Middleweight','Featherweight','Light Heavyweight','Heavyweight','Bantamweight','Flyweight'])

plt.xlabel('Weight Class')

plt.ylabel('Total Submissions')

plt.title('Average Submissions by Weight Class')

plt.xticks(rotation=60)

plt.tight_layout()

sub


wc = sns.countplot(y="weight_class",order=['Lightweight','Welterweight','Middleweight','Light Heavyweight','Featherweight','Heavyweight','Bantamweight','Flyweight'], data=UFC)

plt.title("Number of Fighters by Weight class")

plt.xlabel("# of Fighters")

plt.ylabel("Weight Class")

plt.tight_layout()

wc
fighters = pd.concat([UFC['R_fighter'], UFC['B_fighter']], ignore_index=True)

names = ' '

for name in fighters:

    name = str(name)

    names = names + name + ' '

values = fighters.value_counts().sort_values(ascending=False).head(10)

labels = values.index

fg=sns.barplot(x=values, y=labels)

plt.title("Most Wins by Fighter")

plt.ylabel("Wins")

plt.xlabel("# of Wins")

plt.tight_layout()

fg
countsT = UFC['title_bout'].value_counts()

labels = 'False' ,'True'

sizes = countsT.values

explode = (0.1, 0.1) 

fig1, title = plt.subplots()

title.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

title.axis('equal')  

plt.title('Title Fight Occurence')

display(title)
#dist plots of Height across mult weight classes hopefullysns.distplot(fighters['Height'], bins=50)

UFC['R_age'] = pd.to_numeric(UFC['R_age'], errors='coerce')

UFC = UFC.dropna(subset=['R_age'])

d=sns.distplot(UFC['R_age'],bins=26)

plt.title('Fighter Age Distribuiton')

plt.xlabel('Fighter Age')

plt.ylabel('Frequency')

d
# code obtained from Kaggle user "rizkigt"

colors = ['red', 'blue', 'violet']

plt.figure(figsize=(15,8))

P=UFC['Winner'].value_counts()[:10].plot.pie(explode=[0.05,0.05,0.05],autopct='%1.1f%%',colors=colors,shadow=True)

plt.title('Winner by Corner')

P
#fighter stances

Stance = sns.countplot(x="Stance",order=['Orthodox','Southpaw','Switch'], data=Fighter_df)

plt.title("Number of Fighters by Stance")

plt.ylabel("Number of Fighters")

plt.xlabel("Stance Type")

plt.xticks(size=12)

plt.tight_layout()

Stance



colors = ['red', 'blue', 'violet']

plt.figure(figsize=(15,8))

P=UFC['Winner'].value_counts()[:10].plot.pie(explode=[0.05,0.05,0.05],autopct='%1.1f%%',colors=colors,shadow=True)

plt.title('Winner by Corner')

P
target_0 = UFC.loc[UFC['weight_class'] == 'Heavyweight']

target_1 = UFC.loc[UFC['weight_class'] == 'Light Heavyweight']

target_2 = UFC.loc[UFC['weight_class'] == 'Middleweight']

target_3 = UFC.loc[UFC['weight_class'] == 'Welterweight']

target_4 = UFC.loc[UFC['weight_class'] == 'Lightweight']

target_5 = UFC.loc[UFC['weight_class'] == 'Featherweight']

target_6 = UFC.loc[UFC['weight_class'] == 'Bantamweight']

target_7 = UFC.loc[UFC['weight_class'] == 'Flyweight']



hw=target_0.R_Height_cms

lhw=target_1.R_Height_cms

mw=target_2.R_Height_cms

ww=target_3.R_Height_cms

lw=target_4.R_Height_cms

fw=target_5.R_Height_cms

bw=target_6.R_Height_cms

fly=target_7.R_Height_cms





fig, ax = plt.subplots(figsize=(14, 6))

a=sns.kdeplot(hw, shade=True, label='Heavyweight')

b=sns.kdeplot(lhw, shade=True, label='Light Heavyweight')

c=sns.kdeplot(mw, shade=True, label='Middleweight')

d=sns.kdeplot(ww, shade=True, label='Welterweight')

e=sns.kdeplot(lw, shade=True, label='Lightweight')

f=sns.kdeplot(fw, shade=True, label='Featherweight')

g=sns.kdeplot(bw, shade=True, label='Bantamweight')

h=sns.kdeplot(fly, shade=True, label='Flyweight')



plt.xlabel('Height')

plt.title('Height Difference Across weight classes')

a
target_0 = UFC.loc[UFC['weight_class'] == 'Heavyweight']

target_1 = UFC.loc[UFC['weight_class'] == 'Light Heavyweight']

target_2 = UFC.loc[UFC['weight_class'] == 'Middleweight']

target_3 = UFC.loc[UFC['weight_class'] == 'Welterweight']

target_4 = UFC.loc[UFC['weight_class'] == 'Lightweight']

target_5 = UFC.loc[UFC['weight_class'] == 'Featherweight']

target_6 = UFC.loc[UFC['weight_class'] == 'Bantamweight']

target_7 = UFC.loc[UFC['weight_class'] == 'Flyweight']



hw=target_0.R_Reach_cms

lhw=target_1.R_Reach_cms

mw=target_2.R_Reach_cms

ww=target_3.R_Reach_cms

lw=target_4.R_Reach_cms

fw=target_5.R_Reach_cms

bw=target_6.R_Reach_cms

fly=target_7.R_Reach_cms





fig, ax = plt.subplots(figsize=(14, 6))

a=sns.kdeplot(hw, shade=True, label='Heavyweight')

b=sns.kdeplot(lhw, shade=True, label='Light Heavyweight')

c=sns.kdeplot(mw, shade=True, label='Middleweight')

d=sns.kdeplot(ww, shade=True, label='Welterweight')

e=sns.kdeplot(lw, shade=True, label='Lightweight')

f=sns.kdeplot(fw, shade=True, label='Featherweight')

g=sns.kdeplot(bw, shade=True, label='Bantamweight')

h=sns.kdeplot(fly, shade=True, label='Flyweight')



plt.xlabel('Reach')

plt.title('Reach Difference Across Weight classes')
df_num = df.select_dtypes(include=[np.float, np.int])

y = df['Winner']

X = df.drop(columns = 'Winner')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=43)

model.fit(X_train, y_train)

y_preds = model.predict(X_test)

accuracy_score(y_test, y_preds)
model.feature_importances_

scaler = StandardScaler()

df[list(df_num.columns)] = scaler.fit_transform(df[list(df_num.columns)])
y = df['Winner']

X = df.drop(columns = 'Winner')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate predictions

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# import the class

from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression(solver='liblinear')



# fit the model with data

logreg.fit(X_train,y_train)



#

y_pred=logreg.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
