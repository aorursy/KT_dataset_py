# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.figure_factory as ff

from plotly.offline import iplot

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import cross_val_score,GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier, cv, Pool

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings  

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/bank-marketing-dataset/bank.csv")

df.info()
df.head()
df.describe()
df_age=df[["age","deposit"]]

df_age.describe()
sns.violinplot(x="deposit",y="age",data=df_age)

plt.show()
plt.figure(figsize=(17,10))

sns.countplot("age",data=df,hue="deposit")

plt.show()
df["age_bin"]=pd.cut(df.age,bins=[18,29,40,50,60,100],labels=['young','midAge','Adult',"old",'Elder'])

sns.countplot(x="age_bin",data=df,hue="deposit")

plt.show()
df_job=df[["job","deposit"]]

df_job.describe()
plt.figure(figsize=(15,10))

sns.countplot(x="job",hue="deposit",data=df)

plt.show()
df_marital=df[["marital","deposit"]]

df_marital.describe()
sns.countplot(x="marital",hue="deposit",data=df)

plt.show()
# Notice how divorced have a considerably low amount of balance.

fig = ff.create_facet_grid(

    df,

    x='duration',

    y='balance',

    color_name='marital',

    show_boxes=False,

    marker={'size': 10, 'opacity': 1.0},

    colormap={'single': 'rgb(165, 242, 242)', 'married': 'rgb(253, 174, 216)', 'divorced': 'rgba(201, 109, 59, 0.82)'}

)



iplot(fig, filename='facet - custom colormap')
sns.countplot(x="education",hue="deposit",data=df)

plt.show()
sns.countplot(x="default",hue="deposit",data=df)

plt.show()
sns.countplot(x="housing",hue="deposit",data=df)

plt.show()
sns.countplot(x="loan",hue="deposit",data=df)

plt.show()
sns.boxplot(x="deposit",y="balance",data=df)

plt.show()
b_df = pd.DataFrame()

b_df['balance_yes'] = (df[df['deposit'] == 'yes'][['deposit','balance']].describe())['balance']

b_df['balance_no'] = (df[df['deposit'] == 'no'][['deposit','balance']].describe())['balance']

b_df.drop(['count', '25%', '50%', '75%']).plot.bar(title = 'Balance and deposit statistics')

plt.show()
df["balance_cat"] = np.nan

df.loc[df['balance'] <0, 'balance_cat'] = 'negative'

df.loc[(df['balance'] >=0)&(df['balance'] <=5000), 'balance_cat'] = 'low'

df.loc[(df['balance'] >5000)&(df['balance'] <=20000), 'balance_cat'] = 'mid'

df.loc[(df['balance'] >20000), 'balance_cat'] = 'high'

sns.countplot(x="balance_cat",hue="deposit",data=df)

plt.show()

sns.countplot(x="contact",hue="deposit",data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x="month",hue="deposit",data=df,order=("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))

plt.show()
df["month_bins"]=pd.cut(df.day,bins=4,labels=["q1","q2","q3","q4"])

plt.figure(figsize=(10,5))

sns.countplot(x="month_bins",hue="deposit",data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x="day",hue="deposit",data=df)

plt.show()
df["day_cat"] = np.nan

df.loc[df['day'] <5, 'day_cat'] = '1'

df.loc[(df['day'] >=5)&(df['day'] <=9), 'day_cat'] = '2'

df.loc[(df['day'] >=10)&(df['day'] <=13), 'day_cat'] = '3'

df.loc[(df['day'] >=14)&(df['day'] <=21), 'day_cat'] = '4'

df.loc[(df['day'] >=22), 'day_cat'] = '5'

plt.figure(figsize=(10,5))

sns.countplot(x="day_cat",hue="deposit",data=df)

plt.show()
df["day_bins"]=pd.cut(df.day,bins=4,labels=["w1","w2","w3","w4"])

plt.figure(figsize=(10,5))

sns.countplot(x="day_bins",hue="deposit",data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.swarmplot(x="month",y="day",hue="deposit",data=df,order=("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x="campaign",hue="deposit",data=df)

plt.show()
df["campaign_cat"] = np.nan

df.loc[df['campaign'] ==1, 'campaign_cat'] = 0

df.loc[(df['campaign'] >1), 'campaign_cat'] = 1

sns.countplot(x="campaign_cat",hue="deposit",data=df)

plt.show()

plt.figure(figsize=(10,5))

df["pdays_bin"]=pd.cut(df.pdays,bins=3,labels=["c1","c2","c3"])

sns.countplot(x="pdays_bin",hue="deposit",data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x="previous",hue="deposit",data=df)

plt.show()
df["previous_cat"] = np.nan

df.loc[df['previous'] <=2, 'previous_cat'] = 0

df.loc[(df['previous'] >2), 'previous_cat'] = 1

sns.countplot(x="previous_cat",hue="deposit",data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x="poutcome",hue="deposit",data=df)

plt.show()
df.head()
df.drop(labels = ['duration'], axis = 1, inplace = True)

df['deposit']=df['deposit'].map({'yes':1,'no':0})

df = pd.get_dummies(df, columns=['job','marital','education',"month",'default','housing',"loan","contact","poutcome","age_bin","balance_cat","pdays_bin","day_cat","day_bins","month_bins"])

cor_deposit=df.corr()

cor_deposit["deposit"].sort_values(ascending=False)
x_train=df.drop(labels=['deposit'],axis=1)

y_train=df['deposit'].astype(int)

X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.25, random_state=2)



kfold = StratifiedKFold(n_splits=10)



# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())

classifiers.append(XGBClassifier(random_state = random_state))

classifiers.append(LGBMClassifier(random_state = random_state))

classifiers.append(CatBoostClassifier())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis",'XGBClassifier','LGBMClassifier','CatBoostClassifier']})

plt.figure(figsize=(20,10))

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

plt.axvline(0.74)

plt.axvline(0.72)

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cb = CatBoostClassifier()

cb.fit(X_test,Y_test)



y_pred=cb.predict(X_test)

y_true=pd.DataFrame(Y_test)

from sklearn.metrics import classification_report

cr=classification_report(y_true,y_pred,output_dict=True)

pd.DataFrame(cr)
plt.figure(figsize=(20,5))

fi=cb.get_feature_importance(prettified=True).head(10)

sns.barplot(x="Feature Id",y="Importances",data=fi)

score=cb.score(X_test,Y_test)

plt.title('Accuracy: '+str(score))

plt.show()