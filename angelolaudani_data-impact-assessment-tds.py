# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import matplotlib as mpl

from sklearn import preprocessing

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit
# import and loading

df = pd.read_csv("../input/StudentsPerformance.csv")
# data cleaning

df.rename(columns={"race/ethnicity": "ethnicity", 

"parental level of education":"parent_education",

"test preparation course":"preparation",

"math score":"m_score",

"reading score": "r_score",

"writing score": "w_score"}, inplace = True)



# feature engineering on the data to visualize and solve the dataset more accurately

df['avg_score'] = (df['m_score'] + df['r_score'] + df['w_score'])/3

df.head()
df.isnull().sum()
# start data exploration

corr=df.corr()

corr
ax = sns.heatmap(corr, annot=True, square=True, vmin=0, vmax=1, cmap="RdBu_r")
df.describe()
sns.set(rc={'figure.figsize':(10,7)})

sns.countplot(y = "ethnicity", data = df.sort_values("ethnicity"))

plt.show()
sns.countplot(x = "gender", data = df)

plt.show()
sns.countplot(x = "ethnicity", hue="gender", data = df.sort_values("ethnicity"))

plt.show()
# visualizing the differnt parental education levels

sns.countplot(x = "parent_education", data = df, order=df['parent_education'].value_counts().index)

plt.xticks(rotation=45)

plt.show()
# Come varia la distribuzione dei voti al variare del livello di educazione dei genitori?"

sns.set(rc={'figure.figsize':(20,7)})

fig, axs = plt.subplots(ncols=3)



sns.violinplot(x = "parent_education", y = "w_score",  data = df, ax=axs[0])

sns.violinplot(x = "parent_education", y = "r_score",  data = df, ax=axs[1])

sns.violinplot(x = "parent_education", y = "m_score",  data = df, ax=axs[2])

for ax in axs:

    ax.tick_params(labelrotation=45)

    ax.tick_params(labelsize=12)

plt.show()

# Come varia tra i gruppi la distribuzione dei voti?

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.violinplot(x = "ethnicity", y = "w_score",  data = df.sort_values('ethnicity'), ax=axs[0])

sns.violinplot(x = "ethnicity", y = "r_score",  data = df.sort_values('ethnicity'), ax=axs[1])

sns.violinplot(x = "ethnicity", y = "m_score",  data = df.sort_values('ethnicity'), ax=axs[2])

for ax in axs:

    ax.tick_params(labelrotation=45,labelsize=12)

plt.show()
# Come varia tra i generi la distribuzione dei voti?

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.barplot(x = "gender", y = "w_score",  data = df, ax=axs[0])

sns.barplot(x = "gender", y = "r_score",  data = df, ax=axs[1])

sns.barplot(x = "gender", y = "m_score",  data = df, ax=axs[2])



plt.show()
# Come influisce il tipo di pasto sulla distribuzione dei voti?

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.violinplot(x = "lunch", y = "w_score",  data = df, ax=axs[0])

sns.violinplot(x = "lunch", y = "r_score",  data = df, ax=axs[1])

sns.violinplot(x = "lunch", y = "m_score",  data = df, ax=axs[2])



plt.show()
# Come influisce il tipo di pasto sulla distribuzione dei voti?

sns.boxplot(x = "ethnicity", y = "avg_score",  data = df.sort_values("ethnicity"), hue="lunch")



plt.show()
df.groupby("lunch")["ethnicity"].value_counts().unstack()
v = df.groupby("lunch")["ethnicity"].value_counts().unstack()

lunch_ratio = v.div(v.sum(axis="rows"), axis="columns")



fig, axs = plt.subplots(2, 3, figsize=(12,12))

fig.suptitle('Percentuale tipo di lunch al variare del gruppo')

axs[0, 0].pie(lunch_ratio["group A"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[0, 0].set_title('Group A')

axs[0, 1].pie(lunch_ratio["group B"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[0, 1].set_title('Group B')

axs[0, 2].pie(lunch_ratio["group C"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[0, 2].set_title('Group C')

axs[1, 0].pie(lunch_ratio["group D"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[1, 0].set_title('Group D')

axs[1, 1].pie(lunch_ratio["group E"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[1, 1].set_title('Group E')

axs[1, 2].axis('off')



plt.show()
df.groupby("lunch")["parent_education"].value_counts().unstack()
v = df.groupby("lunch")["parent_education"].value_counts().unstack()

lunch_ratio = v.div(v.sum(axis="rows"), axis="columns")



fig, axs = plt.subplots(2, 3, figsize=(12,12))

fig.suptitle('Percentuale tipo di lunch al variare del titolo di studio dei genitori')

axs[0, 0].pie(lunch_ratio["associate's degree"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[0, 0].set_title("associate's degree")

axs[0, 1].pie(lunch_ratio["bachelor's degree"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[0, 1].set_title("bachelor's degree")

axs[0, 2].pie(lunch_ratio["high school"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[0, 2].set_title("high school")

axs[1, 0].pie(lunch_ratio["master's degree"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[1, 0].set_title("master's degree")

axs[1, 1].pie(lunch_ratio["some college"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[1, 1].set_title("some college")

axs[1, 2].pie(lunch_ratio["some high school"], labels=lunch_ratio.index.values, autopct='%1.1f%%')

axs[1, 2].set_title("some high school")



plt.show()
plt.figure(figsize=(9,6))

sns.countplot(x = "preparation", hue="gender", data = df)



plt.show()
fig, axs = plt.subplots(ncols=3)

colors = [ "amber", "windows blue","dusty purple","faded green", ]

color = sns.xkcd_palette(colors)



sns.violinplot(x = "preparation", y = "w_score", hue="gender", data = df, ax=axs[0])

sns.violinplot(x = "preparation", y = "r_score",  hue="gender", data = df, ax=axs[1],palette=color[2:])

sns.violinplot(x = "preparation", y = "m_score",  hue="gender", data = df, ax=axs[2], palette=color)

plt.show()
fig, axs = plt.subplots(ncols=3)

colors = [ "amber", "windows blue","dusty purple","faded green", ]

color = sns.xkcd_palette(colors)



sns.violinplot(x = "preparation", y = "w_score",  data = df, ax=axs[0])

sns.violinplot(x = "preparation", y = "r_score",  data = df, ax=axs[1],palette=color[2:])

sns.violinplot(x = "preparation", y = "m_score", data = df, ax=axs[2], palette=color)

plt.show()
sns.catplot(y="parent_education", hue="ethnicity", col="preparation", data=df.sort_values('ethnicity'), kind="count")

plt.show()
# Data to plot

labels = ['group A', 'group B', 'group C', 'group D','group E']

sizes_r = df.groupby('ethnicity')['r_score'].mean().values

sizes_w = df.groupby('ethnicity')['w_score'].mean().values

sizes_m = df.groupby('ethnicity')['m_score'].mean().values

 

# Plot

sns.set(rc={'figure.figsize':(16,11)})

fig, axs = plt.subplots(nrows=3)

fig.suptitle('Voto medio per materia al variare del gruppo')

ax = sns.barplot(y=labels ,x=sizes_r,data=df, ax = axs[0])

ax.set(ylabel='ethnicity', xlabel='r_score mean')

ax = sns.barplot(y=labels ,x=sizes_w,data=df, ax = axs[1])

ax.set(ylabel='ethnicity', xlabel='w_score mean')

ax = sns.barplot(y=labels ,x=sizes_m,data=df, ax = axs[2])

ax.set(ylabel='ethnicity', xlabel='m_score mean')

plt.show() 

# Assigning grades to the grades according to the following criteria :

# A: 90-100

# B: 80-89

# C: 70-79

# D: 60-69

# F: 0-59



def getgrade(score):

  if(score >= 90):

    return 'A'

  if(score >= 80):

    return 'B'

  if(score >= 70):

    return 'C'

  if(score >= 60):

    return 'D'

  else:

    return 'F'



df['grades_w'] = df.apply(lambda x: getgrade(x['w_score']), axis = 1 )

df['grades_r'] = df.apply(lambda x: getgrade(x['r_score']), axis = 1 )

df['grades_m'] = df.apply(lambda x: getgrade(x['m_score']), axis = 1 )



df.head(5)

sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=3)



sns.countplot(x = "grades_w", data = df.sort_values('grades_w'), ax=axs[0])

sns.countplot(x = "grades_r", data = df.sort_values('grades_r'), ax=axs[1])

sns.countplot(x = "grades_m", data = df.sort_values('grades_m'), ax=axs[2])

plt.show()
# setting a passing mark for the students to pass on the three subjects individually

passmarks = 60



# creating a new column pass_math, this column will tell us whether the students are pass or fail

df['pass_math'] = np.where(df['m_score']< passmarks, 'Fail', 'Pass')

df['pass_reading'] = np.where(df['r_score']< passmarks, 'Fail', 'Pass')

df['pass_writing'] = np.where(df['w_score']< passmarks, 'Fail', 'Pass')



# checking which student is fail overall



df['status'] = df.apply(lambda x : 'Fail' if x['pass_math'] == 'Fail' or 

                           x['pass_reading'] == 'Fail' or x['pass_writing'] == 'Fail'

                           else 'Pass', axis = 1)



sns.set(rc={'figure.figsize':(10,7)})

sns.countplot(x = "status", data = df)

plt.show()
sns.set(rc={'figure.figsize':(10,5)})

sns.countplot(x = "ethnicity", hue="status", data = df.sort_values("ethnicity"))

plt.show()
X_fair=pd.DataFrame(df[["gender", "ethnicity", "parent_education", "lunch", "preparation","status"]])

X=pd.DataFrame(df[["gender", "ethnicity", "parent_education", "lunch", "preparation"]])

label=df["status"]

X.head(5)
from sklearn.preprocessing import LabelEncoder



X["gender"]=LabelEncoder().fit_transform(X["gender"])

X["ethnicity"]=LabelEncoder().fit_transform(X["ethnicity"])

X["parent_education"]=LabelEncoder().fit_transform(X["parent_education"])

X["lunch"]=LabelEncoder().fit_transform(X["lunch"])

X["preparation"]=LabelEncoder().fit_transform(X["preparation"])



X.head(5)
X_train, X_test, y_train, y_test =  train_test_split(X, label, test_size=0.2, random_state=0)



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="blue")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="orange")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="orange",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "AdaBoost", "Naive Bayes", "QDA", 

    "Random Forest", "Decision Tree", "Log Regression"]



names = ["Linear SVM", "Random Forest", "Log Regression"]



classifiers = [

    #KNeighborsClassifier(3),

    SVC(kernel="linear", C=1, probability=1, random_state=1),

    #SVC(gamma=1, C=1, probability=1, random_state=1),

    #AdaBoostClassifier(random_state=1),

    #GaussianNB(),

    #QuadraticDiscriminantAnalysis(), 

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=1),

    #DecisionTreeClassifier(max_depth=5, random_state=1),

    LogisticRegression(C=1, random_state=2, solver='lbfgs')]





xx, yy = np.mgrid[-1:2:.01, -1:2:.01]

grid = np.c_[xx.ravel(), yy.ravel()]

decision_tree=0

fi=0

fi1=0

confusion_matrixs=[]



for name, model, ax in zip(names, classifiers, grid.flatten() ):

    clf = model.fit(X_train, y_train) 

    a=clf.score(X_test, y_test)

    #if name == "Decision Tree":

        #decision_tree=clf

        #fi=np.array(sorted(zip(X.columns[0:], clf.feature_importances_), key=lambda x: x[1], reverse=False))

    if name == ('Random Forest'):

        fi1=np.array(sorted(zip(X.columns[0:], clf.feature_importances_), key=lambda x: x[1], reverse=False))

    print(name)

    print("Score on test set: %0.3f" % (a))

    scores = cross_val_score(clf, X, label, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    predict = clf.predict(X_test)	

    cm=confusion_matrix(y_test, predict)

    confusion_matrixs.append(cm)



    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(model, name, X, label, ylim=(0.1, 1.05), cv=cv, n_jobs=-1)

    

    

    precision, recall, fscore, support = score(y_test, predict)

    print('precision: {}'.format(precision))

    print('recall: {}'.format(recall))

    print('fscore: {}'.format(fscore))

    print('support: {}'.format(support))

    print('confusion matrix: ')

    print(cm)

    print()

    print("____________________________________________")    

    

 
fig=plt.figure(figsize=(10, 10))

plt.bar(

    x=fi1[:, 0],

    height=fi1[:, 1],

    tick_label=fi1[:, 0]

)

plt.ylabel('Feature importance')

plt.xlabel('Feature')

plt.title('Importanza feature nel Random Forest')

plt.show()
if (len(names) <= 3):

    fig, axs = plt.subplots(nrows=1, ncols=3)

elif ((len(names) > 3) & (len(names) <= 6)):

    fig, axs = plt.subplots(nrows=2, ncols=3)

else:

    fig, axs = plt.subplots(nrows=3,ncols=3)

    

fig.tight_layout(h_pad=5)

sns.set(rc={'figure.figsize':(20,10)})



for i, cm , name in zip(enumerate(names), confusion_matrixs, names):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #normalizzo cm

    

    df_cm = pd.DataFrame(cm, index = ['fail', 'pass'],

                  columns = ['fail', 'pass'])

    

    if(i[0]<3):

        ax = sns.heatmap(df_cm, annot=True, square=True, vmin=0, vmax=1, cbar=False, cmap="RdBu_r", ax=axs[i[0] % 3])

    elif((i[0] >= 3) & (i[0]<6)):

        ax = sns.heatmap(df_cm, annot=True, square=True, vmin=0, vmax=1, cbar=False, cmap="RdBu_r", ax=axs[1,i[0] % 3])

    else:

        ax = sns.heatmap(df_cm, annot=True, square=True, vmin=0, vmax=1, cbar=False, cmap="RdBu_r", ax=axs[2,i[0] % 3])

    ax.set(title=name, ylabel='classe reale', xlabel='classe predetta')
predict = clf.predict(X)

df_predictions = pd.DataFrame(predict)

X_fair["predicted"] = df_predictions
X_fair.head(5)
sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=2)



sns.countplot(x = "lunch", hue="status", data = X_fair, ax=axs[0]).set_ylim(0,630)

sns.countplot(x = "lunch", hue="predicted", data = X_fair, ax=axs[1]).set_ylim(0,630)

plt.show()
v = X_fair.groupby("status")["lunch"].value_counts().unstack()

status_ratio = v.div(v.sum(axis="rows"), axis="columns")

print(status_ratio)
v = X_fair.groupby("predicted")["lunch"].value_counts().unstack()

predicted_ratio = v.div(v.sum(axis="rows"), axis="columns")

print(predicted_ratio)
fig, axs = plt.subplots(2, 2, figsize=(12,12))



fig.suptitle('Rapporto Fail/Pass reale e predetto per "lunch" a confronto')

axs[0, 0].pie(status_ratio["standard"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[0, 0].set_title('Standard Reale')

axs[0, 1].pie(predicted_ratio["standard"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[0, 1].set_title('Standard Predetto')

axs[1, 0].pie(status_ratio["free/reduced"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[1, 0].set_title('Free/Reduced Reale')

axs[1, 1].pie(predicted_ratio["free/reduced"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[1, 1].set_title('Free/Reduced Predetto')



plt.show()
sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=2)



sns.countplot(x = "ethnicity", hue="status", data = X_fair.sort_values("ethnicity"), ax=axs[0]).set_ylim(0,260)

sns.countplot(x = "ethnicity", hue="predicted", data = X_fair.sort_values("ethnicity"), ax=axs[1]).set_ylim(0,260)

plt.show()
v = X_fair.groupby("status")["ethnicity"].value_counts().unstack()

status_ratio = v.div(v.sum(axis="rows"), axis="columns")

print(status_ratio)
v = X_fair.groupby("predicted")["ethnicity"].value_counts().unstack()

predicted_ratio = v.div(v.sum(axis="rows"), axis="columns")

print(predicted_ratio)
fig, axs = plt.subplots(3, 4, figsize=(12,12))



fig.suptitle('Rapporto Fail/Pass reale e predetto a confronto')

axs[0, 0].pie(status_ratio["group A"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[0, 0].set_title('Group A Reale')

axs[0, 1].pie(predicted_ratio["group A"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[0, 1].set_title('Group A Predetto')

axs[0, 2].pie(status_ratio["group B"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[0, 2].set_title('Group B Reale')

axs[0, 3].pie(predicted_ratio["group B"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[0, 3].set_title('Group B Predetto')

axs[1, 0].pie(status_ratio["group C"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[1, 0].set_title('Group C Reale')

axs[1, 1].pie(predicted_ratio["group C"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[1, 1].set_title('Group C Predetto')

axs[1, 2].pie(status_ratio["group D"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[1, 2].set_title('Group D Reale')

axs[1, 3].pie(predicted_ratio["group D"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[1, 3].set_title('Group D Predetto')

axs[2, 0].pie(status_ratio["group E"], labels=status_ratio.index.values, autopct='%1.1f%%')

axs[2, 0].set_title('Group E Reale')

axs[2, 1].pie(predicted_ratio["group E"], labels=predicted_ratio.index.values, autopct='%1.1f%%')

axs[2, 1].set_title('Group E Predetto')

axs[2, 2].axis('off')

axs[2, 3].axis('off')



plt.show()
sns.set(rc={'figure.figsize':(18,6)})

fig, axs = plt.subplots(ncols=2)



sns.countplot(x = "ethnicity", hue="predicted", data = X_fair.sort_values("ethnicity"), ax=axs[0])

sns.countplot(x = "lunch", hue="predicted", data = X_fair, ax=axs[1])

plt.show()
# Y = 0

y0_filter = X_fair["status"] == "Fail"

# Y = 1

y1_filter = X_fair["status"] == "Pass"

# R = 0

r0_filter = X_fair["predicted"] == "Fail"

# R = 1

r1_filter = X_fair["predicted"] == "Pass"



def falseRates(a_filter):

    FP = X_fair[y0_filter & r1_filter & a_filter]

    TN = X_fair[y0_filter & r0_filter & a_filter]

    FN = X_fair[y1_filter & r0_filter & a_filter]

    TP = X_fair[y1_filter & r1_filter & a_filter]

    

    FP_n = FP.shape[0]

    FN_n = FN.shape[0]

    TP_n = TP.shape[0]

    TN_n = TN.shape[0]

    

    PPV = TP_n/(TP_n+FP_n)

    FPR = FP_n/(FP_n + TN_n)

    FNR = FN_n/(FN_n + TP_n)



    return PPV, FPR, FNR

    

def positiveRates(a_filter):    

    FP = X_fair[y0_filter & r1_filter & a_filter]

    TP = X_fair[y1_filter & r1_filter & a_filter]

    tot_F = X_fair[y0_filter & a_filter]

    tot_T = X_fair[y1_filter & a_filter]

    

    FP_n = FP.shape[0]

    TP_n = TP.shape[0]

    tot_F_n = tot_F.shape[0]

    tot_T_n = tot_T.shape[0]

    

    FP_per = FP_n/tot_F_n

    TP_per = TP_n/tot_T_n



    return FP_n, TP_n
a_filter = X_fair["lunch"] == "standard"

b_filter = X_fair["lunch"] == "free/reduced"



a_FP_per, a_TP_per = positiveRates(a_filter)

b_FP_per, b_TP_per = positiveRates(b_filter)

a_tot = X_fair[a_filter]

b_tot = X_fair[b_filter]



print("Veri positivi, lunch 'standard': {0:d}\nFalsi positivi, lunch 'standard': {1:d}\nTotali, lunch 'standard': {2:d}\n".format(a_TP_per, a_FP_per, a_tot.shape[0]))

print("Veri positivi, lunch 'free/reduced': {0:d}\nFalsi positivi, lunch 'free/reduced': {1:d}\nTotali, lunch 'free/reduced': {2:d}".format(b_TP_per, b_FP_per, b_tot.shape[0]))
a_filter = X_fair["lunch"] == "standard"

b_filter = X_fair["lunch"] == "free/reduced"



a_PPV, a_FPR, a_FNR = falseRates(a_filter)

b_PPV, b_FPR, b_FNR = falseRates(b_filter)



print("Valore predittivo positivo, lunch 'standard': {0:.2f}\nValore predittivo positivo, lunch 'free/reduced': {1:.2f}\n".format(a_PPV, b_PPV))

print("Tasso falsi positivi, lunch 'standard': {0:.2f}\nTasso falsi positivi, lunch 'free/reduced': {1:.2f}\n".format(a_FPR, b_FPR))

print("Tasso falsi negativi, lunch 'standard': {0:.2f}\nTasso falsi negativi, lunch 'free/reduced': {1:.2f}\n".format(a_FNR, b_FNR))
df_sep_lunch = pd.DataFrame(columns=['lunch', 'TP', 'FP', 'PPV', 'FPR', 'FNR', 'TPR'])

df_sep_lunch = df_sep_lunch.append({

     "lunch": "standard",

     "TP": a_TP_per,

     "FP": a_FP_per,

     "PPV": a_PPV,

     "FPR": a_FPR,

     "FNR": a_FNR,

     "TPR": 1-a_FNR

      }, ignore_index=True)

df_sep_lunch = df_sep_lunch.append({

     "lunch": "free/reduced",

     "TP": b_TP_per,

     "FP": b_FP_per,

     "PPV": b_PPV,

     "FPR": b_FPR,

     "FNR": b_FNR,

     "TPR": 1-b_FNR

      }, ignore_index=True)
df_sep_lunch
a_filter = X_fair["ethnicity"] == "group A"

b_filter = X_fair["ethnicity"] == "group B"

c_filter = X_fair["ethnicity"] == "group C"

d_filter = X_fair["ethnicity"] == "group D"

e_filter = X_fair["ethnicity"] == "group E"



a_FP_per, a_TP_per = positiveRates(a_filter)

b_FP_per, b_TP_per = positiveRates(b_filter)

c_FP_per, c_TP_per = positiveRates(c_filter)

d_FP_per, d_TP_per = positiveRates(d_filter)

e_FP_per, e_TP_per = positiveRates(e_filter)



a_tot = X_fair[a_filter]

b_tot = X_fair[b_filter]

c_tot = X_fair[c_filter]

d_tot = X_fair[d_filter]

e_tot = X_fair[e_filter]



print("Veri positivi, ethnicity 'group A': {0:d}\nFalsi positivi, ethnicity 'group A': {1:d}\nTotali, ethnicity 'group A': {2:d}\n".format(a_TP_per, a_FP_per, a_tot.shape[0]))

print("Veri positivi, ethnicity 'group B': {0:d}\nFalsi positivi, ethnicity 'group B': {1:d}\nTotali, ethnicity 'group B': {2:d}\n".format(b_TP_per, b_FP_per, b_tot.shape[0]))

print("Veri positivi, ethnicity 'group C': {0:d}\nFalsi positivi, ethnicity 'group C': {1:d}\nTotali, ethnicity 'group C': {2:d}\n".format(c_TP_per, c_FP_per, c_tot.shape[0]))

print("Veri positivi, ethnicity 'group D': {0:d}\nFalsi positivi, ethnicity 'group D': {1:d}\nTotali, ethnicity 'group D': {2:d}\n".format(d_TP_per, d_FP_per, d_tot.shape[0]))

print("Veri positivi, ethnicity 'group E': {0:d}\nFalsi positivi, ethnicity 'group E': {1:d}\nTotali, ethnicity 'group E': {2:d}".format(e_TP_per, e_FP_per, e_tot.shape[0]))
a_filter = X_fair["ethnicity"] == "group A"

b_filter = X_fair["ethnicity"] == "group B"

c_filter = X_fair["ethnicity"] == "group C"

d_filter = X_fair["ethnicity"] == "group D"

e_filter = X_fair["ethnicity"] == "group E"



letters = ["a","b","c","d","e"]



a_PPV, a_FPR, a_FNR = falseRates(a_filter)

b_PPV, b_FPR, b_FNR = falseRates(b_filter)

c_PPV, c_FPR, c_FNR = falseRates(c_filter)

d_PPV, d_FPR, d_FNR = falseRates(d_filter)

e_PPV, e_FPR, e_FNR = falseRates(e_filter)



print("Valore predittivo positivo, ethnicity 'group A': {0:.2f}\nValore predittivo positivo, ethnicity 'group B': {1:.2f}\nValore predittivo positivo, ethnicity 'group C': {2:.2f}\nValore predittivo positivo, ethnicity 'group D': {3:.2f}\nValore predittivo positivo, ethnicity 'group E': {4:.2f}\n".format(a_PPV, b_PPV, c_PPV, d_PPV, e_PPV))

print("Tasso falsi positivi, ethnicity 'group A': {0:.2f}\nTasso falsi positivi, ethnicity 'group B': {1:.2f}\nTasso falsi positivi, ethnicity 'group C': {2:.2f}\nTasso falsi positivi, ethnicity 'group D': {3:.2f}\nTasso falsi positivi, ethnicity 'group E': {4:.2f}\n".format(a_FPR, b_FPR, c_FPR, d_FPR, e_FPR))

print("Tasso falsi negativi, ethnicity 'group A': {0:.2f}\nTasso falsi negativi, ethnicity 'group B': {1:.2f}\nTasso falsi negativi, ethnicity 'group C': {2:.2f}\nTasso falsi negativi, ethnicity 'group D': {3:.2f}\nTasso falsi negativi, ethnicity 'group E': {4:.2f}\n".format(a_FNR, b_FNR, c_FNR, d_FNR, e_FNR))
df_sep_eth = pd.DataFrame(columns=['ethnicity', 'TP', 'FP', 'PPV', 'FPR', 'FNR', 'TPR'])

df_sep_eth = df_sep_eth.append({

     "ethnicity": "group A",

     "TP": a_TP_per,

     "FP": a_FP_per,

     "PPV": a_PPV,

     "FPR": a_FPR,

     "FNR": a_FNR,

     "TPR": 1-a_FNR

      }, ignore_index=True)

df_sep_eth = df_sep_eth.append({

     "ethnicity": "group B",

     "TP": b_TP_per,

     "FP": b_FP_per,

     "PPV": b_PPV,

     "FPR": b_FPR,

     "FNR": b_FNR,

     "TPR": 1-b_FNR

      }, ignore_index=True)

df_sep_eth = df_sep_eth.append({

     "ethnicity": "group C",

     "TP": c_TP_per,

     "FP": c_FP_per,

     "PPV": c_PPV,

     "FPR": c_FPR,

     "FNR": c_FNR,

     "TPR": 1-c_FNR

      }, ignore_index=True)

df_sep_eth = df_sep_eth.append({

     "ethnicity": "group D",

     "TP": d_TP_per,

     "FP": d_FP_per,

     "PPV": d_PPV,

     "FPR": d_FPR,

     "FNR": d_FNR,

     "TPR": 1-d_FNR

      }, ignore_index=True)

df_sep_eth = df_sep_eth.append({

     "ethnicity": "group E",

     "TP": e_TP_per,

     "FP": e_FP_per,

     "PPV": e_PPV,

     "FPR": e_FPR,

     "FNR": e_FNR,

     "TPR": 1-e_FNR

      }, ignore_index=True)
df_sep_eth