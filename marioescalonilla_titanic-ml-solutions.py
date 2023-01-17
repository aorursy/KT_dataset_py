import numpy as np

from numpy import *

import pandas as pd

import seaborn as sns

from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt



#Skewness Distribution and more

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

import datetime

from sklearn.preprocessing import RobustScaler  #Scaling before pipeline

from sklearn.pipeline import make_pipeline # transforming steps pipeline

from sklearn import ensemble

from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from pandas.plotting import scatter_matrix           # For representing variables in EDA

from imblearn.over_sampling import SMOTE             # For oversampling the Dataset

from sklearn import preprocessing                    # For label Encoding

from sklearn import model_selection ,metrics,linear_model

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,roc_curve, auc
train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')

id_test = test["PassengerId"]
train.shape
test.shape
train.head()
scatter_matrix(train, figsize = (12, 12), diagonal = 'kde');



sns.relplot(x="Age", y="Fare", hue="Survived", style="Survived",data=train,height=5, aspect=3);

plt.figure(figsize=(20,20))

sns.relplot(x="Age", y="Fare", size="Pclass", sizes=(15, 200), data=train,height=5, aspect=3)
pal = sns.cubehelix_palette(8)



ax2 = sns.catplot(y="Age",x="Pclass",data=train,kind="bar",palette=pal)

ax2.set(xlabel='Class')
ax = sns.violinplot(x="Survived", y="Age", data=train)

ax = sns.violinplot(x="Survived", y="Age", hue="Pclass",

                   data=train, linewidth=2.5)
pal = sns.cubehelix_palette(8, start=.5, rot=-.75)



 #Plot 1  

ax1 = sns.countplot(train["Pclass"] ,palette=pal)

ax1.set(xlabel='Class', ylabel='Nº Passengers')



class_grouped=train["Pclass"].value_counts().sort_index()

#Plot 2

ax2 = sns.catplot(y="Survived",x="Pclass",data=train,kind="bar",palette=pal)

ax2.set(xlabel='Class', ylabel='Probability Survive')


ax = sns.violinplot(x="Pclass", y="Age", kind="swarm", data=train,hue="Survived");

ax.set( xlabel='Class')





# When creating the legend, only use the first two elements

# to effectively remove the last two.

pal = sns.diverging_palette(220, 20, n=7)



ax2 = sns.catplot(y="Fare",x="SibSp",data=train,kind="bar",hue="Pclass",palette=pal)

ax2.set(xlabel='Siblings/Spouses')
pal = sns.light_palette("navy")



ax2 = sns.catplot(y="Survived",x="SibSp",data=train,kind="bar",palette=pal)

ax2.set(xlabel='Siblings/Spouses')
pal = sns.diverging_palette(220, 20, n=7)



ax2 = sns.catplot(y="Fare",x="Parch",data=train,kind="bar",hue="Pclass",palette=pal)

ax2.set(xlabel='Parents/Children')
pal = sns.light_palette("navy")



ax2 = sns.catplot(y="Survived",x="Parch",data=train,kind="bar",palette=pal)

ax2.set(xlabel='Parents/Children')
#Lets do this small change to represent better the family.It will be temporary,so don't worry..

train_aux= train.copy()

train_aux['FamilySize'] = train_aux['Parch'] + train_aux['SibSp'] + 1
pal = sns.light_palette("navy")



ax2 = sns.catplot(y="FamilySize",x="Pclass",data=train_aux,kind="bar",palette=pal)

ax2.set(xlabel='Class')
pal = sns.light_palette("navy")



ax2 = sns.catplot(y="Survived",x="Sex",data=train,kind="bar",palette=pal)

pal = sns.diverging_palette(220, 20, n=7)



ax2 = sns.catplot(y="Survived",x="Sex",data=train,kind="bar",hue="Pclass",palette=pal)

pal = sns.diverging_palette(220, 20, n=7)



ax2 = sns.catplot(y="Fare",x="Sex",data=train,kind="bar",hue="Pclass",palette=pal)

pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

 

ax = sns.countplot(train["Embarked"], palette=pal,order=['C','Q','S'])

ax.set(xlabel='Embarking Ports', ylabel='Nº Passengers')



ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])
pal = sns.diverging_palette(220, 20, n=7)

 

ax = sns.barplot(x=train["Embarked"], y=train["Survived"], palette=pal,order=['C','Q','S'])

ax.set(xlabel='Embarking Ports', ylabel='Probability Survive')



ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])
pal = sns.diverging_palette(220, 20, n=7)

 

ax = sns.barplot(x=train["Embarked"], y=train["Survived"], palette=pal,order=['C','Q','S'],hue=train["Sex"])

ax.set(xlabel='Embarking Ports', ylabel='Probability Survive')



ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])

plt.legend(bbox_to_anchor=(1.08, 0.7),

           bbox_transform=plt.gcf().transFigure,title="Sex", fancybox=True)
train["Cabin"].describe()
# This is the number of unknown passengers cabins

train["Cabin"].isnull().sum()
# We can take just the letter of the cabin as value so then is easy to plot them

train_aux1= train.copy()

train_aux1["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'U' for i in train_aux1['Cabin'] ])

train_aux1 = train_aux1[train_aux1.Cabin != 'U']

pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

 #Plot 1   

ax = sns.countplot(train_aux1["Cabin"] , order=['A','B','C','D','E','F','G','T'],palette=pal)

ax.set(xlabel='Cabin', ylabel='Nº Passengers')



cabin_grouped=train_aux1["Cabin"].value_counts().sort_index()



for i,index in enumerate(cabin_grouped):

    ax.text(i,index, cabin_grouped[i], color='black', ha="center")

pal = sns.color_palette("coolwarm", 7)

g = sns.catplot(y="Survived",x="Cabin",data=train_aux1,kind="bar",order=['A','B','C','D','E','F','G','T'],palette=pal)

g = g.set_ylabels("Probability Survive")
pal = sns.diverging_palette(220, 20, n=7)

a4_dims = (15, 10)

fig, ax = plt.subplots(figsize=a4_dims)

ax = sns.barplot(x=train_aux1["Embarked"], y=train_aux1["Survived"], palette=pal,order=['C','Q','S'],hue=train_aux1["Cabin"])

ax.set(xlabel='Embarking Ports', ylabel='Probability Survive')



ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])

plt.legend(bbox_to_anchor=(1.08, 0.7),

           bbox_transform=plt.gcf().transFigure,title="Sex", fancybox=True)
df_train = train.copy()

df_test  = test.copy()



print( "Train shape before:" + str(df_train.shape))

print( "Test shape before:" + str(df_test.shape))



#We dont need the Survived column  for now

survivor = df_train['Survived']

df_train.drop("Survived", axis = 1, inplace = True)



dim_train = df_train.shape[0]



print("-----------------------------------")

print( "Train shape after:" + str(df_train.shape))

print( "Test shape after:" + str(df_test.shape))





df_all = pd.concat((df_train,df_test)).reset_index(drop=True)

df_all['FamilySize'] = df_all['Parch'] + df_all['SibSp'] + 1
df_all.dtypes
id_variables = ["Ticket","PassengerId"]

df_all.drop(id_variables, axis = 1, inplace = True)
no_rows = len(df_all.index)

nameVar = []

missing_val = []

for var in df_all.columns:

    missing_per = round(((df_all[var].isna().sum())/no_rows)*100,2)

    if (missing_per) > 0:

        nameVar.append(var)

        missing_val.append(missing_per)



df_missVar= pd.DataFrame({'Variables':nameVar,'Percentage Missing':missing_val})

df_missVar = df_missVar.sort_values(by='Percentage Missing', ascending = False)



if(len(df_missVar.values) >= 1):

    plt.figure(figsize=(10,5))

    b = sns.barplot(x = df_missVar['Variables'],

                    y=df_missVar['Percentage Missing'])

    b.axes.set_title("Percentage of Missing values ",fontsize=10) 

    b.set_xlabel("Features",fontsize=10)

    b.set_ylabel("% Missing values",fontsize=10)

    b.tick_params(axis = 'x',labelsize=10,rotation=0)

    b.tick_params(axis = 'y',labelsize=10)

    

    missing_values=df_missVar['Percentage Missing'].values

    for i,index in enumerate(missing_values):

        b.text(i,index, str(round(index,4))+ ' %', color='black', ha="center",fontsize=10)
df_all.drop(['Cabin'], axis = 1, inplace = True)
#For NAME what is interesting for us is the Titles: Mr,Mrs,Miss..etc

df_all['Title'] = df_all['Name'].str.split(", ", expand=True)[1].str.split(".",

    expand=True)[0]

# We split the Title to get our desired part: Oreskovic, Miss. Marija -> Miss

# To clean up some rare Titles we will use the minimun value 10 for the count of Titles

# If a Title is less than 10 times in the column, it will become Misc

min_count = 10

title_names = (df_all['Title'].value_counts() < min_count)

df_all['Title'] = df_all['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)





# Replacing missing values in FARE, AGE, EMBARKED

df_all['Fare'].fillna(df_all.Fare.median(), inplace = True)

df_all['Age'].fillna(df_all.Age.median(), inplace = True)

df_all['Embarked'].fillna(df_all['Embarked'].mode()[0], inplace = True) 





# SibSp and Parch can be adjusted to a new column containing the Family Size

df_all['FamilySize'] = df_all['Parch'] + df_all['SibSp'] + 1 # +1 because of the own person also counts

df_all.info(5)
df_all.drop('Name', inplace= True, axis = 1)

# We save a copy of the dataframe for Codifying later

df_all_coded = df_all.copy() # before modifying the main dataset

df_all = pd.get_dummies(df_all).reset_index(drop=True)
df_all.info(5)
df_all['IsAlone'] = 0

df_all.loc[df_all['FamilySize'] == 1, 'IsAlone'] = 1
var_toDelete = ['SibSp','Parch','FamilySize'] 

df_all.drop(var_toDelete,1, inplace=True)

df_all.head()
plt.figure(figsize=(15,7))



ax=sns.distplot(df_all['Age'] , fit=norm,color="y")

ax.axes.set_title("Age Distribution ",fontsize=20) 





plt.figure(figsize=(15,7))

res = stats.probplot(df_all['Age'], plot=plt)
df_all['Age'].skew()
plt.figure(figsize=(15,7))



ax=sns.distplot(df_all['Fare'] , fit=norm,color="y")

ax.axes.set_title("Fare Distribution ",fontsize=20) 





plt.figure(figsize=(15,7))

res = stats.probplot(df_all['Fare'], plot=plt)
df_all['Fare'].skew()
from scipy.stats import boxcox_normmax



skew_index = ['Age','Fare']

for i in skew_index:

    df_all[i] =  boxcox1p(df_all[i],boxcox_normmax(df_all[i] +1 ))
sns.set(style="white")



corr = df_all.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(255, 5, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,square=True, linewidths=.9, cbar_kws={"shrink": .9})

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

ax.set_yticklabels(ax.get_xticklabels(),rotation=0,horizontalalignment='right')


#x = train_encoded.copy()

#y = x['Survived']

#x.drop('Survived',axis = 1, inplace = True)

#submission_test = test_encoded.copy()



X = df_all.copy()

X = X[:dim_train]

y = survivor.copy()

test = df_all.copy()

test  = test[dim_train:]



kf = KFold(n_splits=5, random_state=42, shuffle=True).get_n_splits(X.values)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.35, stratify=y,random_state=27)
X.shape
y.shape
def accu_cv(model):

    accu= cross_val_score(model, X, y, scoring="roc_auc", cv = kf)

    return(accu)
from sklearn.metrics import roc_auc_score

def accu(model):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return(roc_auc_score(y_test,pred))
svc_clf     = make_pipeline(RobustScaler(), SVC())

gbc         =  make_pipeline(RobustScaler(),GradientBoostingClassifier(n_estimators=100,

                                learning_rate=1.3,max_depth=2, random_state=1))

gnb_clf     = make_pipeline(RobustScaler(),GaussianNB())

knn_clf     = make_pipeline(RobustScaler(),KNeighborsClassifier(n_neighbors = 3))

dtree_clf   = make_pipeline(RobustScaler(),DecisionTreeClassifier())                    

rforest_clf = make_pipeline(RobustScaler(),RandomForestClassifier())

sdg_clf     = make_pipeline(RobustScaler(), SGDClassifier())

log_clf = make_pipeline(RobustScaler(), LogisticRegression())

etree_clf = make_pipeline(RobustScaler(), ensemble.ExtraTreesClassifier())

adaboost_clf = make_pipeline(RobustScaler(),ensemble.AdaBoostClassifier())

log_cv_clf = make_pipeline(RobustScaler(),LogisticRegressionCV())

list_clfs = {"Support Vector Classifier":svc_clf,

             "GradientBoostingClassifier":gbc,

             "Naive Bayes Classifier":gnb_clf,

             "KNearest":knn_clf,

             "DecisionTreeClassifier":dtree_clf,

             "RandomForestClassifier":rforest_clf,

             "Stochastic Gradient Descent":sdg_clf,

             "LogisiticRegression":log_clf,

             "LogisiticRegression_CV":log_cv_clf,

             "Ada Boost":adaboost_clf,

             "ExtraTrees":etree_clf,



            }



classif_score = {}

for key, clf in list_clfs.items():

    score = round(accu_cv(clf).mean(),5)

    classif_score[key] = score

    

score_df = pd.DataFrame(classif_score.items(), columns= ["Name","Score"]) 

score_df =score_df.sort_values(by='Score', ascending = False).reset_index(drop=True)    

score_df  
svc_clf.fit(X,y)

svc_clf.predict(test)



sub = pd.DataFrame()

sub["PassengerId"] = id_test

sub["Survived"] = svc_clf.predict(test)

sub.to_csv('titanic_svc.csv',index=False)
pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

plt.figure(figsize=(8,6))

ax = sns.barplot(x="Name", y="Score", data=score_df ,palette=pal)

ax.axes.set_title("Accuracy of Classifiers",fontsize=20) 

ax.set_xticklabels(ax.get_xticklabels(),rotation=30,horizontalalignment='right')



for index, row in score_df.iterrows():

    ax.text(index,row.Score, round(row.Score,3), color='black', ha="center")
voting_clf_hard = VotingClassifier(estimators=[('adaboost_clf', list_clfs["Ada Boost"]),

                        ('log_clf', list_clfs['LogisiticRegression']),

                        ('rf', list_clfs["RandomForestClassifier"]),

                        ('log_cv_clf', list_clfs["LogisiticRegression_CV"]),

                        ('svc',list_clfs["Support Vector Classifier"])], voting='hard',n_jobs=5)

def plot_importances(classifier,name, columns):

    indices = np.argsort(classifier.feature_importances_)[::-1][:6]

  

    pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

    ax = sns.barplot(y=X_train.columns[indices][:6],x = np.argsort(classifier.feature_importances_[indices][:]) , orient='h',ax=axes[row][col])

    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

    ax.set_xlabel("Score",fontsize=12)

    ay.set_xlabel("Features",fontsize=12)

    for index, row in score_info.iterrows():

        ax.text(index,row.Score, round(row.Score,2), color='black', ha="center")

        ax.set_title(" Feature Importance of " + key)
f,ax=plt.subplots(2,2,figsize=(10,8))

y_pred = cross_val_predict(make_pipeline(RobustScaler(), SVC()),X,y,cv=10)

aa= sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')

ax[0,0].set_title('Confusion SVC')

bottom, top = aa.get_ylim()#<-- This 3 lines are about to fix a bug with the Ylabel axis

aa.set_ylim(bottom + 0.5, top - 0.5)



y_pred = cross_val_predict(make_pipeline(RobustScaler(),ensemble.AdaBoostClassifier()),X,y,cv=10)

aa= sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')

ax[0,1].set_title('Confusion Matrix for Ada Boost ')

bottom, top = aa.get_ylim()

aa.set_ylim(bottom + 0.5, top - 0.5)



y_pred = cross_val_predict(make_pipeline(RobustScaler(), LogisticRegression()),X,y,cv=10)

aa = sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')

ax[1,0].set_title('Confusion Matrix for Logistic')

bottom, top = aa.get_ylim()

aa.set_ylim(bottom + 0.5, top - 0.5)



y_pred = cross_val_predict(make_pipeline(RobustScaler(), LogisticRegressionCV()),X,y,cv=10)

aa = sns.heatmap(confusion_matrix(y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')

ax[1,1].set_title('Confusion Matrix for Gradient Boosting')

bottom, top = aa.get_ylim()

aa.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
voting_clf_hard.fit(X,y)

sub_votting_hard = (voting_clf_hard.predict(test))



sub = pd.DataFrame()

sub["PassengerId"] = id_test

sub["Survived"] = sub_votting_hard

sub.to_csv('titanic_hard_.csv',index=False)