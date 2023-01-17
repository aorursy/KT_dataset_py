import warnings

warnings.filterwarnings('ignore')

import os

import random 

import re



import pandas as pd 

import numpy as np 

from scipy.stats import kurtosis, skew 

from scipy import stats



import matplotlib.pyplot as plt 

import seaborn as sns 



# Importing librarys to use on interactive graphs

import plotly.offline as plty

from plotly import tools

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot, plot 

import plotly.graph_objs as go 





# to set a style to all graphs

plt.style.use('fivethirtyeight')

init_notebook_mode(connected=True)

sns.set_style("whitegrid")

sns.set_context("paper")


df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

def DataDesc(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary


DataDesc(df_train)



DataDesc(df_test)





df_train['Title'] = df_train['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.',x).group(1).strip())

df_test['Title'] = df_test['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.',x).group(1).strip())



group = df_train.groupby('Title').size().rename('Count').reset_index()

fig = px.pie(group, 

             values='Count', names='Title', 

             color_discrete_sequence=["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51",'#457b9d'],

            title='Name Titles',

            width=600,

            height=400)



fig.update_layout(

    margin=dict(l=100, r=0, t=30, b=50),

    width = 800,

    height = 500,

    paper_bgcolor="#ffffff",

)

fig.show()
Title_Dictionary = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Mrs",

        "Ms":         "Mrs",

        "Mrs" :       "Mrs",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

                   }

    

# we map each title to the desired category

df_train['Title'] = df_train.Title.map(Title_Dictionary)

df_test['Title'] = df_test.Title.map(Title_Dictionary)
group = df_train.groupby(['Title','Survived']).size().rename('Count').reset_index()



fig  = px.histogram(data_frame=group, 

              x='Title', 

              y='Count', 

              color='Survived',

              color_discrete_sequence=["#0d3b66","#faf0ca"],

              template='plotly_white')



fig.update_layout(width=900, height=450, 

                  title= {'text': "Titles after grouping",

                          'y':0.95,'x':0.5,

                          'xanchor': 'center',

                          'yanchor': 'top'},

                 barmode='group',

                 showlegend=True,

                 margin = dict(l=25, r=10, t=50, b=10))
group = df_train.groupby(['Title']).agg({'Survived':'mean'}).reset_index()



fig  = px.bar(data_frame=group, 

              x='Title', 

              y='Survived',

              color='Survived',

              template='plotly_white',

              color_continuous_scale=["#b76935","#a56336","#935e38","#815839","#6f523b","#5c4d3c","#4a473e","#38413f","#263c41","#143642"])



fig.update_layout(width=900, height=450, 

                  title= {'text': "Chances of Survival based on Titles",

                          'y':0.95,'x':0.5,

                          'xanchor': 'center',

                          'yanchor': 'top'},

                 margin = dict(l=25, r=10, t=50, b=10))
age_died = df_train[(df_train["Age"] > 0) & 

                              (df_train["Survived"] == 0)]

age_surv = df_train[(df_train["Age"] > 0) & 

                              (df_train["Survived"] == 1)]



#figure size

plt.figure(figsize=(10,5))



# Ploting the 2 variables that we create and compare the two

sns.distplot(age_surv["Age"], bins=24, color='#0d3b66')

sns.distplot(age_died["Age"], bins=24, color='#f5dd90')

plt.title("Distribuition of Age",fontsize=20)

plt.xlabel("Age",fontsize=15)

plt.ylabel("")

plt.show()




df_train.groupby(['Pclass','Sex','Title'])['Age'].median()





df_train[df_train['Age'].isnull()]['Age'] = df_train.groupby(['Pclass','Sex','Title'])['Age'].transform('median')

df_test[df_test['Age'].isnull()]['Age'] = df_test.groupby(['Pclass','Sex','Title'])['Age'].transform('median')

fig = plt.figure(figsize=(15,5))



ax1 = fig.add_subplot(121)

_ = sns.distplot(age_surv["Age"], bins=24, color='#f5dd90', ax=ax1)

_ = ax1.set_title('Survived', fontsize=20)

_ = ax1.set_xlabel("Age",fontsize=15)

_ = ax1.set_ylabel("")



ax2 = fig.add_subplot(122)

_ = sns.distplot(age_died["Age"], bins=24, color='#0d3b66', ax=ax2)

_ = ax2.set_title('Not Survived', fontsize=20)

_ = ax2.set_xlabel("Age",fontsize=15)

_ = ax2.set_ylabel("")

#creating the intervals that we need to cut each range of ages

interval = (0, 5, 12, 18, 25, 35, 60, 120) 



#Seting the names that we want use to the categorys

categories = ['babies', 'Small Children', 'Big Children','Teen', 'Mid Aged', 'Full Aged', 'Senior']



# Applying the pd.cut and using the parameters that we created 

df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=categories)   



df_test["Age_cat"] = pd.cut(df_test.Age, interval, labels=categories) 
group = df_train.groupby(['Age_cat','Survived']).size().rename('Count').reset_index()



fig  = px.histogram(group, 

              x='Age_cat', 

              y='Count',

              color='Survived',

              color_discrete_sequence=["#457b9d","#fca311"],

              template='plotly_white')



fig.update_layout(width=900, height=400, 

                  barmode='group',

                  title= {'text': "Age Group Survival",

                          'y':0.95,'x':0.5,

                          'xanchor': 'center',

                          'yanchor': 'top'},

                 showlegend=True,

                 margin = dict(l=25, r=10, t=50, b=10))

                 

                 

fig.show()
fare_surv = df_train[(df_train['Fare'] > 0) & (df_train['Survived'] == 1)]

fare_died = df_train[(df_train['Fare'] > 0) & (df_train['Survived'] == 0)]



#figure size

plt.figure(figsize=(15,5))



# Ploting the 2 variables that we create and compare the two

sns.distplot(fare_surv["Fare"], bins=24, color='#f98948')

sns.distplot(fare_died["Fare"], bins=24, color='#9ba2ff')

plt.title("Distribuition of Fare",fontsize=20)

plt.xlabel("Fare",fontsize=15)

plt.ylabel("")

plt.show()
plt.figure(figsize=(12,5))



sns.boxenplot(x="Survived", y = 'Fare', 

              data=df_train[df_train['Fare'] > 0], palette=['#436a36','#802c3e']) 

plt.title("Fare Quartiles", fontsize=20) 

plt.xlabel("Survival", fontsize=18) 

plt.ylabel("Fare", fontsize=16) 

plt.xticks(fontsize=18)
fig = px.box(y = 'Fare',

              data_frame=df_train[(df_train['Fare'] > 0) & (df_train['Fare'] < 250)],

            color_discrete_sequence=["#262a10"],

              template='plotly_white') 



fig.update_layout(width=900, height=400, 

                  title= {'text': "Fare distribution",

                          'y':0.95,'x':0.5,

                          'xanchor': 'center',

                          'yanchor': 'top'},

                 margin = dict(l=25, r=10, t=50, b=10))

                 

                 

fig.show()
#Filling the NA's with -0.5

df_train['Fare'] = df_train['Fare'].fillna(-0.5)



#intervals to categorize



# -1 to 0 for Fare = -0.5/No information(null)



quant = (-1, 0, 8, 15, 31, 64, 600)



#Quartiles

label_quants = ['NA', 'Q1', 'Q2', 'Q3', 'Q4','UpperFence']





df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)



df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)
group = df_train.groupby(['Fare_cat','Survived']).size().rename('Count').reset_index()



fig  = px.histogram(group, 

              x='Fare_cat', 

              y='Count',

              color='Survived',

              color_discrete_sequence=["#3581b8","#fcb07e"],

              template='plotly_white')



fig.update_layout(width=900, height=400, 

                  barmode='group',

                  title= {'text': "Fare Quartiles Survival",

                          'y':0.95,'x':0.5,

                          'xanchor': 'center',

                          'yanchor': 'top'},

                 showlegend=True,

                 margin = dict(l=25, r=10, t=50, b=10))

                 

                 

fig.show()
cols = ['Name','Ticket','Fare','Cabin','Age']



def drop_col(df):

    for column in cols:

        df.drop(column, axis=1, inplace=True)



# Training Data

drop_col(df_train)



#Test Data

drop_col(df_test)

df_train.head()
fig = plt.figure(figsize=(20,20))



ax1 = fig.add_subplot(321)

_ = sns.countplot(df_train['Survived'], palette=['#9fb8ad','#475841'], ax=ax1)

_ = ax1.set_title('Survived', fontsize=20)

_ = ax1.set_xlabel("")

_ = ax1.set_ylabel("")

_ = ax1.set_xticklabels(['No','Yes'], fontsize=13)





ax2 = fig.add_subplot(322)

_ = sns.countplot(data=df_train, x='Sex',hue='Survived',palette=['#9fb8ad','#475841'], ax=ax2)

_ = ax2.set_title('Sex', fontsize=20)

_ = ax2.set_xlabel("")

_ = ax2.set_ylabel("")

_ = ax2.set_xticklabels(['Male','Female'], fontsize=13)



ax3 = fig.add_subplot(323)

_ = sns.countplot(data=df_train, x='Pclass',hue='Survived',palette=['#9fb8ad','#475841'], ax=ax3)

_ = ax3.set_title('Passenger Class', fontsize=20)

_ = ax3.set_xlabel("")

_ = ax3.set_ylabel("")

_ = ax3.set_xticklabels(['First','Second','Third'], fontsize=13)



ax4 = fig.add_subplot(324)

_ = sns.countplot(data=df_train, x='SibSp',hue='Survived',palette=['#9fb8ad','#475841'], ax=ax4)

_ = ax4.set_title('Siblings & Spouses', fontsize=20)

_ = ax4.set_xlabel("")

_ = ax4.set_ylabel("")

#_ = ax3.set_xticklabels(['First','Second','Third'], fontsize=13)



ax5 = fig.add_subplot(325)

_ = sns.countplot(data=df_train, x='Embarked',hue='Survived',palette=['#9fb8ad','#475841'], ax=ax5)

_ = ax5.set_title('Embarked', fontsize=20)

_ = ax5.set_xlabel("")

_ = ax5.set_ylabel("")

_ = ax5.set_xticklabels(['Southampton','Cherbourg','Queenstown'], fontsize=13)





ax6 = fig.add_subplot(326)

_ = sns.countplot(data=df_train, x='Parch',hue='Survived',palette=['#9fb8ad','#475841'], ax=ax6)

_ = ax6.set_title('Parents & Children', fontsize=20)

_ = ax6.set_xlabel("")

_ = ax6.set_ylabel("")

#_ = ax5.set_xticklabels(['Southampton','Cherbourg','Queenstown'], fontsize=13)
_ = sns.factorplot(x="Sex",y="Survived",data=df_train,

                   kind="bar", height = 5,aspect= 1, palette = ["#456990","#ef8354"])

_ = plt.ylabel("Probability(Survive)", fontsize=15)

_ = plt.xlabel("Sex", fontsize=15)

_ = plt.show()

_ = sns.factorplot(x="Title",y="Survived",data=df_train,

                   kind="bar", height = 5,aspect= 2, palette = ["#114b5f","#028090","#e4fde1","#456990","#f45b69"])

_ = plt.ylabel("Probability(Survive)", fontsize=15)

_ = plt.xlabel("Title", fontsize=15)



_ = plt.show()
_ = sns.factorplot(x="SibSp",y="Survived",data=df_train,

                   kind="bar", height = 5,aspect= 2.5, palette = ["#5f0f40","#9a031e","#fb8b24","#e36414","#0f4c5c"])

_ = plt.ylabel("Probability(Survive)", fontsize=15)

_ = plt.xlabel("SibSp Number", fontsize=15)



_ = plt.show()
_ = sns.factorplot(x="Parch",y="Survived",data=df_train,

                   kind="bar", height = 5,aspect= 2.5, palette = ["#5f0f40","#9a031e","#fb8b24","#e36414","#0f4c5c"])

_ = plt.ylabel("Probability(Survive)", fontsize=15)

_ = plt.xlabel("Parch Number", fontsize=15)



_ = plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder





rf = RandomForestClassifier()



params = {

                'max_depth' : [1,2,3,4,5,6],

               'min_samples_leaf' : [0.01,0.02,0.04,0.06],

                'max_features' : [0.1,0.2,0.4,0.8],

                'n_estimators' : [100,150,200,250,300]

                

        }



rf_cv = RandomizedSearchCV(estimator=rf,

                          param_distributions=params,

                           n_iter=100,

                          cv=10,

                          scoring='accuracy',

                          n_jobs=-1,

                           verbose=3

                          )

X = df_train.drop(['PassengerId','Survived'], axis=1)

y = df_train['Survived']



# Label Encoding

for col in ['Embarked','Title','Age_cat','Fare_cat','Sex']:

    X[col] = LabelEncoder().fit_transform(X[col].astype(str))



# Training

rf_cv.fit(X, y)





#Best Estimator

rf_best = rf_cv.best_estimator_
# Get feature importance

selected_features = X.columns.to_list()

feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])

feature_importance["Feature Importance"] = rf_best.feature_importances_



# Sort by feature importance

feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)



# Set graph style

sns.set(font_scale = 1)

sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",

               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",

               'ytick.color': '0.4'})



# Set figure size and create barplot

f, ax = plt.subplots(figsize=(12, 5))

sns.barplot(x = "Feature Importance", y = "Feature Label",

            palette = reversed(sns.color_palette('winter', 15)),  data = feature_importance)



# Generate a bolded horizontal line at y = 0

ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)



# Turn frame off

ax.set_frame_on(False)



# Tight layout

plt.tight_layout()



# Save Figure

plt.savefig("feature_importance.png", dpi = 1080)
import eli5 

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rf_best, random_state=105).fit(X, y)

eli5.show_weights(perm, feature_names = X.columns.to_list())
from sklearn.tree import export_graphviz



estimator = rf_best.estimators_[10]

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = selected_features,

                class_names = ['Not Survived','Survived'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
import shap 



explainer = shap.TreeExplainer(rf_best)

shap_values = explainer.shap_values(X)



shap.summary_plot(shap_values[1], X, plot_type="bar")
shap.summary_plot(shap_values[1], X)
df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\

                          prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)



df_test = pd.get_dummies(df_test, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\

                         prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)
df_train.head(3)
from sklearn.model_selection import train_test_split



# For Training & Validation

X = df_train.drop(['PassengerId','Survived'], axis=1)

y = df_train['Survived']





# For submission.csv

X_test = df_test.drop(['PassengerId'], axis=1)







# Creating train & validation sets, stratified sampling on Survived(dataframe = y)

X_train, X_val, y_train, y_val = train_test_split(X, y, 

                                                 test_size=0.25, 

                                                 random_state=123,

                                                 stratify=y)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X = scaler.fit_transform(X)



X_train = scaler.fit_transform(X_train)



X_val = scaler.fit_transform(X_val)



X_test = scaler.fit_transform(X_test)
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegression(),

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]
from sklearn.model_selection import ShuffleSplit, KFold

from sklearn.model_selection import cross_validate



shuffle_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 123 )

kfold = KFold(n_splits=10, random_state=123)





MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



results, names  = [], [] 

#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    cv_results = cross_validate(alg, X, y, cv  = kfold)

    

    names.append(MLA_name)

    results.append(cv_results['test_score'])

    

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    

    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
# boxplot algorithm comparison

fig = plt.figure(figsize=(15,6))

fig.suptitle('Classifier Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names[:10], y=results[:10])

ax.set_xticklabels(names)

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("Accuracy of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

plt.show()
# boxplot algorithm comparison

fig = plt.figure(figsize=(15,6))

fig.suptitle('Classifier Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names[10:], y=results[10:])

ax.set_xticklabels(names[10:])

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("Accuracy of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

plt.show()
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



C = np.logspace(-4,4,9)

tol = np.logspace(-6,2,9)





param_grid = {'C':C, 'tol':tol}



svc_model = SVC(probability=True)

svc_grid = GridSearchCV(svc_model, param_grid=param_grid, cv=kfold, verbose=1)



svc_grid.fit(X, y)



svc_best = svc_grid.best_estimator_
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

from sklearn.metrics import classification_report, confusion_matrix

y_pred = svc_best.predict(X_val)



y_pred_prob = svc_best.predict_proba(X_val)[:,1]



print(confusion_matrix(y_val, y_pred))



print(classification_report(y_val, y_pred))



cnf_matrix = confusion_matrix(y_val, y_pred)

class_names = ['Did not Survive','Survived']

np.set_printoptions(precision=2)





plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 

                      title='Normalized confusion matrix')
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Titanic classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr,tpr)
from keras.models import Sequential

from keras.layers import Dense



from keras.layers import BatchNormalization, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import RMSprop





# MODEL

model = Sequential()





# LAYERS

# 1.

model.add(Dense(18,input_shape=(X_train.shape[1],),activation='relu', kernel_initializer='uniform'))

model.add(Dropout(0.5))

model.add(BatchNormalization())





# 2.

model.add(Dense(60, activation='relu', kernel_initializer='uniform'))

model.add(Dropout(0.5))

model.add(BatchNormalization())





# OutPut layer

model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))











# COMPILE

model.compile(optimizer='adam',

             loss='binary_crossentropy',

              metrics=['accuracy'],

             )







# CALLBACKS

earlystop = EarlyStopping(monitor='val_loss',patience = 20)

modcheck = ModelCheckpoint('check_pt1.hdf5', save_best_only=True)
# TRAIN



history= model.fit(X_train, y_train,

                     validation_data=(X_val, y_val),

                     callbacks=[earlystop,modcheck],

                     epochs=500,

                     batch_size=30,

                     verbose=2)
# summarizing historical accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarizing historical accuracy

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()