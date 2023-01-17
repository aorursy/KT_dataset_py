# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

data=pd.read_csv('../input/ks-projects-201801.csv')
data.shape
data.head()
data.columns.values
data.drop('ID', axis = 1, inplace = True)

data.drop('goal', axis = 1, inplace = True)

data.drop('pledged', axis = 1, inplace = True)

data.drop('usd pledged', axis = 1, inplace = True)
data['deadline']=pd.to_datetime(data['deadline'], format="%Y/%m/%d").dt.date

data['launched']=pd.to_datetime(data['launched'], format="%Y/%m/%d").dt.date
data['days'] = (data['deadline'] - data['launched']).dt.days

data['launch_year']=pd.to_datetime(data['launched'], format="%Y/%m/%d").dt.year
data.head()
data.isnull().sum()
data['state'].value_counts()
success_projects = data[data['state'] == 'successful']['state'].count()

fail_projects  = data[data['state'] == 'failed']['state'].count()

others_projects  = (

    data[data['state'] == 'canceled']['state'].count() +

    data[data['state'] == 'live']['state'].count() +

    data[data['state'] == 'undefined']['state'].count() +

    data[data['state'] == 'suspended']['state'].count())
total=success_projects+fail_projects+others_projects

suc=success_projects/total

fail=fail_projects/total

other=others_projects/total
sizes = [suc, fail, other]

labels="suc","fail","others"

explode = (0.1, 0, 0)

plt.pie(sizes,labels=labels,explode=explode,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title("Total Success Rate")
#Plotting the Main_category Distrubation

plt.subplots(figsize=(19,5))

sns.countplot(x="main_category",data=data)
plt.subplots(figsize=(19,5))

sns.countplot(x="country",data=data)
data["launch_year"]=data['launch_year'].apply(str)
ax = sns.countplot(data.launch_year)

plt.xlabel("Year")

plt.ylabel("Number of Campaigns")

plt.title("Number of Campaigns vs Year")

plt.show(ax)
launchyear = data.launch_year.value_counts().index
backersort = []

for i in launchyear:

    new = data[data.launch_year == i]

    backersort.append((i,new.backers.sum()))

backersort = pd.DataFrame(backersort, columns = ["Yıl","Backers"])

ax = sns.barplot(x="Yıl", y="Backers", data=backersort)
datasuccess = data[data.state == "successful"]

datafail = data[data.state == "failed"]
suc_categories = datasuccess.groupby("main_category")["usd_pledged_real"].sum()

suc_categories=suc_categories/1000000

failed_categories = datafail.groupby("main_category")["usd_pledged_real"].sum()

failed_categories=failed_categories/1000000
failed_categories_goals = datafail.groupby("main_category")["usd_goal_real"].sum()

failed_categories_goals=failed_categories_goals/1000000

success_categories_goals = datasuccess.groupby("main_category")["usd_goal_real"].sum()

success_categories_goals=success_categories_goals/1000000
f, axarr = plt.subplots(2,2, figsize=(30, 10), sharex=True)

ax1=failed_categories_goals.plot.bar(ax=axarr[0][0])

ax1.set_title("Most Failed Categories in USD Goals")

ax2=success_categories_goals.plot.bar(ax=axarr[1][0])

ax2.set_title("Most Successful Categories in USD Goals")

ax3=suc_categories.plot.bar(ax=axarr[0][1])

ax3.set_title("Money raised (in USD Million) by Main Category ")

ax4=failed_categories.plot.bar(ax=axarr[1][1])

ax4.set_title("Money reduced (in USD Million) by Main Category ")
f, axarr = plt.subplots(2,1, figsize=(20, 6), sharex=True)

ax=sns.countplot(datasuccess.main_category,ax=axarr[0])

ax.set_title("Successful Campaigns Main Categories")

ax1=sns.countplot(datafail.main_category,ax=axarr[1])

ax1.set_title("Failed Main Campaigns Categories")
data.corr()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

objects = ["category","main_category","currency","state","country"]

for i in objects:  

    data[i] = le.fit_transform(data[i])



corr = data.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(900, 100, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot_kws={'size': 12})

plt.savefig('corr.png')
#State = 3 success,fail=1

successrate = []

for i in launchyear:

    try:

        new = data[data.launch_year == i]

        successrate.append((i,new.state.value_counts()[3] / new.state.value_counts().sum()))

    except:

        successrate.append((i,0))



successrate = pd.DataFrame(successrate, columns = ["Yıl","Oran"])

ax = sns.barplot(x="Yıl", y="Oran", data=successrate)  

ax.set_title("Successful Rate")

ax.set_xlabel("Year")

ax.set_ylabel("Ratio")

#Data Splitting Libraries

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report



#Classification Libraries

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier
data_for_model=pd.read_csv('../input/ks-projects-201801.csv')
#Datetime Processing

data_for_model['deadline']=pd.to_datetime(data_for_model['deadline'], format="%Y/%m/%d")

data_for_model['launched']=pd.to_datetime(data_for_model['launched'], format="%Y/%m/%d")



data_for_model['days'] = (data_for_model['deadline'] - data_for_model['launched']).dt.days

data_for_model['launch_year']=pd.to_datetime(data_for_model['launched'], format="%Y/%m/%d").dt.year

data_for_model.drop(['ID',"name","category","launched","currency","deadline","usd pledged","goal","pledged"], axis = 1, inplace = True)
data_for_model["launch_year"]=data_for_model['launch_year'].apply(str) #it has to be string.
data_for_model.corr()
#Relationship with backers and usd pledged real data

sns.jointplot(x="backers", y="usd_pledged_real", data=data_for_model, kind="reg");
print("Unbalanced Data shape", len(data))

datafail = data_for_model[data_for_model.state == "failed"]

datasuccess = data_for_model[data_for_model.state == "successful"]

data_for_model = pd.concat([datafail.sample(len(datasuccess), random_state=5), datasuccess])

print("Balanced data shape:", len(data))
data_for_model.state.value_counts()
def state_process(cell_value):

    if cell_value == 'successful':

        return 1

    else:

        return 0    

data_for_model.state = data_for_model.state.apply(state_process)
data_for_model.head()
print('Original Features:\n', list(data.columns), '\n')

data_for_model= pd.get_dummies(data_for_model)

print('Features after One-Hot Encoding:\n', list(data_for_model.columns))
data_for_model.shape
X = data_for_model.iloc[:,data_for_model.columns != 'state']

y = data_for_model.state

print("X Columns: ",list(X.columns))
def ML_training(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    

    log_lm = LogisticRegression()



    log_lm.fit(X_train, y_train)

    logy_pred = log_lm.predict(X_test)

    acclr = accuracy_score(y_test, logy_pred)*100

    logquestions = pd.DataFrame({'features': X.columns,'Coef': (log_lm.coef_[0])*100})

    logquestions = logquestions.sort_values(by='Coef', ascending=False)

    ##############################

    dtree=DecisionTreeClassifier()



    dtree.fit(X_train, y_train)

    dtreey_pred = dtree.predict(X_test)

    accdtree = accuracy_score(y_test, dtreey_pred)*100

    

    dtreequestions = importance(dtree)

    ##############################

    rf = RandomForestClassifier(n_estimators=150,random_state=431, max_depth=6, min_weight_fraction_leaf =0.1)



    rf.fit(X_train, y_train)

    rfy_pred = rf.predict(X_test)

    accrf = accuracy_score(y_test, rfy_pred)*100

    

    accquestions = importance(rf)

    

    # Reporting

    print("Logistic Regression Report")

    print(classification_report(y_test, logy_pred))

    print(confusion_matrix(y_test,logy_pred))

    print(logquestions)

    print("------------------------------------------------------")

    

    print("Decision Tree Report")

    print(classification_report(y_test, dtreey_pred))

    print(confusion_matrix(y_test,dtreey_pred))

    print(dtreequestions)

    print("------------------------------------------------------")

    

    print("Random Forest Report")

    print(classification_report(y_test, rfy_pred))

    print(confusion_matrix(y_test,rfy_pred))

    print(accquestions)

    print("------------------------------------------------------")

    
def importance(model):

    questions = pd.DataFrame({'features': X.columns,'importance': (model.feature_importances_)*100})

    questions.sort_values(by='importance', ascending=False)

    questions = questions.sort_values(by='importance', ascending=False)

    return questions
ML_training(X,y)
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.noise import GaussianNoise

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
def ANN_training(X,y):

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    

    scaler = MinMaxScaler((-1,1))

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    

    clf = Sequential()

    clf.add(Dense(input_shape = (X.shape[1],), units = 10, activation = 'relu'))

    clf.add(Dense(units = 6, activation = 'relu'))

    clf.add(Dense(units = 1, activation = 'sigmoid'))

    clf.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

    clf.fit(X_train, y_train, batch_size = 100, nb_epoch = 10)



    # Predicting results

    y_pred = clf.predict(X_test, batch_size = 10)

    y_pred = (y_pred > 0.5)



    cm = confusion_matrix(y_test, y_pred)

    print(cm)
ANN_training(X,y)