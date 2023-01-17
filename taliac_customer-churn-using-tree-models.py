# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load the data

df = pd.read_csv("/kaggle/input/telecom-churn/telecom_churn.csv")

df.sample(5)
#check missing values

df.info()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import math
#default churn rate

labels = 'Churn', "Stay"

sizes = [df.Churn[df['Churn'] == 1].count(), df.Churn[df['Churn'] == 0].count()]

explode = (0.1, 0)



fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

ax1.axis('equal')



plt.title("Proportion of customer churned and retained")



plt.show()
#Let's look at the relationship between variables



#preperation

df_hue = df.copy()

df_hue["Churn"] = np.where(df_hue["Churn"] == 0, "S", "C")

df_new = df_hue[["Churn","AccountWeeks", "DataUsage", "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"]]
#A master view at all the numerical variables

sns.pairplot(df_new, hue="Churn", palette="husl")
def boxplots (df, variables, n_rows=None, n_cols=None, hue="Churn"):

    '''Draw boxplots to examine the churn behavior of each continuous variable. 



    Args:

    variables: list. A list of variables that you would like to examine

    n_rows, n_cols: int. 

    hue: Because the context here is a churn analysis, so I set the default to "Churn". 

    

    Returns:

    boxplots of all the variables that were passed in the format of that you specify. 

    '''

    fig=plt.figure(figsize=(20,16))

    sns.set(palette='pastel')

    for i, var in enumerate(variables):

        ax=fig.add_subplot(n_rows, n_cols, i+1)

        sns.boxplot(y=var, x=hue, hue=hue, data=df, ax=ax).set_title(var)

    plt.show()
continue_variables = df[["CustServCalls", "MonthlyCharge", "DataUsage", "RoamMins", "DayMins", "OverageFee"]]



boxplots(df, continue_variables, 3, 2)
#Let's quickly look at the histogram of categorical varaibles

fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(x='DataPlan', hue = 'Churn',data = df_hue, ax=axarr[0], palette="pastel")

sns.countplot(x='ContractRenewal', hue = 'Churn',data = df_hue, ax=axarr[1], palette="pastel")
#Let's do a quick check to see if all the 0 in DataUsage is due to the O in DataPlan

#in percentage

nousage_perce = (len(df[df["DataUsage"]==0])/len(df))*100

noplan_perce = (len(df[df["DataPlan"]==0])/len(df)) *100

print("The percentage for people with 0 data usage is " + str(round(nousage_perce, 2)) + " percent")

print("The percentage for people with no data plan is " + str(round(noplan_perce, 2)) + " percent")



#in absolute number 

nousage = len(df[df["DataUsage"]==0])

noplan = len(df[df["DataPlan"]==0])

print(str(nousage) + " numbers of people have 0 data usage.")

print(str(noplan) + " numbers of people have 0 data plan.")

print("\nInsight: 598 people who did not purchase a data plan were also using data. Let's check the roaming data.")



#for the people who didn't have a data plan but still have data usage, to they belong to roaming?

#Roaming info

roam = len(df[df["RoamMins"]!=0])

print("\n" + str(roam) + " numbers of people used roaming.")

print("\nInsight: Seems like almost everyone uses Roaming and it does not relates to wheather you use data.")
#using a pie chart to visualize the relationship between the three variables



group_names=['no DataPlan', 'DataPlan']

group_size=[2411, (3333-2411)]



subgroup1_names=['no DataUse','DataUse']

subgroup1_size=[1813, (3333-1813)]



subgroup2_names=['no Roam', 'Roaming']

subgroup2_size=[(3333-3315), 3315]

 

# Create colors

a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

 

# First Ring (Outside)

fig, ax = plt.subplots(figsize=(8, 6))

ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[b(0.6), a(0.6)])

plt.setp( mypie, width=0.3, edgecolor='white')

 

# Second Ring (Inside)

mypie2, _ = ax.pie(subgroup1_size, radius=1.3-0.3, labels=subgroup1_names, labeldistance=0.7, colors=[a(0.4), b(0.4)])

plt.setp( mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)



# Third Ring (Most Inner)

mypie3, _ = ax.pie(subgroup2_size, radius=1.3-0.3-0.3, labels=subgroup2_names, labeldistance=0.7, colors=[a(0.2), b(0.2)])

plt.setp( mypie3, width=0.4, edgecolor='white')

plt.margins(0,0)

 

# show it

ax.set(title='DataPlan vs. DataUsage vs. Roam\n')

plt.show()

def piecharts (df, cat_variables, n_rows=None, n_cols=None, hue="Churn"):

    '''Draw pie charts to examine the churn behavior of each categorical variable. 



    Args:

    attributes: list. A list of attributes that you would like to examine

    n_rows, n_cols: int. 

    hue: Because the context here is a churn analysis, so I set the default to "Churn". 

    

    Returns:

    boxplots of all the attributes that were passed in the format of that you specify. 

    '''

    fig=plt.figure(figsize=(18,10))

    explode = (0, 0.1)

    labels=['Churn', 'Not Churn']

    for i, var in enumerate(cat_variables):

        df_0 = [len(df[(df[var]==0) & (df[hue]==1)]), len(df[(df[var]==0) & (df[hue]==0)])] #when the cat_var == 0

        

        ax0=fig.add_subplot(n_rows, n_cols, i+1)

        ax0.pie(df_0, explode=explode, labels=labels, autopct='%1.1f%%')

        ax0.set_title("Do not have {}".format(var))

        

    for i, var in enumerate(cat_variables):

        df_1 = [len(df[(df[var]!=0) & (df[hue]==1)]), len(df[(df[var]!=0) & (df[hue]==0)])] #when the cat_var == 1

        

        ax1=fig.add_subplot(n_rows, n_cols, i+1+n_rows)

        ax1.pie(df_1, explode=explode, labels=labels, autopct='%1.1f%%')

        ax1.set_title("Have {}".format(var))

    plt.show()
cat_variables = df[["DataPlan", "ContractRenewal", "DataUsage"]] 



piecharts(df, cat_variables, 3, 2)
df_model = df.copy()
#To investigate why churn rate is so different between group "Have Dataplan" and "No Dataplan". I create a new variable, peopel who didn't have a data plan but still used data

df_model["data_w_noplan"]=np.where((df_model["DataPlan"]==0) & (df_model["DataUsage"]!=0), 1, 0)
#let's see if it is useful for our prediction

fig, ax = plt.subplots(1, 2, figsize = (12, 6))

data_w_noplan = [len(df_model[(df_model["data_w_noplan"]==1) & (df_model["Churn"]==1)]), len(df_model[(df_model["data_w_noplan"]==1) & (df_model["Churn"]==0)])]

ax[0].pie(data_w_noplan, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

ax[0].set_title("Customers who used data but did not purchase a plan")



data_w_plan = [len(df_model[(df_model["data_w_noplan"]==0) & (df_model["Churn"]==1)]), len(df_model[(df_model["data_w_noplan"]==0) & (df_model["Churn"]==0)])]

ax[1].pie(data_w_plan, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

ax[1].set_title("Customers who used data but did not purchase a plan")
#it was useless. Let's drop that. 

df_model.drop(['data_w_noplan'], axis = 1)
#Let's see if it relates to still using a roaming service while didn't purchase a data plan. 

df_model["roam_w_noplan"]=np.where((df_model["DataPlan"]==0) & (df_model["RoamMins"]!=0), 1, 0)
fig, ax = plt.subplots(1, 2, figsize = (12, 6))

roam_w_noplan = [len(df_model[(df_model["roam_w_noplan"]==1) & (df_model["Churn"]==1)]), len(df_model[(df_model["roam_w_noplan"]==1) & (df_model["Churn"]==0)])]

ax[0].pie(roam_w_noplan, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

ax[0].set_title("Customers who roam but did not purchase a plan")



roam_w_plan = [len(df_model[(df_model["roam_w_noplan"]==0) & (df_model["Churn"]==1)]), len(df_model[(df_model["roam_w_noplan"]==0) & (df_model["Churn"]==0)])]

ax[1].pie(roam_w_plan, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

ax[1].set_title("Customers who roam but purchased a data plan")



#people who doesn't have a data plan and churn were more likely due to the fact that they still use the roaming service. 
#let's see if people with 0 data usage uses calls more often

sns.scatterplot(x='DayMins', y="DataUsage", hue = 'Churn', data = df_hue, palette="pastel")

#Seems like it is not that people with low data usage will call more often but that people who doesn't have data usage and still have make lots of calls are very likely to be churn. However, 0 data usage are mostly due to wheather they have a data plan or not.



#relationship between dataplan and Daymins

fig = plt.subplots()

sns.boxplot(x='DataPlan', y="DayMins", hue = 'Churn', data = df_hue, palette="pastel")
# create new cat_varibale for data usage

df["datausage_dummy"] = np.where(df["DataUsage"]!=0, 1, 0)
#For people who didn't purchase a data plan. I squared the data so that I amplify the affect of havnig lots of call minutes.  

df_model["DayMins_noplan"]=np.where((df_model["DataPlan"]==0) & (df_model["DayMins"]!=0), df_model.DayMins**2, df_model.DayMins**2)
fig, ax = plt.subplots(1, 2, figsize = (14, 6))

sns.boxplot(y="DayMins_noplan", x="Churn", hue = "Churn", ax=ax[0], data=df_model).set_title("\nDay Mins Distribution for customer who \n didn't purchase a data plan \n In Customer Attrition")

sns.boxplot(y="DayMins", x="Churn", hue = "Churn", ax=ax[1], data=df).set_title("\nOriginal Day Mins Distribution \n In Customer Attrition")

#Let's see if it helps with the model
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.model_selection import cross_val_score



from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import precision_recall_curve, auc, roc_curve
#dat preperation

label = df["Churn"]

df_train1 = df.iloc[:, 1:].copy()

feature_names = list(df_train1.columns.values)



#I seperate the data into train, valiation and test. We will reserve the test set till the end to test the performance of the best model.



#set, testset

X_trainval, X_test, y_trainval, y_test = train_test_split(df_train1, label, test_size = 0.2, random_state=1)

#train, validation set split

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size = 0.2, random_state=1)
def modelling(classifiers, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val):

    

    for i, clf in enumerate(classifiers):

        model = clf.__class__.__name__

        clf.fit(X_train, y_train)

        score = clf.score(X_val, y_val)

        cross_val = cross_val_score(clf, X_val, y_val)



        print("\n{}\nModel accuracy: {}.\nCross-valdation score: {}.".format(model, score, cross_val.mean()))
clf_list = [DummyClassifier(random_state=1), DecisionTreeClassifier(random_state=1, max_depth = 6), RandomForestClassifier(random_state=1, max_depth=8), GradientBoostingClassifier(random_state=1), XGBClassifier(random_state=1)]



modelling(clf_list)
def confusion_matrices(clfs, X_train=X_train, y_train=y_train, X_val=X_val, y_test=y_val):

    

    for i, clf in enumerate(clfs):

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        pred_y = clf.predict(X_val)

        matrix_norm = confusion_matrix(y_val, pred_y, normalize='all')

        matrix= confusion_matrix(y_val, pred_y)

        print("\n{} Confusion Matrix:\n{}\n{}".format(name, matrix, matrix_norm))
confusion_matrices(clf_list)
from sklearn.utils import resample
# Separate majority and minority classes

df_maj = df[df.Churn==0]

df_min = df[df.Churn==1]



print("The minority sample size is: {}".format(len(df_min))) #483



# Downsample majority class

df_maj_ds = resample(df_maj, replace=False,    # sample without replacement

                             n_samples=483,     # to match minority class

                             random_state=1) # reproducible results

# Combine minority class with downsampled majority class

df_ds = pd.concat([df_maj_ds, df_min])

 

# Display new class counts

df_ds.Churn.value_counts()

#dat preperation

label2 = df_ds["Churn"]

df_ds_train = df_ds.iloc[:, 1:].copy()

feature_names2 = list(df_ds_train.columns.values)

#set, testset split

Xds_train, Xds_val, yds_train, yds_val = train_test_split(df_ds_train, label2, test_size = 0.2, random_state=1)
modelling(clf_list, Xds_train, yds_train, Xds_val, yds_val)
confusion_matrices(clf_list, Xds_train, yds_train)
# balance that dataset first



# Separate majority and minority classes

df_maj2 = df_model[df_model.Churn==0]

df_min2 = df_model[df_model.Churn==1]



print("The minority sample size is: {}".format(len(df_min2))) #483



# Downsample majority class

df_maj2_ds = resample(df_maj2, replace=False,    # sample without replacement

                             n_samples=483,     # to match minority class

                             random_state=1) # reproducible results

# Combine minority class with downsampled majority class

df_ds2 = pd.concat([df_maj2_ds, df_min2])

 

# Display new class counts

df_ds2.Churn.value_counts()
#data preperation

label3 = df_ds2["Churn"]

df_ds_train2 = df_ds2.iloc[:, 1:].copy()

feature_names3 = list(df_ds_train2.columns.values)

#set, testset split

Xds_train2, Xds_val2, yds_train2, yds_val2 = train_test_split(df_ds_train2, label3, test_size = 0.2, random_state=1)
modelling(clf_list, Xds_train2, yds_train2, Xds_val2, yds_val2)
#tree after feature engineering

tree_ds2 = DecisionTreeClassifier(random_state=1, max_depth = 6).fit(Xds_train2, yds_train2)

print("Tree validation score: {:.2f}".format(tree_ds2.score(Xds_val2, yds_val2)))

tree_ds_crossval2 = cross_val_score(tree_ds2, Xds_val2, yds_val2)

print("Tree cross-validation score: {:.2f}".format(tree_ds_crossval2.mean()))

print("\nFeature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), tree_ds2.feature_importances_), feature_names3), 

             reverse=True)))
xg = XGBClassifier(random_state=1).fit(Xds_train, yds_train)

pred_xg = xg.predict(X_val)
fig= plt.subplots(figsize=(8, 6))

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_val, xg.predict_proba(X_val)[:, 1])

plt.plot(precision_rf, recall_rf, label="rf")



close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k', markersize=10, label='threshold 0.5 rf', fillstyle="none", mew=2)

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.legend(loc="best")
threshold = 0.68 #after different trails, this is the best



predicted_proba = xg.predict_proba(X_val)

tune_pred = (predicted_proba [:,1] >= threshold).astype('int')



#compare the accuracy scores

accuracy_adj = accuracy_score(y_val, tune_pred)

print("accurcy rate with 0.56 threshold {}".format(str(round(accuracy_adj,4,)*100))+"%")



accuracy = accuracy_score(y_val, pred_xg)

print("accurcy rate with 0.5 threshold {}".format(str(round(accuracy,4,)*100))+"%")



#confusion matrix compare

confusion_tune=confusion_matrix(y_val, tune_pred)

print("confusion matrix with new threshold:\n{}".format(confusion_tune))



confusion_ds=confusion_matrix(y_val, pred_xg)

print("\nconfusion matrix original:\n{}".format(confusion_ds))



#classification_report

print("\nxgboost  classification report with adjuested threshold\n")

print(classification_report(y_val, tune_pred, target_names = ["Stay", "Churn"]))
print("XG boost final test score: {:.2f}".format(xg.score(X_test, y_test)))

test_crossval = cross_val_score(xg, df_train1, label)

print("XG boost final cross-validation test score: {:.2f}".format(test_crossval.mean()))



test_xg = xg.predict(X_test)

confusion_test=confusion_matrix(y_test, test_xg)

print("\nconfusion matrix:\n{}".format(confusion_test))



test_proba = xg.predict_proba(X_test)



test = (test_proba [:,1] >= threshold).astype('int')



accuracy_test = accuracy_score(y_test, test)



print("\naccurcy rate with test data with {} threshold is {}".format(threshold, str(round(accuracy_test,4,)*100))+"%")



      

print("\nxg boosting classification report with adjuested threshold\n")

print(classification_report(y_test, test, target_names = ["Stay", "Churn"]))
#if we do not use a model

dummy = DummyClassifier().fit(X_train, y_train)

test_dummy = dummy.predict(X_test)



#comparsion

tn_dummy, fp_dummy, fn_dummy, tp_dummy = confusion_matrix(y_test,test_dummy).ravel()



tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test).ravel()



#assume we have 4000 customers per month 

lost_dummy = (fn_dummy/(tn_dummy+fp_dummy+tp_dummy))*4000

lost_model = (fn_test/(tn_test+fp_test+tp_test))*4000

print("Assume we have 4000 customers per month, with random guessing, we will miss "+str(math.ceil(lost_dummy))+" customers who will change their phone plans.")

print("Assume we have 4000 customers per month, with the final model, we will miss "+str(math.ceil(lost_model))+" customers who will change their phone plans.")                    
# feature importance

tree = DecisionTreeClassifier(random_state=1, max_depth = 8).fit(Xds_train, yds_train)

rf = RandomForestClassifier(random_state=1, max_depth = 6).fit(Xds_train, yds_train)

print("\nFeature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), tree.feature_importances_), feature_names3), 

             reverse=True)))

print("\nFeature Importantce ranking "+ str(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names3), 

             reverse=True)))