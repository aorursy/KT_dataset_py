

# import statements

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling 



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

# read the data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_x = train_data.loc[:, train_data.columns != "Survived"]

train_y = train_data.loc[:, train_data.columns == "Survived"]

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_x = test_data

full_data = train_x.append(test_x)

print ("full data shape",full_data.shape)

print ("train_x shape",train_x.shape)
full_data[full_data['Age'].isnull()]
ftest_pid = test_data["PassengerId"]

ftest_pid
test_data.head()
train_data.describe()
test_data.describe()
# import warnings

# warnings.filterwarnings('ignore')

# profile = pandas_profiling.ProfileReport(train_data)

# profile
# heat map of features

plt.rcParams['figure.figsize'] = (7, 7)

plt.style.use('ggplot')



sns.heatmap(train_data.corr(), annot = True, cmap = 'Wistia')

plt.title('Heatmap for the Dataset', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15,5)

sns.distplot(train_data['Fare'], kde=False, rug=True, bins=90)

plt.title('Distribution of fare', fontsize=20)

plt.show()

plt.rcParams['figure.figsize'] = (15,5)

sns.distplot(train_data['Age'], kde=False, rug=True)

plt.title('Distribution of age', fontsize=20)

plt.show()
# box plot of fare

plt.rcParams['figure.figsize'] = (15,8)

sns.boxplot(train_data['Survived'], full_data['Fare'])

plt.title('box plot of fare with target')

plt.show()
# boxen plot of Age

plt.rcParams['figure.figsize'] = (15,8)

sns.boxplot(train_data['Survived'], full_data['Age'])

plt.title('boxen plot of age with target')

plt.show()
# boxen plot of Age

plt.rcParams['figure.figsize'] = (15,8)

sns.boxenplot(train_data['Survived'], full_data['Fare'])

plt.title('boxen plot of age with target')

plt.show()
# pair plot of different feature

tmp_df = train_data[['Survived', 'Fare', 'Age']]

sns.pairplot(tmp_df)

plt.show()


# bar graph with hue

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train_data)

# sns.catplot(x="Sex", y="Survived", kind="bar", data=train_data)

plt.show()
sns.catplot(x="Survived", y="Fare", hue="Sex", kind="swarm", data=train_data)

plt.show()
# sns.relplot(x="SibSp", y="Ticket", hue="Survived", data=train_data)

# plt.show()

full_data.iloc[891:895,:]
# simpleImpute of Age feature

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan,strategy='mean')

imp = imp.fit(train_data[["Age"]])

# full_data["Age"] = pd.DataFrame(imp.transform(full_data[["Age"]]))

# test_data["Age"] = pd.DataFrame(imp.transform(test_data[["Age"]]))

# sns.distplot(full_data["Age"], kde=False, rug=True)

# plt.show()

tmp_age = pd.DataFrame(imp.transform(full_data[["Age"]]))

tmp_age = tmp_age.rename(columns={0:"Age"})

full_data.drop(['Age'],axis=1,inplace=True)

full_data.iloc[415:420,:]
result = pd.concat([full_data, tmp_age], axis=1, join='inner')

full_data = result

full_data
# # IterativeImpute

# from sklearn.experimental import enable_iterative_imputer

# from sklearn.impute import IterativeImputer

# imp = IterativeImputer(max_iter=10, random_state=0)

# imp.fit(train_x[["Age"]])

# tmp_age1 = pd.DataFrame(imp.transform(train_data[["Age"]]))

# tmp_age2 = pd.DataFrame(imp.transform(test_data[["Age"]]))

# sns.distplot(tmp_age1, kde=False, rug=True)

# plt.show()
full_data=full_data.drop(columns=["PassengerId","Name", "Cabin", "Ticket"])

full_data.columns
# using one hot encoder from pandas.get_dummies

onehot_embark = pd.get_dummies(full_data["Embarked"], prefix="Embark")

onehot_pclass = pd.get_dummies(full_data["Pclass"], prefix="Pclass")

# full_data.drop(columns=["Embarked","Pclass"],axis=1,inplace=True)

# full_data = pd.concat([full_data, onehot_embark, onehot_pclass], axis=1,join='inner')

# full_data



result = pd.concat([full_data, onehot_embark, onehot_pclass], axis=1,join='inner')

result
full_data = result.copy()
full_data.drop(columns=["Embarked","Pclass"],axis=1,inplace=True)

full_data
# modify cat string values of Sex column to int values

full_data.loc[full_data.Sex=="male","Sex"] = 0

full_data.loc[full_data.Sex=="female", "Sex"] = 1

full_data
tmp_full_data = full_data.copy()

tmp_data = full_data.copy()
#experiment the data; To check new features such as age range, Fare range, (sibsp*parch)->family column.

#experimentation is at the end of this notebook

#Even if something happens to full_data or tmp_full_data. we can simply execute this one cell to start at this point.

exp_full_data = tmp_data.copy() 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

tmp_age = pd.DataFrame(tmp_full_data["Age"][0:891])

full_age_0 = pd.DataFrame(tmp_full_data["Age"][:])

scaler.fit(tmp_age)

full_age_1 = pd.DataFrame(scaler.transform(full_age_0))

full_age_1 = full_age_1.rename(columns={0:"Age"})

full_age_1
sns.distplot(full_age_0)

plt.show()
sns.distplot(full_age_1)

plt.show()
full_data.drop(columns=["Age"], axis=1, inplace=True)

full_data
full_data = pd.concat([full_data, full_age_1], axis=1, join="inner")

full_data
tmp_full_data = full_data.copy()
scaler = StandardScaler()

tmp_fare = pd.DataFrame(tmp_full_data[["Fare","SibSp","Parch"]][0:891])

tmp_fare

full_fare_0 = pd.DataFrame(tmp_full_data[["Fare","SibSp","Parch"]][:])

scaler.fit(tmp_fare)

full_fare_1 = pd.DataFrame(scaler.transform(full_fare_0))

full_fare_1 = full_fare_1.rename(columns={0:"Fare",1:"SibSp",2:"Parch"})

full_fare_1
full_data.drop(columns=["Fare","SibSp","Parch"], axis=1, inplace=True)

full_data = pd.concat([full_data, full_fare_1], axis=1, join="inner")

full_data
full_data_copy = full_data.copy()

full_data.describe()
from sklearn.model_selection import train_test_split

#following is the final test

x_ftest = full_data.iloc[891:,:]

print ("shape of x_ftest",x_ftest.shape)

# (418,11)

xdf = pd.DataFrame(full_data.iloc[:891,:])

ydf = train_y

x_train,x_test,y_train, y_test = train_test_split(xdf,ydf,test_size=0.25,random_state=None)

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
# Random forests is well known for classification problems.

# But it is suffering from overfitting. Submission acc is 73.2%

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import classification_report



# model = RandomForestClassifier(n_estimators=100)

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)



# # evaluating the model

# print("Training Accuracy :", model.score(x_train, y_train))

# print("Testing Accuracy :", model.score(x_test, y_test))



# # cofusion matrix

# cm = confusion_matrix(y_test, y_pred)

# plt.rcParams['figure.figsize'] = (5, 5)

# sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')



# # classification report

# cr = classification_report(y_test, y_pred)

# print(cr)
# random forest algorithm is overfitting. So, lets use logistic regression with regularization.

# final subimission accuracy of this model is 79.4%

# but lets try using SVM

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

model = LogisticRegression(random_state=0).fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred_quant = model.predict_proba(x_test)[:, 1]

# evaluating the model

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))



# cofusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')



# classification report

cr = classification_report(y_test, y_pred)

print(cr)
# # submission acc is 75.5% 

# from sklearn import svm

# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import classification_report

# model = svm.SVC(kernel='linear')

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# # evaluating the model

# print("Training Accuracy :", model.score(x_train, y_train))

# print("Testing Accuracy :", model.score(x_test, y_test))



# # cofusion matrix

# cm = confusion_matrix(y_test, y_pred)

# plt.rcParams['figure.figsize'] = (5, 5)

# sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')



# # classification report

# cr = classification_report(y_test, y_pred)

# print(cr)
# y_fpred = pd.DataFrame(model.predict(x_ftest))

# y_fpred = y_fpred.rename(columns={0:"Survived"})

# print (y_fpred)

# fresult = pd.concat([ftest_pid, y_fpred],axis=1,join='inner')

# print (fresult)

# fresult.to_csv("result_with_norm_svm.csv", index=False)
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])



plt.rcParams['figure.figsize'] = (15, 5)

plt.title('ROC curve for titanic classifier', fontweight = 30)

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
from sklearn.metrics import auc

auc = auc(fpr, tpr)

print("AUC Score :", auc)
# Learn this





# importing ML Explanability Libraries

#for purmutation importance

import eli5 

from eli5.sklearn import PermutationImportance



#for SHAP values

import shap 

from pdpbox import pdp, info_plots #for partial plots



# let's check the importance of each attributes



perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())



# plotting the partial dependence plot for num_major_vessels



base_features = full_data.columns.values.tolist()



feat_name = 'Sex'

pdp_dist = pdp.pdp_isolate(model=model, dataset=x_test, model_features = base_features, feature = feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
exp_full_data.describe()
# understand how to use this method of graph

# this is for continuous variables.

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# this is for categorical values

grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()