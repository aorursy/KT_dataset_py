import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
import os

print(os.listdir("../input"))



train=pd.read_csv("../input/train.csv")
train.shape
train.head(3)
#train['rooms'].value_counts().plot(kind='bar', hue='Target')

sns.countplot(x="rooms", hue= "Target", data=train, palette="coolwarm")

plt.xlabel("No. of Rooms", fontsize=10)

plt.ylabel("No. of Household", fontsize=10)

plt.title("No. of Rooms in Households in different poverty Class", fontsize=15)

plt.show()
train['Target'].value_counts().plot(kind='pie',  autopct='%1.1f%%')

plt.show()
sns.countplot(x="r4h3", hue= "Target", data=train, palette="coolwarm")

plt.xlabel("No. of Males", fontsize=10)

plt.ylabel("No. of Household", fontsize=10)

plt.title("No. of Males in Households in different poverty Class", fontsize=15)

plt.show()
train.boxplot(column='r4h3', by='Target',patch_artist=True, )

plt.grid(True)

plt.xlabel("Class")

plt.ylabel("Males ")

plt.title("Boxplot of Males by Poverty Class ")

plt.suptitle("")

plt.show() 
sns.distplot( train["r4t3"], color= 'green',  hist= True, rug= True, bins=15).grid(True)

plt.xlabel("Total persons in the household")

#plt.ylabel("Males ")

plt.title("Household Size ")

plt.suptitle("")

plt.show() 
sns.violinplot( x=train["Target"], y=train["meaneduc"], linewidth=1)

plt.show()
#Rent paid

sns.violinplot( x=train["Target"], y=train["v2a1"], linewidth=1)

plt.show()
sns.countplot(x="refrig", hue= "Target", data=train, palette="coolwarm")

plt.xlabel("Refrigerators", fontsize=10)

plt.ylabel("No. of Household", fontsize=10)

plt.title("No. of Refrigerators in Households in different poverty Class", fontsize=15)

plt.show()
# Create correlation matrix

#Subset only to the columns where parentesco1 == 1 because 

#this is the head of household, the correct label for each household.

heads = train.loc[train['parentesco1'] == 1].copy()

corr_matrix = heads.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]
train.drop(['Id','idhogar','r4t3','tamhog','tamviv','hogar_total', 'SQBmeaned', 'SQBhogar_total',

            'SQBage','SQBescolari','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency',

            'SQBmeaned','agesq'], inplace = True, axis=1)

train.shape
#pd.DataFrame(train.isnull().sum())
train.columns[train.isnull().sum()!=0]
#Replace the na values with the mean value of each variable

train['v2a1'] = train['v2a1'].fillna((train['v2a1'].mean()))

train['v18q1'] = train['v18q1'].fillna((train['v18q1'].mean()))

train['rez_esc'] = train['rez_esc'].fillna((train['rez_esc'].mean()))

train['meaneduc'] = train['meaneduc'].fillna((train['meaneduc'].mean()))

#Check if any na

train.columns[train.isnull().sum()!=0]
train.select_dtypes('object').head()
# Converting the string to float

yes_no_map = {'no':0,'yes':1}

train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)

train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)

train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
# Testing for String 

train.select_dtypes('object').head()
# Splitting data into dependent and independent variable

# X is the independent variables matrix

X = train.drop('Target', axis = 1)



# y is the dependent variable vector

y = train.Target
# Scaling Features

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



X_ss = ss.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

X_PCA = pca.fit_transform(X_ss)
# split into train/test and resample the data

Xdt_train, Xdt_test, ydt_train, ydt_test = train_test_split(X_PCA, y, random_state=1)
Xdt_test.shape
from sklearn.ensemble import RandomForestClassifier
anotherModel1 = RandomForestClassifier(n_estimators=100, max_features=2, oob_score=True, random_state=42)

anotherModel1 = anotherModel1.fit(Xdt_train, ydt_train)
ydt_pred1 = anotherModel1.predict(Xdt_test)

ydt_pred1
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred1)

#print(con_mat_dt)

sns.heatmap(con_mat_dt,annot=True,cmap='Blues', fmt='g')

plt.title('RFM Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

#plt.grid(True)

plt.show()
print('    Accuracy Report: Random Forest Model\n', classification_report(ydt_test, ydt_pred1))
from sklearn.tree import DecisionTreeClassifier
anotherModel2 = DecisionTreeClassifier(max_depth=3, random_state=42)

anotherModel2 = anotherModel2.fit(Xdt_train, ydt_train)
ydt_pred2 = anotherModel2.predict(Xdt_test)

ydt_pred2
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred2)

sns.heatmap(con_mat_dt,annot=True,cmap='Greens', fmt='g')

plt.title('Decision Tree Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: Decision Tree Model\n', classification_report(ydt_test, ydt_pred2))
from sklearn.ensemble import GradientBoostingClassifier as gbm
anotherModel3 = gbm()

anotherModel3 = anotherModel3.fit(Xdt_train, ydt_train)
ydt_pred3 = anotherModel3.predict(Xdt_test)

ydt_pred3
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred3)

sns.heatmap(con_mat_dt,annot=True,cmap='Reds', fmt='g')

plt.title('Decision Tree Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: Gradient Boost Model\n', classification_report(ydt_test, ydt_pred3))
from sklearn.neighbors import KNeighborsClassifier
anotherModel4 = KNeighborsClassifier(n_neighbors=4)

anotherModel4 = anotherModel4.fit(Xdt_train, ydt_train)
ydt_pred4 = anotherModel4.predict(Xdt_test)

ydt_pred4
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred4)

sns.heatmap(con_mat_dt,annot=True,cmap='YlGnBu', fmt='g')

plt.title('K Neighbors Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: K Neighbors Model\n', classification_report(ydt_test, ydt_pred4))
import lightgbm as lgb
anotherModel5 = lgb.LGBMClassifier()

anotherModel5 = anotherModel5.fit(Xdt_train, ydt_train)
ydt_pred5 = anotherModel5.predict(Xdt_test)

ydt_pred5
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred5)

sns.heatmap(con_mat_dt,annot=True,cmap='BuGn_r', fmt='g')

plt.title('Light GBM Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: Light GBM Model\n', classification_report(ydt_test, ydt_pred5))
from sklearn.linear_model import LogisticRegression
anotherModel6 = LogisticRegression(C=0.1, penalty='l1')

anotherModel6 = anotherModel6.fit(Xdt_train, ydt_train)
ydt_pred6 = anotherModel6.predict(Xdt_test)

ydt_pred6
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred6)

sns.heatmap(con_mat_dt,annot=True,cmap='Purples', fmt='g')

plt.title('Logistic Regressioin with L1 Penalty Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: Logistic Regressioin with L1 Penalty Model\n', classification_report(ydt_test, ydt_pred6))
from sklearn.ensemble import ExtraTreesClassifier
anotherModel7 = ExtraTreesClassifier()

anotherModel7 = anotherModel7.fit(Xdt_train, ydt_train)
ydt_pred7 = anotherModel7.predict(Xdt_test)

ydt_pred7
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred7)

sns.heatmap(con_mat_dt,annot=True,cmap='Oranges', fmt='g')

plt.title('Logistic Extra Trees Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: Extra Trees Model\n', classification_report(ydt_test, ydt_pred7))
from xgboost.sklearn import XGBClassifier as XGB
anotherModel8 = XGB()

anotherModel8 = anotherModel8.fit(Xdt_train, ydt_train)
ydt_pred8 = anotherModel8.predict(Xdt_test)

ydt_pred8
con_mat_dt = metrics.confusion_matrix(ydt_test, ydt_pred8)

sns.heatmap(con_mat_dt,annot=True,cmap='BuGn', fmt='g')

plt.title('XGB Confusion Matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('    Accuracy Report: XGB Model\n', classification_report(ydt_test, ydt_pred8))
from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV as BayesSCV
bayes_tuner = BayesSCV(RandomForestClassifier(n_jobs = 2),



    #  Estimator parameters to be change/tune

    {

        'n_estimators': (100, 500),           

        'criterion': ['gini', 'entropy'],    

        'max_depth': (4, 100),               

        'max_features' : (10,64),             

        'min_weight_fraction_leaf' : (0,0.5, 'uniform')   

    },



    # 2.13

    n_iter=32,            

    cv = 3               

)
#bayes_cv_tuner.fit(Xdt_train, ydt_train)
test=pd.read_csv("../input/test.csv")

#test=pd.read_table("E:\\Big Data\\Costa Rica\\Data\\test.csv", engine='python', sep=',')
test.drop(['Id','idhogar','r4t3','tamhog','tamviv','hogar_total', 'SQBmeaned', 'SQBhogar_total',

            'SQBage','SQBescolari','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency',

            'SQBmeaned','agesq'], inplace = True, axis=1)

test.shape
test.columns[test.isnull().sum()!=0]
#Replace the na values with the mean value of each variable

test['v2a1'] = test['v2a1'].fillna((test['v2a1'].mean()))

test['v18q1'] = test['v18q1'].fillna((test['v18q1'].mean()))

test['rez_esc'] = test['rez_esc'].fillna((test['rez_esc'].mean()))

test['meaneduc'] = test['meaneduc'].fillna((test['meaneduc'].mean()))

#Check if any na

test.columns[test.isnull().sum()!=0]
test.select_dtypes('object').head()
# Converting the string to float

yes_no_map = {'no':0,'yes':1}

test['dependency'] = test['dependency'].replace(yes_no_map).astype(np.float32)

test['edjefe'] = test['edjefe'].replace(yes_no_map).astype(np.float32)

test['edjefa'] = test['edjefa'].replace(yes_no_map).astype(np.float32)
test_ss = ss.fit_transform(test)
test_ss.shape
pca = PCA(n_components=83)

test_PCA = pca.fit_transform(test_ss)
ydt_pred41 = anotherModel4.predict(test_PCA)

ydt_pred41
unique_elements, counts_elements = np.unique(ydt_pred41, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
#Saving as tab - seperated values

ydt_pred41.tofile('submit.csv', sep='\t')