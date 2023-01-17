import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt

import scipy.stats as sts

%matplotlib inline

import seaborn as sns

import statistics

import warnings

warnings.filterwarnings('ignore')



#import machine learning

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris 



from sklearn.tree import DecisionTreeClassifier



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn.externals import joblib #save model



from xgboost import XGBClassifier

import xgboost



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, f1_score, recall_score

from sklearn.metrics import roc_auc_score, roc_curve





%matplotlib inline
df = pd.read_csv('../input/telecom-users/telecom_users.csv', delimiter=',')

df # View the table. Our Churn is at the end on the right
df.info()

# Almost all values are object, which means the task is categorical. In the future we will take this into account when choosing training models
df.isnull().sum()# let's check if there are any null
df['InternetService'].value_counts(dropna=False) 
# delete the null column, it is not needed

#df = df.drop(columns='Unnamed')



# the gender attribute is converted to a number

df['gender'] = df['gender'].map({'Female': 0, 

                                 'Male': 1}).astype(int)



# Using a loop for columns with 2 categories: 'yes' or 'no'

list_yes_no = ['Partner', 'Dependents', 'PhoneService', 

               'PaperlessBilling', 'Churn']

for column in list_yes_no:

    df[column] = df[column].map({'No': 0, 

                                 'Yes': 1}).astype(int)



# indication of phone lines in numbers (3 categories!)

df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 

                                               'Yes': 1, 

                                               'No phone service': 2}).astype(int)



df['InternetService'] = df['InternetService'].map({'DSL': 0, 

                                                   'Fiber optic': 1, 

                                                   'No': 2}).astype(int)





list_3_categ = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 

                'TechSupport', 'StreamingTV', 'StreamingMovies']



# loop 

for column in list_3_categ:

    df[column] = df[column].map({'No': 0, 'Yes': 1, 

                                 'No internet service': 2}).astype(int)



df['Contract'] = df['Contract'].map({'Month-to-month': 0, 

                                     'One year': 1, 

                                     'Two year': 2}).astype(int)





df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 

                                               'Mailed check': 1, 

                                               'Bank transfer (automatic)': 2, 

                                               'Credit card (automatic)': 3}).astype(int)

fig, subplot = plt.subplots()

subplot.hist(df['tenure'].values, bins=4, histtype='bar',

             align='mid', orientation='vertical');

fig, subplot = plt.subplots()

subplot.hist(df['MonthlyCharges'].values, bins=4, histtype='bar',

             align='mid', orientation='vertical');
# from the histograms around the dienes dipezona breakdown



df.loc[df['tenure'] <= 18, 'tenure'] = 0

df.loc[(df['tenure'] > 18) & (df['tenure'] <= 36), 'tenure'] = 1

df.loc[(df['tenure'] > 36) & (df['tenure'] <= 54), 'tenure'] = 2

df.loc[df['tenure'] > 54, 'tenure'] = 3



df.loc[df['MonthlyCharges'] <= 42, 'MonthlyCharges'] = 0

df.loc[(df['MonthlyCharges'] > 42) & (df['MonthlyCharges'] <= 70), 'MonthlyCharges'] = 1

df.loc[(df['MonthlyCharges'] > 70) & (df['MonthlyCharges'] <= 95), 'MonthlyCharges'] = 2

df.loc[df['MonthlyCharges'] > 95, 'MonthlyCharges'] = 3
df
# By the way, there is another way of splitting, without cycles. But in this case, it is not convenient.

#df = pd.get_dummies(df_main, columns =['gender','Partner'])
import seaborn as sns

sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

plt.title('Pearson Correlation of Features', y=1.05, size=18)

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
# to avoid redundancy of features, we will delete some features:

# just in case, we'll make a copy of the dataframe

df_main = df.copy()

drop_elements = ['PhoneService', 'StreamingMovies', 'StreamingTV', 

                 'TechSupport', 'DeviceProtection', 'OnlineBackup']

df_main = df_main.drop(drop_elements, axis=1)



# And we will also delete 3 attributes that have:

# - very weak "connections", and they can ruin our models by making noise.

# SeniorCitizen, Partner, Dependents 

drop_elements2 = ['customerID','SeniorCitizen', 'Partner', 'Dependents','TotalCharges']

df_main = df_main.drop(drop_elements2, axis=1)
X = df_main.drop(['Churn'], axis=1)

y = df_main.Churn



TEST_SIZE = 0.3 

RAND_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)
#train XGBoost model



xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)

xgb.fit(X_train,y_train.squeeze().values)

y_train_preds = xgb.predict(X_train)

y_test_preds = xgb.predict(X_test) 

print('XGBoost: {:.2f}'.format(xgb.score(X_test, y_test)))



y_test_preds_itog = xgb.predict(X) 
# KNeighborsClassifier 

# First let's see how many neighbors to take for training



neighbors = np.arange(1, 15)



train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test)



plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')

plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()

plt.xlabel('n_neighbors')

plt.ylabel('Accuracy')

plt.show()
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

print('Accuracy: {:.2f}'.format(knn.score(X_test, y_test)))

y_test_preds_knn = knn.predict(X) 
# GradientBoostingClassifier 

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

predicted_y = gbc.predict(X_test)

print('Accuracy: {:.2f}'.format(gbc.score(X_test, y_test)))

y_test_preds_gbc = gbc.predict(X) 
# LogisticRegression

classifier = LogisticRegression(solver='lbfgs',random_state=40)

classifier.fit(X_train, y_train)

predicted_y = classifier.predict(X_test)

print('Accuracy: {:.2f}'.format(classifier.score(X_test, y_test)))

y_test_preds_classifier = gbc.predict(X) 
# SVC

SVC_model = SVC()  

SVC_model.fit(X_train, y_train)

predicted_y = SVC_model.predict(X_test)

print('Accuracy: {:.2f}'.format(SVC_model.score(X_test, y_test)))     

y_test_preds_SVC_model = SVC_model.predict(X) 
# GaussianNB

clf = GaussianNB()

clf.fit(X_train, y_train)

predicted_y = clf.predict(X_test)

y_test_preds_clf = clf.predict(X) 

print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))
tree = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=40)

tree.fit(X_train, y_train)

predicted_y = tree.predict(X_test)

print('Accuracy: {:.2f}'.format(tree.score(X_test, y_test)))

y_test_preds_tree = tree.predict(X) 
print('XGBoost: {:.2f}'.format(xgb.score(X_test, y_test)))

print('KNN: {:.2f}'.format(knn.score(X_test, y_test)))

print('GradientBoos: {:.2f}'.format(gbc.score(X_test, y_test)))

print('LogisticRegression: {:.2f}'.format(classifier.score(X_test, y_test)))

print('SVC: {:.2f}'.format(SVC_model.score(X_test, y_test)))     

print('GaussianNB: {:.2f}'.format(clf.score(X_test, y_test)))

print('tree: {:.2f}'.format(tree.score(X_test, y_test)))

df3 = pd.DataFrame({'|ACTUAL|': y, 'XGBoost': y_test_preds_itog,'KNN': y_test_preds_knn, 'Gradient': y_test_preds_gbc, 'LogisticR': y_test_preds_classifier, 'SVC_model': y_test_preds_SVC_model, 'GaussianNB': y_test_preds_clf, 'DecisionTree': y_test_preds_tree})

df3.sort_index().head(20) 

#df3.sort_index().tail(60) 