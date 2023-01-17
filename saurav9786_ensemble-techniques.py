# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier



# calculate accuracy measures and confusion matrix

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the dataset



bank_df = pd.read_csv("/kaggle/input/bank-marketing/bank-additional-full.csv",sep = ';')
#Shape of the data



bank_df.shape
#Reading the dataset



bank_df.head()
#Info about the dataset

bank_df.info()
#### this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model



bank_df.drop(['duration'], inplace=True, axis=1)
bank_df['pdays']=bank_df['pdays'].astype('category')

bank_df['y']=bank_df['y'].astype('category')
sns.boxplot(x=bank_df['age'], data=bank_df)
#histograms from the pair plots

sns.pairplot(bank_df)
bank_df['job'].value_counts()
sns.countplot(bank_df['marital'])
plt.figure(figsize=(12,5))

sns.countplot(bank_df['education'])
plt.figure(figsize=(12,5))

sns.countplot(bank_df['default'])
sns.countplot(bank_df['housing'])
sns.countplot(bank_df['loan'])
sns.countplot(bank_df['contact'])
sns.countplot(bank_df['poutcome'])
sns.countplot(bank_df['y'])
bank_df['y'].value_counts(normalize=True)
#Rename the dependant column from 'y ' to 'Target'

bank_df.rename(columns={'y':'Target'}, inplace=True)
bank_df.columns
#Group numerical variables by mean for the classes of Y variable

np.round(bank_df.groupby(["Target"]).mean() ,1)
### Bivariate analysis using crosstab
pd.crosstab(bank_df['job'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank_df['marital'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank_df['education'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
print(pd.crosstab(bank_df['default'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False ))

print(bank_df['default'].value_counts(normalize=True))
bank_df.drop(['default'], axis=1, inplace=True)
bank_df.columns
pd.crosstab(bank_df['housing'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank_df['loan'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank_df['contact'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
pd.crosstab(bank_df['day_of_week'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )[0:10]
pd.crosstab(bank_df['month'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )
#Binning:

def binning(col, cut_points, labels=None):

  #Define min and max values:

  minval = col.min()

  maxval = col.max()



  #create list by adding min and max to cut_points

  break_points = [minval] + cut_points + [maxval]



  #if no labels provided, use default labels 0 ... (n-1)

  if not labels:

    labels = range(len(cut_points)+1)



  #Binning using cut function of pandas

  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)

  return colBin
#Binning campaign

cut_points = [2,3,4]

labels = ["<=2","3","4",">4"]

bank_df['campaign_range'] = binning(bank_df['campaign'], cut_points, labels)

bank_df['campaign_range'].value_counts()
bank_df.drop(['campaign'], axis=1, inplace=True)

bank_df.columns
X = bank_df.drop("Target" , axis=1)

y = bank_df["Target"]   # select all rows and the 17 th column which is the classification "Yes", "No"

X = pd.get_dummies(X, drop_first=True)
test_size = 0.30 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train.shape,X_test.shape
#instantiating decision tree as the default model

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)
# Is the model an overfit model? 

y_pred = dt_model.predict(X_test)

print(dt_model.score(X_train, y_train))

print(dt_model.score(X_test , y_test))
# Note: - Decision Tree is a non-parametric algorithm and hence prone to overfitting easily. This is evident from the difference

# in scores in training and testing



# In ensemble techniques, we want multiple instances (each different from the other) and each instance to be overfit!!!  

# hopefully, the different instances will do different mistakes in classification and when we club them, their

# errors will get cancelled out giving us the benefit of lower bias and lower overall variance errors.
#Confusion matrix



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score



print(confusion_matrix(y_test, y_pred))



print(accuracy_score(y_test, y_pred))





print(recall_score(y_test, y_pred,average="binary", pos_label="yes"))
clf_pruned = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)

clf_pruned.fit(X_train, y_train)
import graphviz

from sklearn.tree import export_graphviz



data = export_graphviz(clf_pruned,out_file=None,feature_names=feature_cols,class_names=['0','1'],   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data)

graph
## Calculating feature importance

#feature_names=feature_cols

feat_importance = clf_pruned.tree_.compute_feature_importances(normalize=False)





feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))

feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')

feat_imp.sort_values(by=0, ascending=False)[0:10] #Top 10 features
preds_pruned = clf_pruned.predict(X_test)

preds_pruned_train = clf_pruned.predict(X_train)

acc_DT = accuracy_score(y_test, preds_pruned)

recall_DT = recall_score(y_test, preds_pruned, average="binary", pos_label="yes")
#Store the accuracy results for each model in a dataframe for final comparison

resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT, 'recall': recall_DT})

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf
## Apply the Random forest model and print the accuracy of Random forest Model





from sklearn.ensemble import RandomForestClassifier

rfcl = RandomForestClassifier(n_estimators = 50)

rfcl = rfcl.fit(X_train, y_train)









pred_RF = rfcl.predict(X_test)

acc_RF = accuracy_score(y_test, pred_RF)

recall_RF = recall_score(y_test, pred_RF, average="binary", pos_label="yes")
tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [acc_RF], 'recall': [recall_RF]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf
## Apply Adaboost Ensemble Algorithm for the same data and print the accuracy.





from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier( n_estimators= 200, learning_rate=0.1, random_state=22)

abcl = abcl.fit(X_train, y_train)









pred_AB =abcl.predict(X_test)

acc_AB = accuracy_score(y_test, pred_AB)

recall_AB = recall_score(y_test, pred_AB, pos_label='yes')
tempResultsDf = pd.DataFrame({'Method':['Adaboost'], 'accuracy': [acc_AB], 'recall':[recall_AB]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf
## Apply Bagging Classifier Algorithm and print the accuracy





from sklearn.ensemble import BaggingClassifier



bgcl = BaggingClassifier(n_estimators=100, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)

bgcl = bgcl.fit(X_train, y_train)











pred_BG =bgcl.predict(X_test)

acc_BG = accuracy_score(y_test, pred_BG)

recall_BG = recall_score(y_test, pred_BG, pos_label='yes')
tempResultsDf = pd.DataFrame({'Method':['Bagging'], 'accuracy': [acc_BG], 'recall':[recall_BG]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.1, random_state=22)

gbcl = gbcl.fit(X_train, y_train)









pred_GB =gbcl.predict(X_test)

acc_GB = accuracy_score(y_test, pred_GB)

recall_GB = recall_score(y_test, pred_GB, pos_label='yes')

tempResultsDf = pd.DataFrame({'Method':['Gradient Boost'], 'accuracy': [acc_GB], 'recall':[recall_GB]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf