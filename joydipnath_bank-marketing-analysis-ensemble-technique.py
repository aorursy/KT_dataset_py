# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer  #DT does not take strings as input for the model fit step....
df = pd.read_csv("../input/bank-full.csv")
df.head(15)
df.shape
df.describe()
df.info()  # many columns are of type object i.e. strings. These need to be converted to ordinal type
for feature in df.columns: # Loop through all columns in the dataframe

    if df[feature].dtype == 'object': # Only apply for columns with categorical strings

        df[feature] = pd.Categorical(df[feature])# Replace strings with an integer

df.head(10)
df.info()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),

            annot=True,

            linewidths=.5,

            center=0,

            cbar=False,

            cmap="YlGnBu")

plt.show()
sns.pairplot(df,diag_kind='kde')
# df_box = df['age', 	'balance', 	'day', 	'duration', 	'campaign', 	'pdays', 	'previous']



sns.boxplot(x=df['age'], y=df['Target'], data=pd.melt(df))
sns.boxplot(x=df['balance'], y=df['Target'], data=pd.melt(df))
sns.boxplot(x=df['campaign'], y=df['Target'], data=pd.melt(df))
sns.boxplot(x=df['duration'], y=df['Target'], data=pd.melt(df))
# Excluding Outcome column which has only 

df.drop(['Target'], axis=1).hist(stacked=False, bins=100, figsize=(30,45), layout=(14,4))
# from sklearn import preprocessing

# from sklearn.preprocessing import StandardScaler



# scaler = StandardScaler()

# print(scaler.fit(df))



# mm_scaler = preprocessing.MinMaxScaler()

# X_train = mm_scaler.fit_transform(X_train)

# mm_scaler.transform(X_test)



oneHotCols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]

df_data = pd.get_dummies(df, columns=oneHotCols)

df_data.head(10)
df_data.info()
# splitting data into training and test set for independent attributes

from sklearn.model_selection import train_test_split



X = df_data.drop("Target" , axis=1)

y = df_data.pop("Target")



from scipy.stats import zscore

# convert the features into z scores as we do not know what units / scales were used and store them in new dataframe

# It is always adviced to scale numeric attributes in models that calculate distances.



XScaled  = X.apply(zscore)  # convert all attributes to Z scale 



# XScaled.describe()



X_train, X_test, y_train, y_test = train_test_split(XScaled, y, test_size=.30, random_state=1)

X_train.shape,X_test.shape
# invoking the decision tree classifier function. Using 'entropy' method of finding the split columns.



model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X_train, y_train)
model_entropy.score(X_train, y_train)  # performance on train data
model_entropy.score(X_test, y_test)  # performance on test data
clf_pruned = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)

clf_pruned.fit(X_train, y_train)
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import pydotplus

import graphviz



xvar = df.drop('Target', axis=1)

feature_cols = xvar.columns

feature_cols
from sklearn import tree

from os import system



train_char_label = ['No', 'Yes']

Credit_Tree_FileR = open('credit_treeR.dot','w')

dot_data = tree.export_graphviz(model_entropy, out_file=Credit_Tree_FileR, feature_names = list(X_train), class_names = list(train_char_label))

Credit_Tree_FileR.close()



#Works only if "dot" command works on you machine



retCode = system("dot -Tpng credit_treeR.dot -o credit_treeR.png")

if(retCode>0):

    print("system command returning error: "+str(retCode))

else:

    display(Image("credit_treeR.png"))
print (pd.DataFrame(model_entropy.feature_importances_, columns = ["Imp"], index = X_train.columns))
# X_train, X_test, y_train, y_test

preds_pruned_test = clf_pruned.predict(X_test)

preds_pruned_train = clf_pruned.predict(X_train)



print(accuracy_score(y_test,preds_pruned_test))

print(accuracy_score(y_train,preds_pruned_train))
#Predict for pruned train set

mat_train = confusion_matrix(y_train, preds_pruned_train)

print("For pruned train confusion matrix = \n",mat_train)
mat_test = confusion_matrix(y_test, preds_pruned_test)



print("For pruned test confusion matrix = \n",mat_test)
#Store the accuracy results for each model in a dataframe for final comparison

acc_DT = accuracy_score(y_test, preds_pruned_test)

resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT})

resultsDf = resultsDf[['Method', 'accuracy']]

resultsDf
dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)

dTree.fit(X_train, y_train)
print(dTree.score(X_train, y_train))

print(dTree.score(X_test, y_test))
train_char_label = ['No', 'Yes']

Credit_Tree_File = open('credit_tree_ginni.dot','w')

dot_data = tree.export_graphviz(dTree, out_file=Credit_Tree_File, feature_names = list(X_train), class_names = list(train_char_label))

Credit_Tree_File.close()


retCode = system("dot -Tpng credit_tree_ginni.dot -o credit_tree_ginni.png")

if(retCode>0):

    print("system command returning error: "+str(retCode))

else:

    display(Image("credit_tree_ginni.png"))
dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)

dTreeR.fit(X_train, y_train)

print(dTreeR.score(X_train, y_train))

print(dTreeR.score(X_test, y_test))
train_char_label = ['No', 'Yes']

Credit_Tree_FileR = open('credit_treeR.dot','w')

dot_data = tree.export_graphviz(dTreeR, out_file=Credit_Tree_FileR, feature_names = list(X_train), class_names = list(train_char_label))

Credit_Tree_FileR.close()



#Works only if "dot" command works on you machine



retCode = system("dot -Tpng credit_treeR.dot -o credit_treeR.png")

if(retCode>0):

    print("system command returning error: "+str(retCode))

else:

    display(Image("credit_treeR.png"))
# importance of features in the tree building ( The importance of a feature is computed as the 

#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )



print (pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = X_train.columns))
print(dTreeR.score(X_test , y_test))

y_predict_DTginni = dTreeR.predict(X_test)



cm_DTginni = metrics.confusion_matrix(y_test, y_predict_DTginni, )



print("DT TFor ginni confusion matrix = \n",cm_DTginni)
from sklearn.ensemble import BaggingClassifier



bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)

#bgcl = BaggingClassifier(n_estimators=50,random_state=1)



bgcl = bgcl.fit(X_train, y_train)
y_predict = bgcl.predict(X_test)



print(bgcl.score(X_test , y_test))
cm_bagging = confusion_matrix(y_test, y_predict)



print("For Bagging confusion matrix = \n",cm_bagging)
from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier(n_estimators=50, random_state=1)

abcl = abcl.fit(X_train, y_train)
y_predict_adaboost = abcl.predict(X_test)

print(abcl.score(X_test , y_test))
cm_adaboost = metrics.confusion_matrix(y_test, y_predict_adaboost)



print("For adaboost confusion matrix = \n",cm_adaboost)
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 50,random_state=1)

gbcl = gbcl.fit(X_train, y_train)
y_predict_gradientboost = gbcl.predict(X_test)

print(gbcl.score(X_test, y_test))

cm_gradientboost = metrics.confusion_matrix(y_test, y_predict_gradientboost)

print("For gradient boost confusion matrix = \n",cm_gradientboost)
from sklearn.ensemble import RandomForestClassifier

rfcl = RandomForestClassifier(n_estimators = 50, random_state=1,max_features=12)

rfcl = rfcl.fit(X_train, y_train)
y_predict_randomforest = rfcl.predict(X_test)

print(rfcl.score(X_test, y_test))

cm_randomforest = metrics.confusion_matrix(y_test, y_predict_randomforest)

print("For gradient boost confusion matrix = \n",cm_randomforest)