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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (10, 8)

plt.rcParams['font.size'] = 14
feature=  ['Class','handicapped-infants', 'water-project-cost-sharing', 

                    'adoption-of-the-budget-resolution', 'physician-fee-freeze',

                    'el-salvador-aid', 'religious-groups-in-schools',

                    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',

                    'mx-missle', 'immigration', 'synfuels-corporation-cutback',

                    'education-spending', 'superfund-right-to-sue', 'crime',

                    'duty-free-exports', 'export-administration-act-south-africa']



votes= pd.read_csv("../input/house-votes-84.data.txt",na_values=['?'], names=feature)
# filling missing values with a = Did not vote



votes.fillna('a', inplace = True)



#def fillna(col):

 #   col.fillna(col.value_counts().index[0], inplace=True)

  #  return col

#votes=votes.apply(lambda col:fillna(col))
votes.head()


votes_original=votes.copy() 

votes.columns, votes.shape
# null values in attributes. 



votes.isnull().sum()
# Print data types for each variable 

print(votes.dtypes)
# assigning numerical values to categories



votes.replace(('a','n','y'), (0,-1,1), inplace=True)



votes.replace(('democrat', 'republican'), (1, 0), inplace=True)
# Print data types for each variable 

print(votes.dtypes)
votes.head()
votes.describe()
votes.shape
# Correlation Matrix



corr = votes.corr()

corr

# correlation matriix visualization 



f, ax = plt.subplots(figsize=(30, 18)) 

sns.heatmap(corr, vmax=1, square=True,annot=True, fmt=".2f")

sns.countplot(x='Class', data=votes)

plt.title('Class:Republican=0, Democrat =1')

votes['Class'].value_counts()



plt.rcParams['figure.figsize'] = (20, 18)    

plt.subplot(2, 3, 1)

sns.countplot(votes['handicapped-infants'], color = 'violet')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.countplot(votes['water-project-cost-sharing'], color = 'blue')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.countplot(votes['adoption-of-the-budget-resolution'], color = 'green')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.countplot(votes['physician-fee-freeze'], color = 'red')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.countplot(votes['el-salvador-aid'], color = 'purple')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.countplot(votes['religious-groups-in-schools'], color = 'orange')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.show()


plt.rcParams['figure.figsize'] = (20, 18)    

plt.subplot(2, 3, 1)

sns.countplot(votes['anti-satellite-test-ban'], color = 'violet')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.countplot(votes['aid-to-nicaraguan-contras'], color = 'blue')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.countplot(votes['mx-missle'], color = 'green')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.countplot(votes['immigration'], color = 'red')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.countplot(votes['synfuels-corporation-cutback'], color = 'purple')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.countplot(votes['education-spending'], color = 'orange')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.show()
plt.rcParams['figure.figsize'] = (15, 15)    

plt.subplot(2, 2, 1)

sns.countplot(votes['superfund-right-to-sue'], color = 'violet')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 2)

sns.countplot(votes['crime'], color = 'blue')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 3)

sns.countplot(votes['duty-free-exports'], color = 'green')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.subplot(2, 2, 4)

sns.countplot(votes['export-administration-act-south-africa'], color = 'red')

plt.title('No=-1,Yes=1,Undecided=0')

plt.xticks(rotation = 45)



plt.show()
print(votes.dtypes)
plt.rcParams['figure.figsize'] = (20, 15) 



#plt.subplot(8, 2, 1)

sns.catplot(x='handicapped-infants', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 2)

sns.catplot(x='water-project-cost-sharing', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 3)

sns.catplot(x='adoption-of-the-budget-resolution', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 4)

sns.catplot(x='physician-fee-freeze', col='Class', kind='count', data=votes);

plt.rcParams['figure.figsize'] = (20, 15) 

#plt.subplot(8, 2, 5)

sns.catplot(x='el-salvador-aid', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 6)

sns.catplot(x='religious-groups-in-schools', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 7)

sns.catplot(x='anti-satellite-test-ban', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 8)

sns.catplot(x='aid-to-nicaraguan-contras', col='Class', kind='count', data=votes);

plt.rcParams['figure.figsize'] = (20, 15) 



#plt.subplot(8, 2, 9)

sns.catplot(x='mx-missle', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 10)

sns.catplot(x='immigration', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 11)

sns.catplot(x='synfuels-corporation-cutback', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 12)

sns.catplot(x='education-spending', col='Class', kind='count', data=votes);



plt.rcParams['figure.figsize'] = (20, 15) 



#plt.subplot(8, 2, 13)

sns.catplot(x='crime', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 14)

sns.catplot(x='superfund-right-to-sue', col='Class', kind='count', data=votes);





#plt.subplot(8, 2, 15)

sns.catplot(x='duty-free-exports', col='Class', kind='count', data=votes);



#plt.subplot(8, 2, 16)

sns.catplot(x='export-administration-act-south-africa', col='Class', kind='count', data=votes);

# dropping the attributes to avoid multicollinearty 



votes_drop = votes.drop(['physician-fee-freeze','el-salvador-aid','education-spending','aid-to-nicaraguan-contras','adoption-of-the-budget-resolution'], axis=1)



votes_drop.shape
votes_drop.head()

#x=votes.iloc[:, :1]

#y=votes.iloc[:, 1:12]



X = votes_drop.drop(['Class'], axis=1)

y = votes_drop["Class"]



# Stratified sampling



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101,stratify=y)
# Importing the packages for Decision Tree Classifier



from sklearn import tree

tree_one = tree.DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=101, min_samples_leaf=3, class_weight="balanced")  #, class_weight="balanced"

tree_one
# Fitting the decision tree model on your features and label



tree_one = tree_one.fit(X_train, y_train)
# The feature_importances_ attribute make it simple to interpret the significance of the predictors you include



list(zip(X_train.columns,tree_one.feature_importances_))
# The accuracy of the model on Train data



print(tree_one.score(X_train, y_train))





# The accuracy of the model on Test data



print(tree_one.score(X_test, y_test))
# Visualize the decision tree graph



with open('tree.dot','w') as dotfile:

    tree.export_graphviz(tree_one, out_file=dotfile, feature_names=X_train.columns, filled=True)

    dotfile.close()

    

    

from graphviz import Source



with open('tree.dot','r') as f:

    text=f.read()

    plot=Source(text)

plot
y_pred = tree_one.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report



df_confusion = confusion_matrix(y_test, y_pred)

df_confusion
plt.rcParams['figure.figsize'] = (10, 6) 

cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion,cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,

            fmt='d')


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# Setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two



tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 101, class_weight='balanced')

tree_two = tree_two.fit(X_train, y_train)



#Print the score of both the decision tree



print("New Decision Tree Accuracy: ",tree_two.score(X_train, y_train))

print("Original Decision Tree Accuracy",tree_one.score(X_train,y_train))
# The accuracy of the model on Train data



print(tree_two.score(X_train, y_train))





# The accuracy of the model on Test data



print(tree_two.score(X_test, y_test))
# Building confusion matrix of our improved model

predict = tree_two.predict(X_test)

df_confusion_new = confusion_matrix(y_test, predict)

df_confusion_new
cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_new, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,

            fmt='d')


from sklearn.metrics import classification_report

print(classification_report(y_test, predict))
# Different parameters we want to test



max_depth = [5,10,15] 

criterion = ['gini', 'entropy']

min_samples_split = [5,10,15]
# Importing GridSearch



from sklearn.model_selection import GridSearchCV
# Building the model



tree_three = tree.DecisionTreeClassifier(class_weight="balanced")



# Cross-validation tells how well a model performs on a dataset using multiple samples of train data

grid = GridSearchCV(estimator = tree_three, cv=3, 

                    param_grid = dict(max_depth = max_depth, criterion = criterion, min_samples_split=min_samples_split), verbose=2)
grid.fit(X_train,y_train)
# Best accuracy score



print('Avg accuracy score across 54 models:', grid.best_score_)
# Best parameters for the model



grid.best_params_
# Building the model based on new parameters



tree_three = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 10, random_state=42, min_samples_split=5, class_weight="balanced")
tree_three.fit(X_train,y_train)
# Accuracy Score for new model



print ("DT_three accuracy Train:",tree_three.score(X_train,y_train))



# The accuracy of the model on Test data



print("DT_three accuracy Test:",tree_two.score(X_test, y_test))
# Building confusion matrix of our improved model

pred_three = tree_three.predict(X_test)

df_confusion_three = confusion_matrix(y_test, pred_three)

df_confusion_three
cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_three, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,

            fmt='d')


from sklearn.metrics import classification_report

print(classification_report(y_test, pred_three))
test = pd.read_csv("../input/testing.csv")
submission = pd.DataFrame({'id':test['id'],'predicted':pred_three})


submission.to_csv("submission_DTgrid.csv", index=False)

submission.head()
# Building and fitting Random Forest



from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(criterion = 'gini',  n_estimators = 100, max_depth = 10,random_state = 101, class_weight="balanced")
rf_forest = forest.fit(X_train, y_train)
# Print the accuracy score of the fitted random forest



print("RF Accuracy Train:", rf_forest.score(X_train, y_train))

print("RF Accuracy Test:", rf_forest.score(X_test, y_test))
# Making predictions



pred_rf = rf_forest.predict(X_test)
list(zip(X_train.columns,rf_forest.feature_importances_))
df_confusion_rf = confusion_matrix(y_test, pred_rf)

df_confusion_rf
cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_rf, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,

            fmt='d')


from sklearn.metrics import classification_report

print(classification_report(y_test, pred_rf))
submission = pd.DataFrame({'id':test['id'],'predicted':pred_rf})


submission.to_csv("submission_RForest.csv", index=False)

submission.head()
from sklearn import naive_bayes



clf = naive_bayes.GaussianNB()

model=clf.fit(X_train, y_train)
# Print the accuracy score of the fitted random forest



print("NB Accuracy Train:", model.score(X_train, y_train))

print("NG Accuracy Test:", model.score(X_test, y_test))
pred_NB=model.predict(X_test)

print(pred_NB)
df_confusion_NB = confusion_matrix(y_test, pred_NB)

df_confusion_NB
cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_NB, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,

            fmt='d')


from sklearn.metrics import classification_report

print(classification_report(y_test, pred_NB))
submission = pd.DataFrame({'id':test['id'],'predicted':pred_NB})


submission.to_csv("submission_NB.csv", index=False)

submission.head()