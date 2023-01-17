# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # using for plot interacting plots

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#The data sets are read
train_titanic=pd.read_csv('/kaggle/input/titanic/train.csv')
test_titanic=pd.read_csv('/kaggle/input/titanic/test.csv')
#The gender_submission data set is an expample how the final result has to look like. Is is needed neither to create the model nor to test it.
gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train_titanic.shape
train_titanic.info()
train_titanic.describe()
print('Oldest Passenger was of:',train_titanic['Age'].max(),'Years')
print('Youngest Passenger was of:',train_titanic['Age'].min(),'Years')
print('Average Age on the ship:',train_titanic['Age'].mean(),'Years')
import pandas_profiling
pandas_profiling.ProfileReport(train_titanic)
#Seeing how the different features are distributed
import matplotlib.pyplot as plt
train_titanic.hist(figsize=(15,8))
plt.figure()
#Seeing the people who survived VS deaths
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
col = "Survived"
grouped = train_titanic[col].value_counts().reset_index()
grouped=grouped.rename(columns={'index':'Survived', 'Survived':'count'})
grouped
#plot
trace = go.Pie(labels=grouped['Survived'], values=grouped['count'], pull=[0.05, 0.05])
trace
title = {'title': 'Survived(0 = No, 1 = Yes)'}
fig = go.Figure(data = [trace],layout =title)
iplot(fig)
col = "Sex"
grouped = train_titanic[col].value_counts().reset_index()
grouped=grouped.rename(columns={'index':'Sex', 'Sex':'count'})
#plot
trace = go.Pie(labels=grouped['Sex'], values=grouped['count'], pull=[0.05, 0.05])
trace
title = {'title': 'Sex(male, female)'}
fig = go.Figure(data = [trace],layout =title)
iplot(fig)

grouped = train_titanic['Sex'].value_counts().reset_index()
grouped=grouped.rename(columns={'index':'Sex', 'Sex':'count'})
grouped['%']=(grouped['count']/sum(grouped['count'])*100).round(2)
print('O', grouped['%'][0],'% de todos os pasaxeiros son homes e o ', grouped['%'][1], '% son mulleres')
count_sex_survived=train_titanic.groupby(by=['Sex','Survived']).count().reset_index()
female=[]
male=[]
n=0
for gender in count_sex_survived['Sex']:
    if gender=='female':
        female.append(count_sex_survived['PassengerId'][n])
    else:
        male.append(count_sex_survived['PassengerId'][n])
    n=n+1
survived_male=pd.Series({'Survived':male[1],
                   'Dead':male[0]})
survived_female=pd.Series({'Survived':female[1],
                          'Dead':female[0]})
df_bar_plot=pd.DataFrame([survived_male, survived_female], index=['male', 'female'])
df_bar_plot.plot.bar()
passenger_class=train_titanic.groupby(by='Pclass').count()['PassengerId']
passenger_class.plot.bar()
surviving_class=train_titanic.groupby(by=['Pclass', 'Survived']).count()['PassengerId'].reset_index()
surviving_class=surviving_class.set_index('Pclass')
lista_1=list(surviving_class['PassengerId'][1])
lista_2=list(surviving_class['PassengerId'][2])
lista_3=list(surviving_class['PassengerId'][3])

surviving_class_plot=pd.DataFrame([lista_1, lista_2, lista_3], index=['class_1', 'class_2', 'class_3'])
surviving_class_plot.plot.bar()
embarked_count=train_titanic.groupby(by='Embarked').count()['PassengerId']
embarked_count.plot.bar()
surviving_class=train_titanic.groupby(by=['Embarked', 'Survived']).count()['PassengerId'].reset_index()
surviving_class=surviving_class.set_index('Embarked')
lista_1=list(surviving_class['PassengerId']['C'])
lista_2=list(surviving_class['PassengerId']['Q'])
lista_3=list(surviving_class['PassengerId']['S'])

surviving_class_plot=pd.DataFrame([lista_1, lista_2, lista_3], index=['C', 'Q', 'S'])
surviving_class_plot.plot.bar()
people_age=train_titanic.groupby(by='Age').count()['PassengerId'].reset_index()
y=people_age['PassengerId']
plt.plot(y)
age_survivors_vs_deads=train_titanic.dropna(how='any',subset=['Age'])
age_survivors_vs_deads=age_survivors_vs_deads.groupby(by=['Age', 'Survived']).count()['PassengerId'].reset_index()
age_survivors_vs_deads=age_survivors_vs_deads.rename(columns={'PassengerId':'count'})

age_survivors=age_survivors_vs_deads[age_survivors_vs_deads['Survived']==1]
x_survivors=age_survivors['Age']
y_survivors=age_survivors['count']

age_deads=age_survivors_vs_deads[age_survivors_vs_deads['Survived']==0]
x_deads=age_deads['Age']
y_deads=age_deads['count']
plt.plot(x_survivors, y_survivors, label='survivors')
plt.plot(x_deads, y_deads, label='dead')
plt.legend()
plt.show()
fare_ticket=train_titanic.groupby(by='Fare').count()
fare_ticket=fare_ticket.rename(columns={'PassengerId':'count'})
plt.plot(fare_ticket['count'])
fare_ticket=train_titanic.dropna(how='any', subset=['Fare'])
fare_ticket=fare_ticket.groupby(by=['Fare', 'Survived']). count().reset_index()
fare_ticket=fare_ticket.rename(columns={'PassengerId':'count'})

fare_ticket_survivors=fare_ticket[fare_ticket['Survived']==1]
x_survivor=fare_ticket_survivors['Fare']
y_survivor=fare_ticket_survivors['count']

fare_ticket_deads=fare_ticket[fare_ticket['Survived']==0]
x_dead=fare_ticket_deads['Fare']
y_dead=fare_ticket_deads['count']

plt.plot(x_survivor, y_survivor, label='survivors')
plt.plot(x_dead, y_dead, label='deads')
plt.legend()
plt.show()
plt.hist(train_titanic['Age'], bins=15)
train_titanic.corr()
fig,ax = plt.subplots(figsize=(8,7))
ax = sns.heatmap(train_titanic.corr(), annot=True, linewidths=.5, fmt='.1f')
plt.show()
train_titanic.groupby(by='Pclass')['Survived'].mean()
train_titanic.groupby(by='Sex')['Survived'].mean()

#that's pretting amazing correlation. females are 4 times more likely to survive
hist_survivors_class=train_titanic
#Importante os paréntesis cando usas o operadoes ('&' e '|')
hist_survivors_class_1_0=hist_survivors_class['Age'][(hist_survivors_class['Pclass']==1) & (hist_survivors_class['Survived']==0)]
hist_survivors_class_1_1=hist_survivors_class['Age'][(hist_survivors_class['Pclass']==1) & (hist_survivors_class['Survived']==1)]
hist_survivors_class_2_0=hist_survivors_class['Age'][(hist_survivors_class['Pclass']==2) & (hist_survivors_class['Survived']==0)]
hist_survivors_class_2_1=hist_survivors_class['Age'][(hist_survivors_class['Pclass']==2) & (hist_survivors_class['Survived']==1)]
hist_survivors_class_3_0=hist_survivors_class['Age'][(hist_survivors_class['Pclass']==3) & (hist_survivors_class['Survived']==0)]
hist_survivors_class_3_1=hist_survivors_class['Age'][(hist_survivors_class['Pclass']==3) & (hist_survivors_class['Survived']==1)]

fig = plt.figure()
fig.suptitle('Histogram Class|Survived|Age')
fig.set_figheight(10)
fig.set_figwidth(15)

plt.subplot(2, 3, 1)
plt.title("Class 1|Dead")
plt.hist(hist_survivors_class_1_0)

plt.subplot(2, 3, 2)
plt.title("Class 1|Survive")
plt.hist(hist_survivors_class_1_1)

plt.subplot(2, 3, 3)
plt.title("Class 2|Dead")
plt.hist(hist_survivors_class_2_0)

plt.subplot(2, 3, 4)
plt.title("Class 2|Survive")
plt.hist(hist_survivors_class_2_1)

plt.subplot(2, 3, 5)
plt.title("Class 3|Dead")
plt.hist(hist_survivors_class_3_0)

plt.subplot(2, 3, 6)
plt.title("Class 3|Survive")
plt.hist(hist_survivors_class_3_1)

plt.legend()
plt.show()

boxplot_=train_titanic
boxplot_survived_male=boxplot_['Age'][(boxplot_['Survived']==1)&(boxplot_['Sex']=='male')].dropna()
boxplot_survived_female=boxplot_['Age'][(boxplot_['Survived']==1)&(boxplot_['Sex']=='female')].dropna()
boxplot_dead_male=boxplot_['Age'][(boxplot_['Survived']==0)&(boxplot_['Sex']=='male')].dropna()
boxplot_dead_female=boxplot_['Age'][(boxplot_['Survived']==0)&(boxplot_['Sex']=='female')].dropna()

l=pd.DataFrame({'survived_male': boxplot_survived_male,
                'survived_female': boxplot_survived_female,
               'dead_male':boxplot_dead_male,
               'dead_female':boxplot_dead_female})

l.boxplot(column=['survived_male', 'survived_female','dead_male','dead_female'])
hist_survivors_class=boxplot_
boxplot_survived_male=boxplot_['Age'][(boxplot_['Survived']==1)&(boxplot_['Sex']=='male')].dropna()
boxplot_survived_female=boxplot_['Age'][(boxplot_['Survived']==1)&(boxplot_['Sex']=='female')].dropna()
boxplot_dead_male=boxplot_['Age'][(boxplot_['Survived']==0)&(boxplot_['Sex']=='male')].dropna()
boxplot_dead_female=boxplot_['Age'][(boxplot_['Survived']==0)&(boxplot_['Sex']=='female')].dropna()

box_plot_data=[boxplot_survived_male,boxplot_survived_female,boxplot_dead_male,boxplot_dead_female]
plt.boxplot(box_plot_data,patch_artist=True,labels=['survived_male','survived_female','dead_male','dead_female'])
plt.show()
 
plt.show()
train_titanic
train_titanic.drop(['Name','SibSp','Ticket','Cabin'],axis=1,inplace=True)
#Setting as the index column the 'PassengerId feature. This is necessary to submit the result into kaggle'
train_titanic=train_titanic.set_index(['PassengerId'])
train_titanic_dummy=pd.get_dummies(train_titanic)
train_titanic_dummy
missing_values=train_titanic_dummy.isnull().sum()
missing_values_percentage=train_titanic_dummy.isnull().sum()/train_titanic_dummy.isnull().count()
table_missing_values=pd.concat([missing_values, missing_values_percentage], axis=1, keys=['total', 'percentage']).sort_values(by='total',ascending=False)
table_missing_values
#Onde hai máis missing values é en 'Cabin', despois Age e por último Embarked.
#Para aqueles valores onde non hai datos, vams a meter o valor medio da columna de Age
train_titanic_dummy['Age']=train_titanic_dummy['Age'].fillna(train_titanic_dummy['Age'].mean())
#Para o cabin e o age, deixoos igual que están
#Volvemos a comprobar como queda o tema dos missing values
missing_values=train_titanic_dummy.isnull().sum()
missing_values_percentage=train_titanic_dummy.isnull().sum()/train_titanic_dummy.isnull().count()
table_missing_values=pd.concat([missing_values, missing_values_percentage], axis=1, keys=['total', 'percentage']).sort_values(by='total',ascending=False)
table_missing_values
from sklearn.model_selection import train_test_split #training and testing data split
train,test=train_test_split(train_titanic_dummy,test_size=0.3,random_state=0, stratify=train_titanic_dummy['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]

#Explicació do de stratify
#This stratify parameter makes a split so that the proportion of values in the sample produced
#will be the same as the proportion of values provided to parameter stratify.
#For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% 
#of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% 
#of 1's.

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logisticRegr = LogisticRegression()
logisticRegr.fit(train_X, train_Y)
#Eiqui predicimos os valores de Y
logisticRegr_pred=logisticRegr.predict(test_X)
#Eiqui enfrentamos os valores de Y predecidos, cos valores de Y reales (test_Y)
logisticRegr_accuracy=accuracy_score(test_Y, logisticRegr_pred)
print('The accuracy of the Logistic Regression is', logisticRegr_accuracy)
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_logistic=confusion_matrix(test_Y,logisticRegr_pred)

confusion_matrix_logistic_table=plot_confusion_matrix(confusion_matrix_logistic,
                     colorbar=True,
                    show_absolute=True,
                    show_normed=True,
                    class_names=['Survived', 'Not Survived'])
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#fpr: False positive rate
#tpr: True positive rate
auc_logistic=roc_auc_score(test_Y, logisticRegr_pred)
fpr, tpr, thresholds= roc_curve(test_Y, logisticRegr_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('ROC curve. Area Under the Curve: %0.3f' %auc_logistic)
plt.show()
#KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(train_X, train_Y)
#Eiqui predicimos os valores de Y
knn_pred=knn.predict(test_X)
#Eiqui enfrentamos os valores de Y predecidos, cos valores de Y reales (test_Y)
knn_accuracy=accuracy_score(test_Y,knn_pred)
print('The accuracy of the K nearest neighbors is', knn_accuracy)
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_knn= confusion_matrix(test_Y, knn_pred)

confusion_matrix_knn_table= plot_confusion_matrix(confusion_matrix_knn,
                                            show_absolute= True,
                                            show_normed= True,
                                            colorbar= True,
                                           class_names=['Survived', 'Not Survived'])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

auc_knn= roc_auc_score(test_Y, knn_pred)
fpr, tpr, thresholds= roc_curve(test_Y, knn_pred)
plt.plot(fpr,tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve_ Area under the curve %0.3f' %auc_knn)
plt.show()
from sklearn.tree import DecisionTreeClassifier
tree_clasiffier=DecisionTreeClassifier(max_depth=3)
tree_clasiffier.fit(train_X, train_Y)
#Creamos os y predictivos
tree_pred=tree_clasiffier.predict(test_X)
tree_accu=accuracy_score(test_Y, tree_pred)
print('The accuracy of the tree Classifier is', tree_accu)
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_tree=confusion_matrix(test_Y, tree_pred)

confusion_matrix_tree_table=plot_confusion_matrix(confusion_matrix_tree,
                     colorbar=True,
                    show_absolute=True,
                    show_normed=True,
                    class_names=['Survived', 'Not Survived'])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#fpr: False positive rate
#tpr: True positive rate
auc_tree=roc_auc_score(test_Y, tree_pred)
fpr, tpr, thresholds= roc_curve(test_Y, tree_pred )
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('ROC curve. Area Under the Curve: %0.3f' %auc_tree)
plt.show()
import graphviz
from sklearn import tree

dot_data = tree.export_graphviz(tree_clasiffier, 
                                out_file=None,
                                filled=True, 
                                rounded=True,  
                                special_characters=True,
                               feature_names=train_X.columns.values,
                               class_names=['Survived','Not Survived']) 
graph = graphviz.Source(dot_data)
graph
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

NaiveBayes = GaussianNB()
NaiveBayes.fit(train_X, train_Y)
#Eiqui predicimos os valores de Y
NaiveBayes_pred=NaiveBayes.predict(test_X)
#Eiqui enfrentamos os valores de Y predecidos, cos valores de Y reales (test_Y)
NaiveBayes_accuracy=accuracy_score(test_Y, NaiveBayes_pred)
print('The accuracy of the Naice Bayes model is', NaiveBayes_accuracy)
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_NaiveBayes= confusion_matrix(test_Y, NaiveBayes_pred)
confusion_matrix_NaiveBayes_table= plot_confusion_matrix(confusion_matrix_NaiveBayes,
                                                   colorbar=True,
                                                   show_absolute= True,
                                                   show_normed= True,
                                                  class_names=['Survived', 'Not Survived'])
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#fpr: False positive rate
#tpr: True positive rate
auc_NaiveBayes=roc_auc_score(test_Y, NaiveBayes_pred)
fpr, tpr, thresholds= roc_curve(test_Y, NaiveBayes_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('ROC curve. Area Under the Curve: %0.3f' %auc_NaiveBayes)
plt.show()
from sklearn.svm import SVC

SupportVectorClassifier= SVC(kernel='linear', degree=1, gamma='auto')
SupportVectorClassifier.fit(train_X, train_Y)
SupportVectorClassifier_pred= SupportVectorClassifier.predict(test_X)
SupportVectorClassifier_accu = accuracy_score(test_Y, SupportVectorClassifier_pred)
print(SupportVectorClassifier)
print('The accuracy of the Support Vector Classifier is', SupportVectorClassifier_accu)
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_SVC= confusion_matrix(test_Y, SupportVectorClassifier_pred)
confusion_matrix_SVC_table= plot_confusion_matrix(confusion_matrix_SVC,
                                                   colorbar=True,
                                                   show_absolute= True,
                                                   show_normed= True,
                                                  class_names=['Survived', 'Not Survived'])
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#fpr: False positive rate
#tpr: True positive rate
auc_SVC=roc_auc_score(test_Y, SupportVectorClassifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, SupportVectorClassifier_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('ROC curve. Area Under the Curve: %0.3f' %auc_SVC)
plt.show()
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier_clf = RandomForestClassifier(max_depth=5, random_state=0)
RandomForestClassifier_clf.fit(train_X, train_Y)
RandomForestClassifier_pred=RandomForestClassifier_clf.predict(test_X)
RandomForestClassifier_accu= accuracy_score(test_Y, RandomForestClassifier_pred)
print(RandomForestClassifier_clf)
print('The accuracy of the Random Forest Classifier is', RandomForestClassifier_accu)

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_RandomForest= confusion_matrix(test_Y, RandomForestClassifier_pred)
confusion_matrix_RandomForest_table= plot_confusion_matrix(confusion_matrix_RandomForest,
                                                   colorbar=True,
                                                   show_absolute= True,
                                                   show_normed= True,
                                                  class_names=['Survived', 'Not Survived'])
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#fpr: False positive rate
#tpr: True positive rate
auc_RandomForestClassifier=roc_auc_score(test_Y, RandomForestClassifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, RandomForestClassifier_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('ROC curve. Area Under the Curve: %0.3f' %auc_RandomForestClassifier)
plt.show()
from sklearn.neural_network import MLPClassifier

MLP_Classifier = MLPClassifier(activation='logistic')
MLP_Classifier.fit(train_X, train_Y)
MLP_Classifier_pred=MLP_Classifier.predict(test_X)
MLP_Classifier_accu= accuracy_score(test_Y, MLP_Classifier_pred)
print(MLP_Classifier)
print('The accuracy of the Random Forest Classifier is', MLP_Classifier_accu)
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_MLPClassifier= confusion_matrix(test_Y, MLP_Classifier_pred)
confusion_matrix_MLPClassifier_table= plot_confusion_matrix(confusion_matrix_MLPClassifier,
                                                   colorbar=True,
                                                   show_absolute= True,
                                                   show_normed= True,
                                                  class_names=['Survived', 'Not Survived'])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#fpr: False positive rate
#tpr: True positive rate
auc_MLP=roc_auc_score(test_Y, MLP_Classifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, MLP_Classifier_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('ROC curve. Area Under the Curve: %0.3f' %auc_MLP)
plt.show()
print('The accuracy of the Logistic Regression is', logisticRegr_accuracy)
print('The accuracy of the K nearest neighbors is', knn_accuracy)
print('The accuracy of the NaiveBayes is', NaiveBayes_accuracy)
print('The accuracy of the Tree Classifier is', tree_accu)
print('The accuracy of the Random Forest Classifier is', RandomForestClassifier_accu)
print('The accuracy of the SVC classifier is', SupportVectorClassifier_accu)
print('The accuracy of the MLP classifier is', MLP_Classifier_accu)
model_accuracy= pd.Series([logisticRegr_accuracy, knn_accuracy, NaiveBayes_accuracy, 
                          tree_accu, RandomForestClassifier_accu, SupportVectorClassifier_accu, 
                          MLP_Classifier_accu], index=['Logistic regression', 'KNN', 'Naive Bayes',
                                                      'Tree', 'Random Forest', 'SVC', 'MLP'])
model_accuracy.sort_values().plot.barh(color=['pink', 'red', 'green', 'blue', 'cyan', 'yellow', 'purple'])
plt.title('Model accuracies')
#https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

fig = plt.figure()
fig.suptitle('Confusion Matrix')
fig.set_figheight(20)
fig.set_figwidth(20)

plt.subplot(3, 3, 1)
plt.title("Logistic Regression")
sns.heatmap(confusion_matrix_logistic,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(3, 3, 2)
plt.title("KNN")
sns.heatmap(confusion_matrix_knn,annot=True,cmap="Reds",fmt="d",cbar=False)

plt.subplot(3, 3, 3)
plt.title("Naive Bayes")
sns.heatmap(confusion_matrix_NaiveBayes,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(3, 3, 4)
plt.title("Tree")
sns.heatmap(confusion_matrix_tree,annot=True,cmap="Reds",fmt="d",cbar=False)

plt.subplot(3, 3, 5)
plt.title("random forest")
sns.heatmap(confusion_matrix_RandomForest,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(3, 3, 6)
plt.title("random forest")
sns.heatmap(confusion_matrix_SVC,annot=True,cmap="Reds",fmt="d",cbar=False)

plt.subplot(3, 3, 7)
plt.title("random forest")
sns.heatmap(confusion_matrix_MLPClassifier,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()
fig = plt.figure()
fig.suptitle('ROC curves')
fig.set_figheight(20)
fig.set_figwidth(20)

plt.subplot(3, 3, 1)
auc_logistic=roc_auc_score(test_Y, logisticRegr_pred)
fpr, tpr, thresholds= roc_curve(test_Y, logisticRegr_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('AUC logistic Regression %0.3f' %auc_logistic)

plt.subplot(3, 3, 2)
auc_knn= roc_auc_score(test_Y, knn_pred)
fpr, tpr, thresholds= roc_curve(test_Y, knn_pred)
plt.plot(fpr,tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC KNN: %0.3f' %auc_knn)

plt.subplot(3,3,3)
auc_NaiveBayes= roc_auc_score(test_Y, NaiveBayes_pred)
fpr, tpr, thresholds= roc_curve(test_Y, NaiveBayes_pred)
plt.plot(fpr,tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC Naive Bayes: %0.3f' %auc_NaiveBayes)

plt.subplot(3, 3, 4)
auc_tree=roc_auc_score(test_Y, tree_pred)
fpr, tpr, thresholds= roc_curve(test_Y, tree_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('AUC Tree Classifier: %0.3f' %auc_tree)

plt.subplot(3, 3, 5)
auc_RandomForestClassifier=roc_auc_score(test_Y, RandomForestClassifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, RandomForestClassifier_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('AUC Random Forest: %0.3f' %auc_RandomForestClassifier)

plt.subplot(3, 3, 6)
auc_SVC=roc_auc_score(test_Y, SupportVectorClassifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, SupportVectorClassifier_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('AUC Support Vector Machine: %0.3f' %auc_SVC)

plt.subplot(3, 3, 7)
auc_MLP=roc_auc_score(test_Y, MLP_Classifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, MLP_Classifier_pred)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')
plt.title('AUC Multi Layer Percepton Classifier: %0.3f' %auc_MLP)

plt.show()
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)

auc_logistic=roc_auc_score(test_Y, logisticRegr_pred)
fpr, tpr, thresholds= roc_curve(test_Y, logisticRegr_pred)
plt.plot(fpr, tpr)

auc_knn= roc_auc_score(test_Y, knn_pred)
fpr, tpr, thresholds= roc_curve(test_Y, knn_pred)
plt.plot(fpr,tpr)

auc_NaiveBayes= roc_auc_score(test_Y, NaiveBayes_pred)
fpr, tpr, thresholds= roc_curve(test_Y, NaiveBayes_pred)
plt.plot(fpr,tpr)

auc_tree=roc_auc_score(test_Y, tree_pred)
fpr, tpr, thresholds= roc_curve(test_Y, tree_pred)
plt.plot(fpr, tpr)

auc_RandomForestClassifier=roc_auc_score(test_Y, RandomForestClassifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, RandomForestClassifier_pred)
plt.plot(fpr, tpr)

auc_SVC=roc_auc_score(test_Y, SupportVectorClassifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, SupportVectorClassifier_pred)
plt.plot(fpr, tpr)

auc_MLP=roc_auc_score(test_Y, MLP_Classifier_pred)
fpr, tpr, thresholds= roc_curve(test_Y, MLP_Classifier_pred)
plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'r--')

plt.legend(['Logistic regression', 'KNN', 'NaiveBayes', 'Tree',
           'Random Forest', 'Support Vecto Machine', 'Multi Layer Perceptron'])

plt.xlabel('False Postive Rate')
plt.ylabel ('True Postive Rate')

plt.title('ROC curves')

plt.show()
train_titanic=pd.read_csv('/kaggle/input/titanic/train.csv')
test_titanic=pd.read_csv('/kaggle/input/titanic/test.csv')
train_X_submission=train_titanic_dummy[train_titanic_dummy.columns[1:]]
train_Y_submission=train_titanic_dummy[train_titanic_dummy.columns[:1]]
test_titanic.drop(['Name','SibSp','Ticket','Cabin'],axis=1,inplace=True)
test_X_submission=pd.get_dummies (test_titanic)

train_X_submission.head()
missing_values_submission=test_X_submission.isnull().sum()
missing_values_percentage_submission=test_X_submission.isnull().sum()/test_X_submission.isnull().count()
table_missing_values_submission=pd.concat([missing_values_submission, missing_values_percentage_submission], axis=1, keys=['total', 'percentage']).sort_values(by='total',ascending=False)
table_missing_values_submission
#Onde hai máis missing values é en 'Cabin', despois Age e por último Embarked.
#Para aqueles valores onde non hai datos, vams a meter o valor medio da columna de Age
test_X_submission['Age']=test_X_submission['Age'].fillna(test_X_submission['Age'].mean())
#Para o fare tamén metemos o valor medio
test_X_submission['Fare']=test_X_submission['Fare'].fillna(test_X_submission['Fare'].mean())
test_X_submission=test_X_submission.set_index(['PassengerId'])
missing_values_submission=test_X_submission.isnull().sum()
missing_values_percentage_submission=test_X_submission.isnull().sum()/test_X_submission.isnull().count()
table_missing_values_submission=pd.concat([missing_values_submission, missing_values_percentage_submission], axis=1, keys=['total', 'percentage']).sort_values(by='total',ascending=False)
table_missing_values_submission
tree_clasiffier=DecisionTreeClassifier(max_depth=3)
tree_clasiffier.fit(train_X_submission, train_Y_submission)
#Creamos os y predictivos
test_Y_submission=tree_clasiffier.predict(test_X_submission)
output = pd.DataFrame({'PassengerId': test_X_submission.index.values, 'Survived': test_Y_submission})
output.to_csv('my_submission.csv', index=False)
output.shape