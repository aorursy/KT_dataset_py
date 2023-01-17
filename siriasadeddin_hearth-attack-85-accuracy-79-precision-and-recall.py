import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn import preprocessing

from sklearn import tree

from sklearn import metrics

import graphviz
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.T
columns=df.columns

columns_new=[]

for i in columns:

    columns_new.append(any(df[i].isnull()|df[i].isnull()))

df=df.drop(columns[columns_new],axis=1)
df.shape
ax = sns.countplot(df.DEATH_EVENT,label="Count")       # M = 212, B = 357

df.DEATH_EVENT.value_counts()
plt.hist(df.age[df.DEATH_EVENT==1],label='muere',bins=20,density = True,  

                            color ='green', 

                            alpha = 0.7)

plt.hist(df.age[df.DEATH_EVENT==0],label='vive',bins=20,density = True,  

                            color ='blue', 

                            alpha = 0.7)

plt.legend()

plt.show()
x = df.drop(['DEATH_EVENT','time'], axis=1)

X = x.values

y = df['DEATH_EVENT'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.tree import DecisionTreeClassifier



for i in range(4, 6):

    death_tree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

    death_tree.fit(x_train, y_train)

    pred_death = death_tree.predict(x_test)

    print('with max_depth of {} , death_tree accuracy is {}'.format (i, metrics.accuracy_score(y_test, pred_death)))
def conf_matrix(matrix,pred):

    class_names= [0,1]# name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.show()
cnf_matrix = metrics.confusion_matrix(y_test, pred_death,normalize='true')

conf_matrix(cnf_matrix,y_test)
print(classification_report(y_test, pred_death))
plt.figure(figsize=(20, 20))

filename ='death_tree.png'



featureNames = df.columns[:-2]

targetNames = ['death', 'alive']



dot_data = tree.export_graphviz(death_tree, feature_names=featureNames, class_names=targetNames,filled=True)



# Draw graph

graph = graphviz.Source(dot_data, format="png") 

graph