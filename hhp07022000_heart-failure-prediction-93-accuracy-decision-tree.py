import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.columns
df.shape
missing_data = df.isnull()
for column in df:

    print(column)

    print(missing_data[column].value_counts())

    print('')
plt.style.use('ggplot')

Y = df[['creatinine_phosphokinase', 'ejection_fraction', 'platelets',

        'serum_creatinine','age', 'serum_sodium']]

for col in Y:

    sns.boxplot(x='DEATH_EVENT', y=col, data=df)

    plt.title( 'Death caused by ' + col)

    plt.show()
x = ['male', 'female']

y = df['sex'].value_counts()

plt.style.use('fivethirtyeight')

plt.bar(x, y, color='r')

plt.title('Death Rate Comparison Between sex')

plt.show()

x =['non-smokers', 'smokers']

y = df['smoking'][(df['DEATH_EVENT']==1)].value_counts()

plt.bar(x, y, color='b')

plt.title('death count between smokers and non smokers')

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
x = df.drop('DEATH_EVENT', axis=1)

X = x.values

y = df['DEATH_EVENT'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn import metrics
for i in range(4, 10):

    death_tree = DecisionTreeClassifier(criterion='entropy', max_depth=i)

    death_tree.fit(x_train, y_train)

    pred_death = death_tree.predict(x_test)

    print('with max_depth of {} , death_tree accuracy is {}'.format (i, metrics.accuracy_score(y_test, pred_death)))
death_tree = DecisionTreeClassifier(criterion='entropy', max_depth=6)

death_tree.fit(x_train, y_train)

pred_death = death_tree.predict(x_test)

print('with max_depth of {} , death_tree accuracy is {}'.format (6, metrics.accuracy_score(y_test, pred_death)))
from sklearn import tree

import graphviz
plt.figure(figsize=(200, 200))

filename ='death_tree.png'



featureNames = df.columns[:-1]

targetNames = ['death', 'alive']



dot_data = tree.export_graphviz(death_tree, feature_names=featureNames, class_names=targetNames,filled=True)



# Draw graph

graph = graphviz.Source(dot_data, format="png") 

graph