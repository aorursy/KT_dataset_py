import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier
#Loading the dataset..

df = pd.read_csv('../input/mushroom.csv')
df.head()
df.columns
# converting the data from categorical to ordinal ..

labelencoder=LabelEncoder()

for column in df.columns:

    df[column] = labelencoder.fit_transform(df[column])
#checking the information of the dataset......

df.info()
#dropping the column "veil-type" is 0 

df=df.drop(["veil-type"],axis=1)

df.head()
df.describe()
#Question a:

plt.figure()

pd.Series(df['edible']).value_counts().sort_index().plot(kind = 'bar')

plt.ylabel("Count")

plt.xlabel("edible")

plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)');

plt.figure(figsize=(14,12))

sns.heatmap(df.corr(),linewidths=.1, annot=True)

plt.yticks(rotation=0);
df.corr()
df[['edible', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='edible', ascending=False)
#Looking closely at the feature 'gill-color'

new_var=df[['edible', 'gill-color']]

new_var=new_var[new_var['gill-color']<=3.5]

sns.factorplot('edible', col='gill-color', data=new_var, kind='count', size=2.5, aspect=.8, col_wrap=4);
new_var=df[['edible', 'gill-color']]

new_var=new_var[new_var['gill-color']>3.5]



sns.factorplot('edible', col='gill-color', data=new_var, kind='count', size=2.5, aspect=.8, col_wrap=4);
X=df.drop(['edible'], axis=1)

Y=df['edible']
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)
# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(X_train, Y_train)



# Print the score of the fitted random forest

print(my_forest.score(X, Y)*100)

acc_randomforest=(my_forest.score(X, Y)*100)
def plot_confusion_matrix(df, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df, cmap=cmap)

    plt.title('Confusion Matrix')

    plt.colorbar()

    plt.ylabel('Actual')

    plt.xlabel('Predicted')



plot_confusion_matrix(df)
clf = DecisionTreeClassifier()

clf = clf.fit(X_train, Y_train)
features_list = X.columns.values

feature_importance = clf.feature_importances_

sorted_idx = np.argsort(feature_importance)



plt.figure(figsize=(5,7))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()

X=df.drop(['edible'], axis=1)

Y=df['edible']

y_pred=clf.predict(X_test)
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred7 = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)

acc_decision_tree
cfm=confusion_matrix(Y_test, y_pred)



sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)

plt.title('Decision Tree Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label');
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=3, n_neighbors=10, p=2, 

                           weights='uniform')

knn.fit(X_train, Y_train)

knn_predictions = knn.predict(X_test)

acc_knn = round(knn.score(X_test, Y_test) * 100, 2)

acc_knn
objects = ('Decision Tree', 'Random Forest','KNN Model ')

x_pos = np.arange(len(objects))

accuracies1 = [acc_decision_tree, acc_randomforest, acc_knn]

    

plt.bar(x_pos, accuracies1, align='center', alpha=0.5, color='b')

plt.xticks(x_pos, objects, rotation='vertical')

plt.ylabel('Accuracy')

plt.title('Classifier Outcome')

plt.show()