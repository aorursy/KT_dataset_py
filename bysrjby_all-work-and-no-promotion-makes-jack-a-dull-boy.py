import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('../input/HR_comma_sep.csv')



from sklearn import preprocessing

le1 = preprocessing.LabelEncoder()

le1.fit(data.sales)

data.sales = le1.transform(data.sales)



le2 = preprocessing.LabelEncoder()

le2.fit(data.salary)

data.salary = le2.transform(data.salary)
%matplotlib inline

corrs = data.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corrs)

fig.colorbar(cax)

ticks = np.arange(0,10,1)

#ax.set_xticks(ticks)

ax.set_yticks(ticks)

#ax.set_xticklabels(data.columns)

ax.set_yticklabels(data.columns)

plt.show()
#scale features

SS = preprocessing.StandardScaler()

SS.fit_transform(data)



df = pd.DataFrame(dict(x=data.promotion_last_5years, y=data.average_montly_hours, label=data.left))

groups = df.groupby('label')



# Plot

fig, ax = plt.subplots()

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)

ax.legend()



plt.show()
#split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('left',axis=1), data.left, test_size=0.33, random_state=1)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

labels = ['stay','leave']



fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

plt.show()

print(cm)
#classification after PCA

from sklearn import decomposition

pca = decomposition.PCA(n_components=6)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train_pca,y_train)

y_pred_pca = clf.predict(X_test_pca)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred_pca)

labels = ['stay','leave']



fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

plt.show()

print(cm)
data.groupby('left').last_evaluation.hist(alpha=0.4)

left_employees = data.loc[data.left==1]

left_employees = left_employees.assign(good = left_employees.last_evaluation > 0.75)

left_employees.groupby('good').salary.hist(alpha=0.4)
le2.inverse_transform([0,1,2])