import pandas as pd

import numpy as np
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train
train.isnull().sum()
from scipy import stats



survived = train[train['Survived']==1]

did_not_survive = train[train['Survived']==0]



# libraries

import numpy as np

import matplotlib.pyplot as plt

 

plt.rcParams["figure.figsize"] = (10,6)

plt.rcParams.update({'font.size': 13})

# set width of bar

barWidth = 0.25





male = [len(survived[survived['Sex']=='male']), \

        len(did_not_survive[did_not_survive['Sex']=='male'])]

female = [len(survived[survived['Sex']=='female']), \

         len(did_not_survive[did_not_survive['Sex']=='female'])]



# set height of bar

bars1 = [len(survived[survived['Sex']=='male']), \

         len(did_not_survive[did_not_survive['Sex']=='male'])]

bars2 = [len(survived[survived['Sex']=='female']), \

         len(did_not_survive[did_not_survive['Sex']=='female'])]

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

 

# Make the plot

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survived')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Did not survive')

 

# Add xticks on the middle of the group bars

plt.xlabel('Sex', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], ['Male', 'Female'])

 

# Create legend & Show graphic

plt.legend()

plt.show()



p_value = stats.chi2_contingency([male, female])[1]

print("Chi-Square P-Value: " + str(p_value))


# libraries

import numpy as np

import matplotlib.pyplot as plt

 

# set width of bar

barWidth = 0.25



class1 = [len(survived[survived['Pclass']==1]), \

        len(survived[survived['Pclass']==2]), \

        len(survived[survived['Pclass']==3]),]

class2 = [len(did_not_survive[did_not_survive['Pclass']==1]), \

        len(did_not_survive[did_not_survive['Pclass']==2]), \

        len(did_not_survive[did_not_survive['Pclass']==3])]

 

# Set position of bar on X axis

r1 = np.arange(len(class1))

r2 = [x + barWidth for x in r1]



# Make the plot

plt.bar(r1, class1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survived')

plt.bar(r2, class2, color='#557f2d', width=barWidth, edgecolor='white', label='Did not survive')

 

# Add xticks on the middle of the group bars

plt.xlabel('Class', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(class1))], ['Class 1', 'Class 2', 'Class 3'])

 

# Create legend & Show graphic

plt.legend()

plt.show()





class1 = [len(survived[survived['Pclass']==1]), \

        len(did_not_survive[did_not_survive['Pclass']==1])]

class2 = [len(survived[survived['Pclass']==2]), \

        len(did_not_survive[did_not_survive['Pclass']==2])]

class3 = [len(survived[survived['Pclass']==3]), \

        len(did_not_survive[did_not_survive['Pclass']==3])]



p_value = stats.chi2_contingency([class1, class2, class3])[1]

print("Chi-Square P-Value: " + str(p_value))


# libraries

import numpy as np

import matplotlib.pyplot as plt

 

# set width of bar

barWidth = 0.25



class1 = [len(survived[survived['Embarked']=='Q']), \

        len(survived[survived['Embarked']=='C']), \

        len(survived[survived['Embarked']=='S']),]

class2 = [len(did_not_survive[did_not_survive['Embarked']=='Q']), \

        len(did_not_survive[did_not_survive['Embarked']=='C']), \

        len(did_not_survive[did_not_survive['Embarked']=='S'])]

 

# Set position of bar on X axis

r1 = np.arange(len(class1))

r2 = [x + barWidth for x in r1]



# Make the plot

plt.bar(r1, class1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survived')

plt.bar(r2, class2, color='#557f2d', width=barWidth, edgecolor='white', label='Did not survive')

 

# Add xticks on the middle of the group bars

plt.xlabel('Embarked', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(class1))], ['Queenstown', 'Cherbourg', 'Southampton'])

 

# Create legend & Show graphic

plt.legend()

plt.show()



Q_embark = [len(survived[survived['Embarked']=='Q']), \

        len(did_not_survive[did_not_survive['Embarked']=='Q'])]

C_embark = [len(survived[survived['Embarked']=='C']), \

        len(did_not_survive[did_not_survive['Embarked']=='C'])]

S_embark = [len(survived[survived['Embarked']=='S']), \

        len(did_not_survive[did_not_survive['Embarked']=='S'])]



p_value = stats.chi2_contingency([Q_embark, C_embark, S_embark])[1]

print("Chi-Square P-Value: " + str(p_value))
import matplotlib.pyplot as plt

import numpy as np

import random



plt.rcParams["figure.figsize"] = (20,3)

plt.rcParams.update({'font.size': 13})



data = {}

data['Age'] = {

    'Survived': list(survived.dropna()['Age']),

    'Did not Survive': list(did_not_survive.dropna()['Age']),

}

data['SibSp'] = {

    'Survived': list(survived.dropna()['SibSp']),

    'Did not Survive': list(did_not_survive.dropna()['SibSp']),

}

data['Parch'] = {

    'Survived': list(survived.dropna()['Parch']),

    'Did not Survive': list(did_not_survive.dropna()['Parch']),

}

data['Fare'] = {

    'Survived': list(survived.dropna()['Fare']),

    'Did not Survive': list(did_not_survive.dropna()['Fare']),

}



fig, axes = plt.subplots(ncols=4)

fig.subplots_adjust(wspace=0)



for ax, name in zip(axes, ['Age', 'SibSp', 'Parch', 'Fare']):

    ax.boxplot([data[name][item] for item in ['Survived', 'Did not Survive']])

    ax.set(xticklabels=['Survived', 'Did not Survive'], xlabel=name)

    ax.margins(0.05) # Optional



plt.show()
train = pd.get_dummies(train, columns=['Sex', 'Pclass', 'Embarked'])
train.corr()
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()



X = train[['Fare', 'Sex_male', 'Pclass_1', 'Pclass_3', 'Embarked_C', 'Embarked_S']]

y = train['Survived']

X = my_imputer.fit_transform(X)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)



clf = LogisticRegression(solver='liblinear')



# Create regularization hyperparameter space

C = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 10, 100]

penalty = ['l1','l2']



hyperparameters = dict(C=C, penalty=penalty)



grid_clf_acc = GridSearchCV(clf, param_grid=hyperparameters, cv=5, verbose=0)

grid_clf_acc.fit(X_train, y_train)
grid_clf_acc.best_params_
clf = LogisticRegression(solver='liblinear', **grid_clf_acc.best_params_)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



clf.fit(X_train, y_train)



X_test = scaler.transform(X_test)



y_pred = clf.predict(X_test)



print(accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred, target_names=['Survived', 'Did not survive']))
lr = clf.fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)



clf = RandomForestClassifier()



hyperparameters = {

    'bootstrap': [True, False],

    'max_depth': [80, 90, 100, 110],

    'n_estimators': [100, 200, 300, 1000]

}



grid_clf_acc = GridSearchCV(clf, param_grid=hyperparameters, cv=5, verbose=0)

grid_clf_acc.fit(X_train, y_train)
clf = RandomForestClassifier(**grid_clf_acc.best_params_)
clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)



print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Survived', 'Did not survive']))
rf = clf.fit(X, y)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test = pd.get_dummies(test, columns=['Sex', 'Pclass', 'Embarked'])

X = test[['Fare', 'Sex_male', 'Pclass_1', 'Pclass_3', 'Embarked_C', 'Embarked_S']]

X = my_imputer.fit_transform(X)
pred = rf.predict(X)
submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': pred})
submission.to_csv('/kaggle/working/titanic',index=False)