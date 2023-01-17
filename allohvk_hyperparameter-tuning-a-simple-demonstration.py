import pandas as pd

test_data = pd.read_csv ('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
from sklearn.tree import DecisionTreeClassifier



##Fill in missing values, convert features to numbers where applicable

for df in [train_data,test_data]:

    df['Sex_boolean']=df['Sex'].map({'male':1,'female':0})

    df['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

    df['Age'].fillna(train_data['Age'].mean(), inplace=True)



##Generate model and predict

model=DecisionTreeClassifier().fit(train_data[['Pclass','Sex_boolean']],train_data['Survived'])

predictions=model.predict(test_data[['Pclass', 'Sex_boolean']])



##Append the predictions to the PassengerID and convert to CSV

pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions}).to_csv('KaggleOutput', index=False)



##A detailed explanation of the above code can be found at:

##https://www.kaggle.com/allohvk/titanic-simplest-tutorial-ever-code-as-a-story
from sklearn import tree

import matplotlib.pyplot as plt



##Plot the decision tree

plt.figure(figsize=(40,20))  

_ = tree.plot_tree(model, feature_names = ['Pclass', 'Sex_boolean'], filled=True, fontsize=30, rounded = True)
##Meanwhile you can even print the final rules that go into taking a decision

##This is a new feature of Scikit and shows exactly how the machines 'make the

##program from the data'

from sklearn.tree.export import export_text



tree_rules = export_text(model, feature_names=['Pclass', 'Sex_boolean'])

print(tree_rules)
from sklearn.model_selection import train_test_split



#Split data into training features and labels

X, y = train_data[['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare']], train_data['Survived']



X_train, X_validate, y_train, y_validate = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 200)
model=DecisionTreeClassifier(max_depth=4).fit(train_data[['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare']],train_data['Survived'])



plt.figure(figsize=(40,20))  

_ = tree.plot_tree(model, feature_names = ['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare'], filled=True, fontsize=24, rounded = True)

display(model.get_params())
from sklearn.metrics import roc_curve, auc

from matplotlib.legend_handler import HandlerLine2D





##Let us compare difference in results for depth=3 and depth=12

for max_depth in [3,12]:



    ##Train on training data. We call the model dt henceforth

    dt = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)



    ##Predict on training data

    train_prediction = dt.predict(X_train)



    ##Get the results - false +ve and true +ves. We will discuss this in a minute

    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_train,train_prediction)



    ##Generate the roc. We will discuss this also

    roc_auc = auc(false_positive_rate, true_positive_rate)

    print(roc_auc)



    ##Plot the ROC in a graph

    plt.plot(false_positive_rate, true_positive_rate)

    plt.axis([0,1,0,1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.show()
for max_depth in [3,12]:



    ##Train on training data

    dt = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)



    ##Predict on validation data

    validate_prediction = dt.predict(X_validate)



    ##Get the results - false +ve and true +ves

    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_validate,validate_prediction)



    ##Generate the roc

    roc_auc = auc(false_positive_rate, true_positive_rate)

    print(roc_auc)



    ##Plot the ROC in a graph

    plt.plot(false_positive_rate, true_positive_rate)

    plt.axis([0,1,0,1])

    plt.xlabel('False Positive Rate - Validation')

    plt.ylabel('True Positive Rate - Validation')

    plt.show()
train_results = []

validate_results = []



for max_depth in range(1,20):

    dt = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)

    

    train_prediction = dt.predict(X_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_prediction)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    print('For Depth = ', max_depth, ' for training data, the AUC = ', roc_auc)

    

    validate_prediction = dt.predict(X_validate)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validate, validate_prediction)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    validate_results.append(roc_auc)

    print('For Depth = ', max_depth, ' for validation data, the AUC = ', roc_auc, '\n')







line1 = plt.plot(range(1,20), train_results, label="Training data AUC")

line2 = plt.plot(range(1,20), validate_results, label="Validation data AUC")

plt.show()
from sklearn.model_selection import GridSearchCV 

import numpy as np



params = {'max_depth': range(1,15),

          'min_samples_leaf': np.arange(1,25,2), 

          'max_features': range(1,6),

          'criterion' : ['gini', 'entropy'],

          'splitter' : ['random', 'best'],

          'random_state' : [1] }



grid_dt = GridSearchCV(estimator=dt, param_grid=params, scoring='accuracy', cv=8, n_jobs=-1)

grid_dt.fit(X, y)

##Note above we are using the full trainig data. We dont need to split.

##Gridsearch automatically splits this into 8 sets, trains on 7 and tests on the 8th 

##for each combination of hyperparameters



print('Best hyerparameters:\n', grid_dt.best_params_)

print('Best CV roc aus', grid_dt.best_score_)

##print('Score on validation data', grid_dt.best_estimator_.score(X_validate,y_validate))
imp_features = pd.DataFrame({'feature':X_train.columns,'importance':np.round(grid_dt.best_estimator_.feature_importances_,2)})

print(imp_features.sort_values('importance',ascending=False))



##You can even display the individual scores. I commented this because it is a lengthy o/p

##for mean_score, params in zip(grid_dt.cv_results_['mean_test_score'], grid_dt.cv_results_['params']):

##display(mean_score, params)
predictions=grid_dt.best_estimator_.predict(test_data[['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare']])



pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions}).to_csv('KaggleOutput', index=False)