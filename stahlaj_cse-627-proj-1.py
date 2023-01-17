%pylab inline
import os

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelBinarizer, Imputer

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.svm import SVC

from sklearn import linear_model

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt



#read in the csv

df = pd.read_csv("../input/train.csv")



#format the test_data for predictions

df_test = pd.read_csv("../input/test.csv")



df
# visualize the different predictors vs the target values



fig, axes = subplots(3)

fig.set_size_inches(10, 25)

j = 0;



predictors = array(['Pclass', 'SibSp', 'Parch'])



for predictor in predictors:

    values = unique(df[predictor])

    survived = numpy.zeros(len(values))

    totals = numpy.zeros(len(values))

    

    

    for i in range(0, len(values)):

        survived[i] = sum(df['Survived'][df[predictor] == values[i]] == 1)

        totals[i] = len(df['Survived'][df[predictor] == values[i]])

    sca(axes[j])

    

    bar(arange(len(values)), survived/totals)

    xticks(arange(len(values)), values)

    

    axes[j].set_title( 'Survival Rate by ' + predictor)

    

    j+= 1

    

    
# a look at the embarked data

sns.factorplot(x="Embarked", kind="count", size=8, data=df)

sns.factorplot(x="Embarked", hue="Survived", col="Survived", kind="count", size=8, data=df)
sns.factorplot(x="Embarked", y="Survived", kind="bar", size=8, data=df)

sns.plt.title('Survival Rate by Embarked')
sns.factorplot(x="Embarked", hue = 'Pclass', kind="count", size=8, data=df)
sns.factorplot(x="Sex", hue="Survived", col="Survived", kind="count", size=8, data=df)

sns.factorplot(x="Sex", y="Survived", kind="bar", size=6, data=df)

sns.plt.title('Survival Rate by Sex')
sns.factorplot(x="Sex", y="Survived", hue="Pclass", kind="bar", size=8, data=df)
def age_range(age):

    if age > 60: return 'elderly'

    elif 39 < age <= 60: return 'adult'

    elif 18 < age <= 39: return 'young adult'

    elif age <= 18: return 'child'

    else: return 'unknown'  

    

df['Age_Groups'] = df.Age.map(age_range)

sns.factorplot(x="Age_Groups", hue="Survived", kind="count", size = 8, data=df)

sns.plt.title('Survival Count by Age Group')

sns.factorplot(x="Age_Groups", y="Survived", kind="bar", size = 8, data=df)

sns.plt.title('Survival Rate by Age Group')

sns.factorplot(x="Age_Groups", hue="Survived", col="Sex", kind="count", aspect=0.5, size = 8, data=df)
sns.factorplot(x="Age_Groups", y="Survived", col="Pclass", kind="bar", aspect=0.5, size = 8, data=df)
sns.factorplot(x="Survived", y="Fare", hue="Sex", col="Sex", kind="bar", size = 8, data=df)

sns.factorplot(x="Pclass", y="Fare", col = "Survived", kind="box", size = 8, data=df)
def prefix(name):

    return name[name.find(", ") + 2 :name.find(".")].strip()
df["Title"] = df.Name.map(prefix)
sns.factorplot(x="Title", y="Survived", kind="bar", size=8, aspect = 2, data=df)
df.loc[(df.Title == 'Don') |

       (df.Title == 'Sir') |

       (df.Title == 'Col') |

       (df.Title == 'Don') |

       (df.Title == 'Dr') |

       (df.Title == 'Jonkheer') |

       (df.Title == 'Capt') |

       (df.Title == 'Major') |

       (df.Title == 'Rev'), 'Title']='Sir'
df.loc[(df.Title == 'Lady') |

       (df.Title == 'the Countess') |

       (df.Title == 'Mme') |

       (df.Title == 'Ms') |

       (df.Title == 'Mlle'), 'Title']='Lady'


sns.factorplot(x="Title", y="Survived", kind="bar", size=8, data=df)

df["Family_size"] = df.SibSp + df.Parch + 1
sns.factorplot(x="Family_size", y="Survived", kind="bar", size=8, data=df)
lb = LabelBinarizer()

scaler = StandardScaler()

imputer = Imputer(strategy="median")
df.groupby(['Pclass','Title'])['Age'].median()
df['Age'] = df.groupby(['Pclass','Title'])['Age'].transform(lambda x:x.fillna(x.median()))
df[df.Embarked.isnull()]
sns.factorplot(x="Embarked", kind="count", data=df[(df['Sex'] == 'female') & (df['Pclass'] == 1)], size=5)
embark_filler = 'S'

df.loc[(df.Embarked.isnull()), 'Embarked'] = embark_filler



embark = lb.fit_transform(df['Embarked'])

embark_columns = lb.classes_

embarked = pd.DataFrame(data=embark, columns=embark_columns)
title = lb.fit_transform(df['Title'])

title_columns = lb.classes_

titles = pd.DataFrame(data=title, columns=title_columns)
pclass = lb.fit_transform(df['Pclass'])

pclass_columns = ['Class1', 'Class2', 'Class3']

pclasses = pd.DataFrame(data=pclass, columns=pclass_columns)
sex = lb.fit_transform(df['Sex'])

genders = pd.DataFrame(data=sex, columns=['Sex_transform'])
numerical_attributes=['SibSp', 'Parch', 'Age', 'Family_size', 'Fare']

data = pd.concat([df[numerical_attributes], embarked, titles, pclasses, genders], axis=1)

numerical_attributes=['SibSp', 'Parch', 'Age', 'Family_size', 'Fare']

# scale the numerical attributes

data[numerical_attributes] = scaler.fit_transform(data[numerical_attributes])

targets = df['Survived']
seed = 12345
# determine the best SVC model



X_train, X_test, y_train, y_test = train_test_split(

    data, targets, test_size=0.25, random_state=0)



tuned_parameters = [{'kernel': ['rbf'], 

                    'gamma': np.arange(1e-4, 1e-2, 1e-3),#[1e-1, 1e-2, 1e-3],

                     'C': [150, 175, 180, 200]}]



clf = GridSearchCV(SVC(), tuned_parameters, cv=10,

                       scoring='accuracy')

clf.fit(X_train, y_train)



print('Best parameters for our model: ')

print(clf.best_params_)



y_true, y_pred = y_test, clf.predict(X_test)

print('\nClassification Report for SVC Model with above parameters:')

print(classification_report(y_true, y_pred))

# determine the best RandomForest model



X_train, X_test, y_train, y_test = train_test_split(

    data, targets, test_size=0.5, random_state=0)



tuned_parameters = [{'n_estimators': [300, 400], 

                     'criterion': ['gini', 'entropy'],

                     'max_features': ['auto', 'sqrt', 'log2'],

                     'max_depth': [5, 7],

                     'bootstrap': [True, False]}]



clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,

                       scoring='accuracy')



clf.fit(X_train, y_train)



print('Best parameters for our model: ')

print(clf.best_params_)



y_true, y_pred = y_test, clf.predict(X_test)

print('\nClassification Report for Random Forest Model with above parameters:')

print(classification_report(y_true, y_pred))
clf = RandomForestClassifier(n_estimators = 300, bootstrap = False, max_depth = 5)

clf.fit(data, targets)



importances = clf.feature_importances_

features = np.array(['SibSp', 'Parch', 'Age', 'Family_size', 'Fare', 'C', 'Q', 'S', 'Lady', 'Master', 'Miss', 'Mr',

                     'Mrs', 'Sir', 'Class1', 'Class2', 'Class3', 'Sex_transform'])

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features) ## removed [indices]

plt.xlabel('Relative Importance')

plt.show()
#determine the best Adaboost model



X_train, X_test, y_train, y_test = train_test_split(

    data, targets, test_size=0.5, random_state=0)



tuned_parameters = [{'n_estimators': [50, 100, 150], 

                     'learning_rate': [1, .5, .25, .1],

                     'algorithm': ['SAMME', 'SAMME.R']}]



clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=10,

                       scoring='accuracy')



clf.fit(X_train, y_train)



print('Best parameters for our model: ')

print(clf.best_params_)



y_true, y_pred = y_test, clf.predict(X_test)

print('\nClassification Report for Random Forest Model with above parameters:')

print(classification_report(y_true, y_pred))
# Tune parameters and fit Gradient Boosting Classifier to identify the best parameters



X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.25, random_state=seed)



parameters = {

   'n_estimators': [1000, 1100],

    'max_depth': [4, 6, 8, 10, 12],

    'min_samples_leaf': [5, 11, 17, 21],

    'max_features': ['auto', 'sqrt', 'log2'],

    'learning_rate': [0.001, 0.01, 0.1]   

}



gbc = GradientBoostingClassifier(random_state=seed)

gbc_gscv = GridSearchCV(estimator=gbc, cv=10, param_grid=parameters, scoring='accuracy').fit(X_train, y_train)



print('Best parameters for our model: ')

print(gbc_gscv.best_params_)



y_true, y_pred = y_test, gbc_gscv.predict(X_test)

print('\nClassification Report for Gradient Boosting Classifier Model with above parameters:')

print(classification_report(y_true, y_pred))
params = {'n_estimators': 1000, 'max_depth': 6,

          'min_samples_leaf': 5, 'learning_rate': 0.001,

          'max_features': 'sqrt', 'random_state': seed}



gbcm = GradientBoostingClassifier(**params).fit(data, targets)



importances = gbcm.feature_importances_

features = np.array(['SibSp','Parch','Fare', 'Age',

              'Family_Size', 'Sex_transform',

              'Class1','Class2','Class3',

              'C','Q','S','Lady','Master','Miss',

              'Mr','Mrs','Sir'])



indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features) ## removed [indices]

plt.xlabel('Relative Importance')

plt.show()
X_train = data.iloc[:600, :].astype(np.float32)

y_train = targets.iloc[:600].astype(np.float32)

X_test = data.iloc[600:, :].astype(np.float32)

y_test = targets.iloc[600:].astype(np.float32)



feature_columns = [tf.feature_column.numeric_column('x', shape=[18])]



#FOR TESTING (evaluate 144 different combos of hidden layers)

accuracies = np.zeros((17, 17))





for i in range(17):

    for j in range(17):

        

        classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,

                                                  hidden_units=[i+16, j+16],

                                                  n_classes=2,

                                                  optimizer=tf.train.AdamOptimizer())



        train_input_fn = tf.estimator.inputs.numpy_input_fn(

              x={"x": np.array(X_train)},

              y=np.array(y_train),

              num_epochs=None,

              shuffle=True)



        classifier.train(input_fn=train_input_fn, steps=5000)



        test_input_fn = tf.estimator.inputs.numpy_input_fn(

              x={"x": np.array(X_test)},

              y=np.array(y_test),

             num_epochs=1,

              shuffle=False)



        accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

        accuracies[i, j] = accuracy_score



print("Finished")
accuracies = pd.DataFrame(accuracies)



plt.figure(figsize = (10,10))

ax = sns.heatmap(data=accuracies, cmap="YlGnBu", annot=True, linewidths=.5, cbar_kws={'label': 'Accuracy on Test Set'})

ax.set(xlabel='# of 2nd hidden layer nodes', ylabel='# of 1st hidden layer nodes')

ax.set_xticklabels(np.arange(1, 18))

ax.set_yticklabels(reversed(np.arange(1, 18)))
class_rates = np.array([.81, .82, .79, .79, .85])

features = np.array(['SVC', 'RFC', 'AdaBoost', 'GBC', 'DNN'])



plt.title('Tuned Model Accuracy Rates')

plt.barh(range(len(class_rates)), class_rates, color='b', align='center')

plt.yticks(range(len(class_rates)), features) ## removed [indices]

plt.xlabel('Accuracy')

plt.show()
feature_columns = [tf.feature_column.numeric_column('x', shape=[18])]

skf = StratifiedKFold(n_splits = 10, shuffle = True)

j = 0



#create a numpy array with the CV predictions for the training data

predictions = np.zeros((len(data), 5))



#fit all of the models with the training data and fill and array with the predictions for the test data

for train_index, test_index in skf.split(data, targets):

    X_train = data.iloc[train_index]

    X_test = data.iloc[test_index]

    y_train = targets.iloc[train_index]

    y_test = targets.iloc[test_index]



    rf = RandomForestClassifier(max_depth = 9, n_estimators = 300)

    rf.fit(X_train, y_train)



    sv = SVC(C=100, gamma = .005, kernel = 'rbf')

    sv.fit(X_train, y_train)



    ada = AdaBoostClassifier(algorithm = 'SAMME', learning_rate = .5, n_estimators = 100)

    ada.fit(X_train, y_train)

    

    params={'n_estimators': 1000, 'max_depth': 12,

            'min_samples_leaf': 5, 'learning_rate': 0.001,

            'max_features': 'sqrt', 'random_state': seed}

    

    gbcm = GradientBoostingClassifier(**params).fit(X_train, y_train)

    

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,

                                                  hidden_units=[12, 3],

                                                  n_classes=2,

                                                  optimizer=tf.train.AdamOptimizer())



    train_input_fn = tf.estimator.inputs.numpy_input_fn(

              x={"x": np.array(X_train)},

              y=np.array(y_train),

              num_epochs=None,

              shuffle=True)



    classifier.train(input_fn=train_input_fn, steps=5000)



    predict_input_fn = tf.estimator.inputs.numpy_input_fn(

            x={"x": np.array(X_test)},

            num_epochs=1,

            shuffle=False)



    kaanPredictions = list(classifier.predict(input_fn=predict_input_fn))

    kaanPredictions=np.array(kaanPredictions)

    finalPreds = []

    for i in range(len(kaanPredictions)):

        temp = kaanPredictions[i]['class_ids'][0]

        finalPreds.append(temp)





    rf_test = rf.predict(X_test)

    sv_test = sv.predict(X_test)

    ada_test = ada.predict(X_test)

    gbcm_test = gbcm.predict(X_test)



    y_true = y_test



    for k in range(0, len(test_index)):

        predictions[test_index[k]][0] = rf_test[k]

        predictions[test_index[k]][1] = sv_test[k]

        predictions[test_index[k]][2] = ada_test[k]

        predictions[test_index[k]][3] = gbcm_test[k]

        predictions[test_index[k]][4] = kaanPredictions[k]['class_ids'][0]

 

#stacked ensemble model with best SV, RF, LOGREG, ADA, and Kaan's model

acc = np.zeros(10)

j = 0

for train_index, test_index in skf.split(predictions, targets):

    logreg_stack = linear_model.LogisticRegression(penalty = 'l2', C=1e2)

    logreg_stack.fit(predictions[train_index], targets[train_index])

    

    y_true, y_pred = targets[test_index], logreg_stack.predict(predictions[test_index])

    

    print('\nFold {} Classification Report'.format(j+1))

    print(classification_report(y_true, y_pred))

    acc[j] = 1.0 * sum(y_pred == y_true) / len(test_index)

    j += 1



print('\nAverage Accuracy of the model:')

print(mean(acc))
sum(predictions[1])
# majority vote ensemble with best models



maj = np.zeros(len(predictions))



for i in range(0, len(maj)):

    maj[i] = sum(predictions[i])

    

maj[maj < 3] = 0

maj[maj >= 3] = 1  



print('\nClassification Report')

print(classification_report(targets, maj))
df_test.info()
test_fare = imputer.fit_transform(df_test[['Fare']])

fares_test = pd.DataFrame(test_fare, columns=['Fare'])
unique(df['Title'])
df_test['Title'] = df_test.Name.map(prefix)

unique(df_test['Title'])


df_test.loc[(df_test.Title == 'Rev') |

            (df_test.Title == 'Dr') |

            (df_test.Title == 'Col'), 'Title']='Sir'



df_test.loc[(df_test.Title == 'Dona'), 'Title']='Lady'



title_test = lb.fit_transform(df_test['Title'])

title_columns_test = lb.classes_

titles_test = pd.DataFrame(data=title_test, columns=title_columns_test)
df_test['Family_size'] = df_test.SibSp + df_test.Parch + 1 
pclass_test = lb.fit_transform(df_test['Pclass'])

pclass_columns_test = ['Class1', 'Class2', 'Class3']

pclasses_test = pd.DataFrame(data=pclass_test, columns=pclass_columns_test)
sex_test = lb.fit_transform(df_test['Sex'])

genders_test = pd.DataFrame(data=sex_test, columns=['Sex_transform'])
embark_test = lb.fit_transform(df_test['Embarked'])

embark_columns = lb.classes_

embarked_test = pd.DataFrame(data=embark_test, columns=embark_columns)
print(df_test.groupby(['Pclass','Title'])['Age'].median())

df.groupby(['Pclass','Title'])['Age'].median()
group_train_test_age = pd.concat([df[['Age','Pclass', 'Title']], df_test[['Age','Pclass', 'Title']]], ignore_index = True)

group_train_test_age.groupby(['Pclass','Title'])['Age'].median()
median_age = group_train_test_age.median()

df_test.loc[(df_test.Title == 'Ms'), 'Age'] = median_age[0]
df_test['Age'] = group_train_test_age.groupby(['Pclass','Title'])['Age'].transform(lambda x:x.fillna(x.median()))
numerical_attributes=['SibSp', 'Parch', 'Age', 'Family_size']

data_test = pd.concat([df_test[numerical_attributes], fares_test, embarked_test, titles_test, pclasses_test, genders_test], axis=1)

numerical_attributes=['SibSp', 'Parch', 'Age', 'Family_size', 'Fare']

# scale the numerical attributes

data_test[numerical_attributes] = scaler.fit_transform(data_test[numerical_attributes])
df_test.info()
data_test.drop('Ms', axis=1, inplace=True)
#Make the final predictions





#fit the models with all of the test data

rf.fit(data, targets)



sv.fit(data, targets)



ada.fit(data, targets)

     

gbcm.fit(data, targets)

    

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,

                                                  hidden_units=[12, 3],

                                                  n_classes=2,

                                                  optimizer=tf.train.AdamOptimizer())

    

train_input_fn = tf.estimator.inputs.numpy_input_fn(

              x={"x": np.array(data)},

              y=np.array(targets),

              num_epochs=None,

              shuffle=True)



classifier.train(input_fn=train_input_fn, steps=10000)





#get predictions for the Test data

rf_test = rf.predict(data_test)

sv_test = sv.predict(data_test)

ada_test = ada.predict(data_test)

gbcm_test = gbcm.predict(data_test)







predict_input_fn = tf.estimator.inputs.numpy_input_fn(

            x={"x": np.array(data_test)},

            num_epochs=1,

            shuffle=False)



kaanPredictions = list(classifier.predict(input_fn=predict_input_fn))

kaanPredictions=np.array(kaanPredictions)

finalPreds = []

for i in range(len(kaanPredictions)):

    temp = kaanPredictions[i]['class_ids'][0]

    finalPreds.append(temp)



model_predictions = np.zeros((len(data_test), 5))



model_predictions[:,0] = rf_test

model_predictions[:,1] = sv_test

model_predictions[:,2] = gbcm_test

model_predictions[:,3] = ada_test

model_predictions[:,4] = finalPreds



#use the stack_model fitted with all of the training data

logreg_final = linear_model.LogisticRegression(penalty = 'l2', C=1e2)

logreg_final.fit(predictions, targets)



test_preds = logreg_final.predict(model_predictions)
#export the data

submission = pd.DataFrame({'PassengerId': df_test['PassengerId'] , 'Survived': test_preds})

submission.to_csv('titanic.csv', index=False)