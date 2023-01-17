import pandas as pd

import numpy as np

import random as rnd

import math



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier



training_features = pd.read_csv('../input/train.csv')

testing_features = pd.read_csv('../input/test.csv')
training_features.info()
training_features.describe(include='all')
def chartFeatures(dataFrame, features, group, showLog=False):

  chart = dataFrame[features].groupby(group).agg([np.sum, np.size, np.mean])

  if showLog:

    print(chart)

    print('\n\n')



  return chart



print('Mean Survival Rate')

print(training_features['Survived'].mean())



featureCharts = [

  chartFeatures(training_features, features=['Sex', 'Survived'], group=['Sex']),

  chartFeatures(training_features, features=['Pclass', 'Survived'], group=['Pclass']),

  chartFeatures(training_features, features=['SibSp', 'Survived'], group=['SibSp']),

  chartFeatures(training_features, features=['Parch', 'Survived'], group=['Parch']),

  chartFeatures(training_features, features=['Embarked', 'Survived'], group=['Embarked'])

]



# Pclass Feature

featureCharts[1] = featureCharts[1].rename(index={1: '1st Class', 2: '2nd Class', 3: '3rd Class'})



# SibSp Feature

featureCharts[2] = featureCharts[2].rename(index={0: '0-SibSp', 1: '1-SibSp', 2: '2-SibSp', 3: '3-SibSp', 4: '4-SibSp', 5: '5-SibSp', 8: '8-SibSp'})



# Parch Feature

featureCharts[3] = featureCharts[3].rename(index={0: '0-Parch', 1: '1-Parch', 2: '2-Parch', 3: '3-Parch', 4: '4-Parch', 5: '5-Parch', 6: '6-Parch'})



# Embarked Feature

featureCharts[4] = featureCharts[4].rename(index={'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})



print(pd.concat(featureCharts).sort_values(by=[('Survived', 'mean')], ascending=False))
def preprocessSex(df, printLog=False):

  df['Sex'] = df['Sex'].map( { 'female': 0, 'male': 1 }).astype(int)



  if printLog:

    print(df[['Sex', 'Survived']].groupby(['Sex']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  return df



training_features = preprocessSex(training_features, printLog=True)

testing_features = preprocessSex(testing_features)
def preprocessSexClass(df, printLog=False):

  df['SexClass'] = (df['Sex'] * 3) + df['Pclass']

  df['SexClass'] = df['SexClass'].astype(int)



  if printLog:

    print(df[['SexClass', 'Survived']].groupby(['SexClass']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  return df



training_features = preprocessSexClass(training_features, printLog=True)

testing_features = preprocessSexClass(testing_features)
g = sns.FacetGrid(training_features, col='Survived')

g.map(plt.hist, 'Age', bins=20)
def fillAge(df):

  sexClassMean = np.zeros((6))

  for i in range(0,6):

    temp_df = df[df['SexClass'] == i+1]['Age'].dropna()

    sexClassMean[i] = int(temp_df.mean())



  for i in range(0, 6):

    df.loc[ ((df.Age.isnull()) & (df.SexClass == (i+1))), 'Age'] = sexClassMean[i]

  df['Age'] = df['Age'].astype(int)



  return df



def preprocessAge(df, printLog=False):

  df = fillAge(df)

  freq = training_features['Age'].mode()[0]

  size = training_features[training_features['Age'] == freq]['Age'].count()

  cutSize = math.floor(training_features['Age'].count() / size) + 1



  df['AgeRange'] = pd.cut(df['Age'], cutSize)

  df['AgeFreq'] = pd.qcut(df['Age'], cutSize)



  if printLog:

    print(df[['AgeRange', 'Survived']].groupby(['AgeRange']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))

    print(df[['AgeFreq', 'Survived']].groupby(['AgeFreq']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  catlen = len(df['AgeRange'].cat.categories)

  df['AgeRange'].cat.categories = [str(i) for i in range(0, catlen)]

  df['AgeRange'] = df['AgeRange'].astype(int)



  catlen = len(df['AgeFreq'].cat.categories)

  df['AgeFreq'].cat.categories = [str(i) for i in range(0, catlen)]

  df['AgeFreq'] = df['AgeFreq'].astype(int)



  if printLog:

    print(df[['AgeRange', 'Survived']].groupby(['AgeRange']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))

    print(df[['AgeFreq', 'Survived']].groupby(['AgeFreq']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  return df



training_features = preprocessAge(training_features, printLog=True)

testing_features = preprocessAge(testing_features)
def fillFare(df):

  sexClassMean = np.zeros((6))

  for i in range(0,6):

    temp_df = df[df['SexClass'] == i+1]['Fare'].dropna()

    sexClassMean[i] = int(temp_df.mean())



  for i in range(0, 6):

    df.loc[ ((df.Fare.isnull()) & (df.SexClass == (i+1))), 'Fare'] = sexClassMean[i]

  df['Fare'] = df['Fare'].astype(int)



  return df



def preprocessFare(df, printLog=False):

  df = fillFare(df)

  freq = training_features['Fare'].mode()[0]

  size = training_features[training_features['Fare'] == freq]['Fare'].count()

  cutSize = math.floor(training_features['Fare'].count() / size) + 1



  df['FareRange'] = pd.cut(df['Fare'], cutSize)

  df['FareFreq'] = pd.qcut(df['Fare'], cutSize)



  if printLog:

    print(df[['FareRange', 'Survived']].groupby(['FareRange']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))

    print(df[['FareFreq', 'Survived']].groupby(['FareFreq']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  catlen = len(df['FareRange'].cat.categories)

  df['FareRange'].cat.categories = [str(i) for i in range(0, catlen)]

  df['FareRange'] = df['FareRange'].astype(int)



  catlen = len(df['FareFreq'].cat.categories)

  df['FareFreq'].cat.categories = [str(i) for i in range(0, catlen)]

  df['FareFreq'] = df['FareFreq'].astype(int)



  if printLog:

    print(df[['FareRange', 'Survived']].groupby(['FareRange']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))

    print(df[['FareFreq', 'Survived']].groupby(['FareFreq']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  return df



training_features = preprocessFare(training_features, printLog=True)

testing_features = preprocessFare(testing_features)
def preprocessSibSp(df, printLog=False):



  if printLog:

    print(df[['SibSp', 'Survived']].groupby(['SibSp']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  return df



training_features = preprocessSibSp(training_features, printLog=True)

testing_features = preprocessSibSp(testing_features)
def preprocessParch(df, printLog=False):



  if printLog:

    print(df[['Parch', 'Survived']].groupby(['Parch']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))

    

  return df



training_features = preprocessParch(training_features, printLog=True)

testing_features = preprocessParch(testing_features)
def preprocessFamily(df, printLog=False):

  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

  df['FamilySize'] = df['FamilySize'].astype(int)



  if printLog:

    print(df[['FamilySize', 'Survived']].groupby(['FamilySize']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))



  return df



training_features = preprocessFamily(training_features, printLog=True)

testing_features = preprocessFamily(testing_features)
def preprocessEmbark(df, printLog=False):

  df['Embarked'] = df['Embarked'].fillna(df.Embarked.dropna().mode()[0])

  df['Embarked'] = df['Embarked'].map( { 'C': 0, 'Q': 1, 'S': 2 }).astype(int)



  if printLog:

    print(df[['Embarked', 'Survived']].groupby(['Embarked']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))

  return df



training_features = preprocessEmbark(training_features, printLog=True)

testing_features = preprocessEmbark(testing_features)
def preprocessTitle(df, printLog=False):

  df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

  df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')

  df['Title'] = df['Title'].replace('Mme', 'Mrs')



  df['Title'] = df['Title'].replace(to_replace='^[^M].*|Maj.*', regex=True, value='Other')

  df['Title'] = df['Title'].map({ 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4, 'Mr': 5 })

  df['Title'] = df['Title'].astype(int)



  if printLog:

    print(df[['Title', 'Survived']].groupby(['Title']).agg([np.sum, np.size, np.mean]).sort_values(by=[('Survived', 'mean')], ascending=False))





  return df



training_features = preprocessTitle(training_features, printLog=True)

testing_features = preprocessTitle(testing_features)
def preprocessDrops(df, printLog=False):

  df = df.drop(['Name', 'Ticket', 'Cabin', 'AgeRange', 'FareRange', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Pclass'], axis=1)



  if printLog:

    print(df.columns.values)

    

  return df



training_features = preprocessDrops(training_features, printLog=True)

testing_features = preprocessDrops(testing_features)
def logisticRegressionModel(training_features):

  X_train = training_features.drop(['Survived', 'PassengerId'], axis=1)

  Y_train = training_features['Survived']



  

  logreg = LogisticRegression()

  logreg.fit(X_train, Y_train)

  acc = round(logreg.score(X_train, Y_train) * 100, 2)



  # print(train_df.columns)

  # print(train_df.columns.delete([0, 1]))

  coeff_df = pd.DataFrame(training_features.columns.delete([0, 1]))

  # print(coeff_df)

  coeff_df.columns = ['Feature']

  coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

  # print(logreg.coef_)



  print('\nLogistic Regression')

  print(coeff_df.sort_values(by='Correlation', ascending=False))

  print(acc)



  return

logisticRegressionModel(training_features)
def trainWithModel(X_train, Y_train, X_test, model):

  model.fit(X_train, Y_train)

  return [round(model.score(X_train, Y_train) * 100, 2), model.predict(X_test)]



def testModels(dfs, printLog=False):

  training_df = dfs[0]

  testing_df = dfs[1]

  X_train = training_df.drop(['Survived', 'PassengerId'], axis=1)

  Y_train = training_df['Survived']



  X_test = testing_df.drop('PassengerId', axis=1)



  models = list(map(lambda model: trainWithModel(X_train, Y_train, X_test, model), [SVC(), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=4), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=7), GaussianNB(), Perceptron(), LinearSVC(), SGDClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,2,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,8), random_state=1)]))



  modelsAcc = pd.DataFrame({

    'Model': ['Support Vector Machine', 'KNN-3', 'KNN-4', 'KNN-5', 'KNN-7', 'Naive Bayes', 'Perceptron', 'Linear SVC', 'Stochastic Gradient Descent', 'Decision Tree', 'Random Forest', 'Neural Network (12,2)', 'Neural Network (5,5,5)', 'Neural Network (15,2)', 'Neural Network (8,2)', 'Neural Network (8,5)', 'Neural Network (12,2,2)'],

    'Score': list(map(lambda results: results[0], models)),

    })



  bestModelIndex = modelsAcc.sort_values(by='Score', ascending=False).head(1).index.values[0]

  

  if printLog:

    print(modelsAcc.sort_values(by='Score', ascending=False))

    print(bestModelIndex)

    print(modelsAcc['Model'][bestModelIndex])



  return [models, modelsAcc]



def makeCrossSection(df, crossSize):

  minRange = df['PassengerId'].min()

  maxRange = df['PassengerId'].max() + 1

  selection = rnd.sample(range(minRange, maxRange), crossSize)



  return (df[df['PassengerId'].isin(selection)], df[~df['PassengerId'].isin(selection)].drop(['Survived'], axis=1))









fullValidation = testModels([training_features, testing_features], printLog=True)



crossValidations = [

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True),

  testModels(makeCrossSection(training_features, 300), printLog=True)

]



#print(pd.concat(crossValidations[0]).sort_values(by='Score', ascending=False))

#print(crossValidations[1][0])#.sort_values(by='Score', ascending=False))



for i in range(0, len(fullValidation[0])):

  submission = pd.DataFrame({

    'PassengerId': testing_features['PassengerId'],

    'Survived': fullValidation[0][i][1]

  })



  # print(submission.shape)

  submission.to_csv('submission-model' + str(i) + '.csv', index=False)