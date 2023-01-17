import pandas as pd

import numpy as np



from sklearn.svm import SVC

from sklearn import neighbors

from sklearn.ensemble import RandomForestClassifier



import sklearn as sk

from sklearn.model_selection import cross_validate



from sklearn.preprocessing import StandardScaler 

from sklearn.pipeline import make_pipeline
df_treino = pd.read_csv('../input/titanic/train.csv')

df_teste = pd.read_csv('../input/titanic/test.csv')
df_treino.shape
df_teste.shape
df_treino.head()
df_treino.dtypes
df_treino.describe()
df_treino.isnull().sum()
df_treino['Survived'].value_counts()
df_survived = df_treino.groupby(['Survived']).count()



df_survived = df_survived[['PassengerId']]



df_survived.plot.bar()
df_treino['Pclass'].value_counts()
df_class = df_treino.groupby(['Pclass']).count()



df_class = df_class[['PassengerId']]



df_class.plot.bar()
df_treino['Name'].value_counts()
df_treino['Sex'].value_counts()
df_sex = df_treino.groupby(['Sex']).count()



df_sex = df_sex[['PassengerId']]



df_sex.plot.bar()
df_treino['Age'].value_counts()
df_treino.hist(column='Age')
df_treino.hist(column='SibSp')
df_treino.hist(column='Parch')
df_treino.hist(column='Fare')
df_treino['Cabin'].value_counts()
df_cabin = df_treino.groupby(['Cabin']).count()

df_cabin
df_treino['Ticket'].value_counts()
df_ticket = df_treino.groupby(['Ticket']).count()

df_ticket
df_treino['Embarked'].value_counts()
df_embarked = df_treino.groupby(['Embarked']).count()



df_embarked = df_embarked[['PassengerId']]



df_embarked.plot.bar()
#Cria um data frame apenas com os passageiros que sobreviveram

df_survived = df_treino[df_treino['Survived']==1]



#Cria um data frame apenas com os passageiros que não sobreviveram

df_nsurvived = df_treino[df_treino['Survived'] ==0]

#Agrupa os sobreviventes e os não sobreviventes pelo atributo 'Pclass'



df_class_survived = df_survived.groupby(['Pclass']).count()



df_class_nsurvived = df_nsurvived.groupby(['Pclass']).count()
df_class_survived = df_class_survived[['PassengerId']]



df_class_nsurvived = df_class_nsurvived[['PassengerId']]
index = df_treino['Pclass'].unique()



df = pd.DataFrame({'Survived': df_class_survived['PassengerId'], 'nSurvived': df_class_nsurvived['PassengerId']}, index=index)

df.plot.bar(rot=0)
df_sex_survived = df_survived.groupby(['Sex']).count()



df_sex_nsurvived = df_nsurvived.groupby(['Sex']).count()
df_sex_survived = df_sex_survived[['PassengerId']]



df_sex_nsurvived = df_sex_nsurvived[['PassengerId']]
index = df_treino['Sex'].unique()



df = pd.DataFrame({'Survived': df_sex_survived['PassengerId'], 'nSurvived': df_sex_nsurvived['PassengerId']}, index=index)

df.plot.bar(rot=0)
df_sibsp_survived = df_survived.groupby(['SibSp']).count()



df_sibsp_nsurvived = df_nsurvived.groupby(['SibSp']).count()
df_sibsp_survived = df_sibsp_survived[['PassengerId']]



df_sibsp_nsurvived = df_sibsp_nsurvived[['PassengerId']]
index = df_treino['SibSp'].unique()



df = pd.DataFrame({'Survived': df_sibsp_survived['PassengerId'], 'nSurvived': df_sibsp_nsurvived['PassengerId']}, index=index)

df.plot.bar(rot=0)
df_parch_survived = df_survived.groupby(['Parch']).count()



df_parch_nsurvived = df_nsurvived.groupby(['Parch']).count()
df_parch_survived = df_parch_survived[['PassengerId']]



df_parch_nsurvived = df_parch_nsurvived[['PassengerId']]
index = df_treino['Parch'].unique()



df = pd.DataFrame({'Survived': df_parch_survived['PassengerId'], 'nSurvived': df_parch_nsurvived['PassengerId']}, index=index)

df.plot.bar(rot=0)
df_embarked_survived = df_survived.groupby(['Embarked']).count()



df_embarked_nsurvived = df_nsurvived.groupby(['Embarked']).count()
df_embarked_survived = df_embarked_survived[['PassengerId']]



df_embarked_nsurvived = df_embarked_nsurvived[['PassengerId']]
index = df_treino['Embarked'].unique()



df = pd.DataFrame({'Survived': df_embarked_survived['PassengerId'], 'nSurvived': df_embarked_nsurvived['PassengerId']}, index=index)

df.plot.bar(rot=0)
df_treino = df_treino.drop(columns=['PassengerId', 'Name'])

df_treino.head()
df_teste = df_teste.drop(columns=['Name'])

df_teste.head()
df_treino.isnull().sum()
df_teste.isnull().sum()
idade_media_treino = df_treino['Age'].mean()



df_treino.update(df_treino['Age'].fillna(idade_media_treino))



idade_media_teste = df_teste['Age'].mean()



df_teste.update(df_teste['Age'].fillna(idade_media_teste))
df_treino.update(df_treino['Cabin'].fillna('Sem valor'))



df_teste.update(df_teste['Cabin'].fillna('Sem valor'))
df_treino = df_treino.dropna(subset=['Embarked'])

df_treino.isnull().sum()

media_fare = df_teste['Fare'].mean()



df_teste.update(df_teste['Fare'].fillna(media_fare))



df_teste.isnull().sum()
df_treino['Sex'] = df_treino['Sex'].replace(['male'], 0)

df_treino['Sex'] = df_treino['Sex'].replace(['female'], 1)



df_treino['Sex'].value_counts()
df_teste['Sex'] = df_teste['Sex'].replace(['male'], 0)

df_teste['Sex'] = df_teste['Sex'].replace(['female'], 1)



df_teste['Sex'].value_counts()
df_treino['Embarked'] = df_treino['Embarked'].replace(['S'], 0)

df_treino['Embarked'] = df_treino['Embarked'].replace(['C'], 1)

df_treino['Embarked'] = df_treino['Embarked'].replace(['Q'], 2)



df_treino['Embarked'].value_counts()
df_teste['Embarked'] = df_teste['Embarked'].replace(['S'], 0)

df_teste['Embarked'] = df_teste['Embarked'].replace(['C'], 1)

df_teste['Embarked'] = df_teste['Embarked'].replace(['Q'], 2)



df_teste['Embarked'].value_counts()
from pandas.plotting import parallel_coordinates



parallel_coordinates(df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']], class_column='Survived')
df_treino.plot(kind='scatter', x='Age', y='Fare', c='Survived', colormap='viridis');
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=50)

clf = clf.fit(df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df_treino[['Survived']])



#o atributo 'feature_importances_' é uma lista com grau de contribuição de cada elemento

importance = clf.feature_importances_

importance
from sklearn.feature_selection import SelectFromModel



model = SelectFromModel(clf, prefit=True)

dados_novo = model.transform(df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])



n_attrs = dados_novo.shape[1]



#ordena por importância

idx_most_important = importance.argsort()[-n_attrs:]

print(idx_most_important)
clf_rf = RandomForestClassifier(n_estimators=50)



clf_svm = make_pipeline(StandardScaler(), SVC())



clf_knn = make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier(5))
scoring_list = ['accuracy', 'recall_macro']



scores_svm = cross_validate(clf_svm, df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df_treino[['Survived']], cv=4, scoring=scoring_list)

scores_knn = cross_validate(clf_knn, df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df_treino[['Survived']], cv=4, scoring=scoring_list)

scores_rf = cross_validate(clf_rf, df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df_treino[['Survived']], cv=4, scoring=scoring_list)



avg_scores_svm = []

avg_scores_knn = []

avg_scores_rf = []



for score in scoring_list:

    avg_scores_svm.append(scores_svm['test_'+score].mean())

    avg_scores_knn.append(scores_knn['test_'+score].mean())

    avg_scores_rf.append(scores_rf['test_'+score].mean())



print('SVM - ', avg_scores_svm)

print('KNN - ', avg_scores_knn)

print('RF - ', avg_scores_rf)
scoring_list = ['accuracy', 'recall_macro']



scores_svm = cross_validate(clf_svm, df_treino[['Sex', 'Age', 'Fare']], df_treino[['Survived']], cv=4, scoring=scoring_list)

scores_knn = cross_validate(clf_knn, df_treino[['Sex', 'Age', 'Fare']], df_treino[['Survived']], cv=4, scoring=scoring_list)

scores_rf = cross_validate(clf_rf, df_treino[['Sex', 'Age', 'Fare',]], df_treino[['Survived']], cv=4, scoring=scoring_list)



avg_scores_svm = []

avg_scores_knn = []

avg_scores_rf = []



for score in scoring_list:

    avg_scores_svm.append(scores_svm['test_'+score].mean())

    avg_scores_knn.append(scores_knn['test_'+score].mean())

    avg_scores_rf.append(scores_rf['test_'+score].mean())



print('SVM - ', avg_scores_svm)

print('KNN - ', avg_scores_knn)

print('RF - ', avg_scores_rf)
clf_svm.fit(df_treino[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df_treino[['Survived']])
clf_svm.predict(df_teste[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
df_result = pd.DataFrame()

df_result['PassengerId'] = df_teste['PassengerId']

df_result['Survived'] = clf_svm.predict(df_teste[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])

df_result