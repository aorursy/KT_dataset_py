import numpy as np

import pandas as pd



titan = pd.read_csv('/kaggle/input/titanic/train.csv')



#Droping unused attributes

titan = titan.drop(['Name', 'PassengerId', 'Ticket', 'Fare', 'Cabin'], axis=1)



#droping NAs

titan = titan.dropna(axis=0)



#Setting Targets

titan_labels = titan['Survived']

titan_labels = titan_labels.to_numpy()



#Setting Data

titan_data = titan.drop(['Survived'], axis=1)



#Converting to INT

titan_data['Sex'] = titan_data['Sex'].replace('male', 1)

titan_data['Sex'] = titan_data['Sex'].replace('female', 0)



#Converting Category to INT

titan_data['Embarked'] = titan_data['Embarked'].astype('category')

titan_data['Embarked'] = titan_data['Embarked'].cat.rename_categories(range(1, titan_data['Embarked'].nunique()+1))

titan_data['Embarked'] = titan_data['Embarked'].astype('Int64')



titan_data_numpy = titan_data.to_numpy()



from sklearn.model_selection import train_test_split

dados_titan_treino, dados_titan_teste, rotulos_titan_treino, rotulos_titan_teste = train_test_split(titan_data_numpy, titan_labels, test_size=0.4, stratify=titan_labels)
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



clf_svm = SVC()

clf_rf = RandomForestClassifier()
clf_svm.fit(dados_titan_treino, rotulos_titan_treino)
clf_rf.fit(dados_titan_treino, rotulos_titan_treino)
from sklearn.model_selection import cross_validate



scoring_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']



scores_svm = cross_validate(clf_svm, titan_data, titan_labels, cv=4, scoring=scoring_list)

scores_rf = cross_validate(clf_rf, titan_data, titan_labels, cv=4, scoring=scoring_list)



avg_scores_svm = []

avg_scores_rf = []



for score in scoring_list:

    avg_scores_svm.append(scores_svm['test_'+score].mean())

    avg_scores_rf.append(scores_rf['test_'+score].mean())



print('SVM - ', avg_scores_svm)

print('RF - ', avg_scores_rf)

comparative = pd.DataFrame({'SVM': avg_scores_svm,

                   'RF': avg_scores_rf}, index=scoring_list)

graph_metrics = comparative.plot.bar(rot=0)

from sklearn.feature_selection import SelectFromModel



importance = clf_rf.feature_importances_

model = SelectFromModel(clf_rf, prefit=True)

dados_novo = model.transform(titan_data_numpy)

n_attrs = dados_novo.shape[1]



idx_most_important = importance.argsort()[-n_attrs:]

name_important_attrs = np.array(list(titan_data.columns))[idx_most_important]



print(['Most important attributes:'], name_important_attrs)

titan_newdata = titan_data[list(name_important_attrs)]

titan_newdata_array = titan_newdata.to_numpy()



from sklearn.model_selection import train_test_split

dados_titan_treino, dados_titan_teste, rotulos_titan_treino, rotulos_titan_teste = train_test_split(titan_newdata_array, titan_labels, test_size=0.4, stratify=titan_labels)



from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()

clf_rf.fit(dados_titan_treino, rotulos_titan_treino)



from sklearn.model_selection import cross_validate



scoring_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

scores_rf = cross_validate(clf_rf, titan_data, titan_labels, cv=4, scoring=scoring_list)



avg_scores_rf = []



for score in scoring_list:



    avg_scores_rf.append(scores_rf['test_'+score].mean())



print('RF - ', avg_scores_rf)



comparative = pd.DataFrame({'RF': avg_scores_rf}, index=scoring_list)

comparative.plot.bar(rot=0)





titan_test = pd.read_csv('/kaggle/input/titanic/test.csv')

ids = titan_test['PassengerId']

titan_test = titan_test[list(name_important_attrs)]

titan_test = titan_test.dropna(axis=0)



#Converting to INT

titan_test['Sex'] = titan_test['Sex'].replace('male', 1)

titan_test['Sex'] = titan_test['Sex'].replace('female', 0)



titan_test.head()

resultado = clf_rf.predict(titan_test)

titan_final = titan_test.copy()

titan_final['Predicted'] = resultado

titan_final['Ids'] = ids

titan_final