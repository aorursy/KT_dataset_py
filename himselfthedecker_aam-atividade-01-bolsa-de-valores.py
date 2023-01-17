import pandas as pd

import numpy as np

from sklearn import metrics, preprocessing, linear_model



# deixa fixo o fator de aleatoriedade

np.random.seed(0)
# carrega os dados

data = pd.read_csv('../input/training_data.csv', header=0)



data = data.drop([

    'id', 'era', 'data_type', 'target_charles', 'target_elizabeth',

    'target_jordan', 'target_ken', 'target_frank', 'target_hillary'],axis=1)
# transforma o CSV em numpy

features = [f for f in list(data) if "feature" in f]

X = data[features]

y = data['target_bernie']
# exibe quantidade de amostras e atributos

print(X.shape)

print(y.shape)
# a partir daqui é com você...

# 1) separe 30% dos dados para teste e utilize os outros 70% como achar melhor

# 2) lembre-se de aplicar os conceitos vistos em aula
data.info()
data.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = X_train, X_test, y_train, y_test
print('Using the entire dataset for baseline definition')

print('Using no Feature Extraction and simple Logistic Regression. Mostly a baseline suggested by Numer.ai\' guide')

print('-------')



from sklearn import metrics, preprocessing, linear_model

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



model = linear_model.LogisticRegression(n_jobs=-1, solver='lbfgs')

model.fit(X_train, y_train)

y_pred_class = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)



print(f'With Logistic Regression - Accuracy: {accuracy_score(y_test, y_pred_class) * 100}% | Log Loss: {log_loss(y_test, y_pred_proba, eps=1e-15)}')
# Define a quantidade de features que queremos extrair do processo de seleção

relevant_columns_amount = 4



from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



print('Since we\'ll be using PCA for extraction (and it\'s quite efficient at it),we\'ll use the entire datasets from this round of modelling')

print(f'Using PCA for Feature Extraction, with the {relevant_columns_amount} most relevant features as output')

print('-------')



# Extração de Features com PCA

pca = PCA(n_components=relevant_columns_amount)

fit = pca.fit(X_train)



# Efetuamos o fit() com dados de treino e os tranform() diretamente nas duas massas de dados que geramos durante a separação (X_train e X_test)

X_train = pd.DataFrame(pca.transform(X_train))

X_test = pd.DataFrame(pca.transform(X_test))



for i in np.arange(11,21,2):

    # treinando o modelo 

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)



    # testando o modelo contra os 30% de testes que foram removidos anteriormente

    y_pred_class = knn.predict(X_test)

    y_pred_proba = knn.predict_proba(X_test)

    print(f'KNN with k={i} - Accuracy: {accuracy_score(y_test, y_pred_class) * 100}% | Log Loss: {log_loss(y_test, y_pred_proba, eps=1e-15)}')

    print('-------')
# Thanks to our previous usage of PCA on the train/validation/test data, we'll need to regenerate the datasets, based on the same random_state for ensured replication of the experiment

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)



# How many features we want our RFE to output for further use in the algorithms

relevant_columns_amount = 4



print(f'From now on, we\'ll use a reduced version of the datasets, with only {relevant_columns_amount} features instead of the whole 50. Mostly for optimization reasons, since time is still a constraint.')

print(f'Using PCA for discarting half of the features and following up with RFE with Logistic Regression for Feature Extraction, with the {relevant_columns_amount} most relevant features as output')

print('-------')



from sklearn.decomposition import PCA

pca = PCA(n_components=25)

fit = pca.fit(X_train)



X_train = pd.DataFrame(pca.transform(X_train))

X_test = pd.DataFrame(pca.transform(X_test))



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



model = LogisticRegression(solver='lbfgs')

rfe = RFE(model, relevant_columns_amount)

fit = rfe.fit(X_train, y_train)



X_train = pd.DataFrame(rfe.transform(X_train))

X_test = pd.DataFrame(rfe.transform(X_test))



y_train = np.ravel(pd.DataFrame(y_train).reset_index(drop=True))

y_test = np.ravel(pd.DataFrame(y_test).reset_index(drop=True))



print(f'Will use KNN with k ranging from 9 to 21, both for validation and testing datasets')

print('-------')

for i in np.arange(9,23,2):

    # treinando o modelo 

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)



    # testando o modelo contra os 30% de testes que foram removidos anteriormente

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import log_loss

    y_pred_class = knn.predict(X_test)

    y_pred_proba = knn.predict_proba(X_test)

    print(f'KNN with k={i} - Accuracy: {accuracy_score(y_test, y_pred_class) * 100}% | Log Loss: {log_loss(y_test, y_pred_proba, eps=1e-15)}')

    print('-------')
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit



svm = LinearSVC()

clf = CalibratedClassifierCV(svm, cv=ShuffleSplit())

clf.fit(X_train, y_train)



y_pred_class = clf.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)



print(f'Linear Support Vector Classifier - Accuracy: {accuracy_score(y_test, y_pred_class) * 100}% | Log Loss: {log_loss(y_test, y_pred_proba, eps=1e-15)}')
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



for i in np.arange(1,5,1):

    # treinando o modelo 

    clf = DecisionTreeClassifier(max_depth=i)

    clf.fit(X_train_raw,y_train_raw)



    # testando o modelo contra os 30% de testes que foram removidos anteriormente

    from sklearn.metrics import accuracy_score

    y_pred_class = clf.predict(X_test_raw)

    y_pred_proba = clf.predict_proba(X_test_raw)

    print(f'Decision Tree with max_depth={i} - Accuracy: {round(accuracy_score(y_test_raw, y_pred_class) * 100,6)}% | Log Loss: {round(log_loss(y_test_raw, y_pred_proba, eps=1e-15),6)}')
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



for i in np.arange(20,31,1):

    # treinando o modelo 

    clf = RandomForestClassifier(n_estimators=i, max_depth=6, random_state=0)

    clf.fit(X_train_raw,y_train_raw)



    # testando o modelo contra os 30% de testes que foram removidos anteriormente

    y_pred_class = clf.predict(X_test_raw)

    y_pred_proba = clf.predict_proba(X_test_raw)

    print(f'Random Forest with n_estimators={i} and max_depth=6 - Accuracy: {round(accuracy_score(y_test_raw, y_pred_class) * 100,6)}% | Log Loss: {round(log_loss(y_test_raw, y_pred_proba, eps=1e-15),6)}')
## With Logistic Regression - Accuracy: 51.7650179% | Log Loss: 0.692209



# 0.691975 0.692793
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



# treinando o modelo 

clf = SGDClassifier(loss='modified_huber', shuffle=True, random_state=0, max_iter=1000, tol=1e-3)

clf.fit(X_train_raw,y_train_raw)



# testando o modelo contra os 30% de testes que foram removidos anteriormente

y_pred_class = clf.predict(X_test_raw)

y_pred_proba = clf.predict_proba(X_test_raw)

print(f'Stochastic Gradient Descent with loss=modified_huber - Accuracy: {accuracy_score(y_test, y_pred_class) * 100}%  | Log Loss: {log_loss(y_test, y_pred_proba, eps=1e-15)}')
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss



print('Nosso Random Forest terá 29 estimadores e utiliza 6 níveis de profundidade, servindo como base para nosso estudo final.')

print('Alias, nossa base é de: Accuracy: 51.765018% | Log Loss: 0.692209')



clf = RandomForestClassifier(n_estimators=29, max_depth=6, random_state=0)

clf.fit(X_train_raw,y_train_raw)



# testando o modelo contra os 30% de testes que foram removidos anteriormente

y_pred_class = clf.predict(X_test_raw)

y_pred_proba = clf.predict_proba(X_test_raw)

print(f'Random Forest with n_estimators=29 and max_depth=6 - Accuracy: {round(accuracy_score(y_test_raw, y_pred_class) * 100,6)}% | Log Loss: {round(log_loss(y_test_raw, y_pred_proba, eps=1e-15),6)}')
predictionResult = pd.DataFrame(y_pred_class)

predictedProbability = pd.DataFrame(y_pred_proba)

predictionResult.rename(index=str, columns={0: "predicted_class"}, inplace=True)

predictionResult.index = X_test_raw.index

predictedProbability.index = X_test_raw.index

predictionResult['actual_class'] = pd.DataFrame(y_test_raw)

predictionResult['probability_of_target'] = predictedProbability[1]

predictionResult = predictionResult[['actual_class', 'predicted_class','probability_of_target']]

matches = len(predictionResult[(predictionResult.actual_class == predictionResult.predicted_class)])



print(f'Dos {len(predictionResult)} valores previstos, {matches} estão corretos, totalizando {round((matches / len(predictionResult)) * 100, 6)}% de acurácia')

display(predictionResult)