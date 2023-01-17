import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/datasetsnaive/train_data.csv')
train.shape #(Número de e-mails, número de features)
train.head()
train.isnull().sum()
print("Número de features:" ,len(train.columns))
correlation = train.corr()
correlation = correlation[abs(correlation.ham) > 0.3]
correlation
feat = list(correlation.index.drop('ham'))
feat
X_train = train[feat]
y_train = train.ham
from sklearn import naive_bayes
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_val_score
scores = []
classifier = naive_bayes.GaussianNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
classifier = naive_bayes.ComplementNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
classifier = naive_bayes.MultinomialNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
classifier = naive_bayes.BernoulliNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
plt.bar(['Gaussian', 'Complement', 'Multinomial', 'Bernoulli'], height=[scores[0], scores[1], scores[2], scores[3]])
pd.options.mode.chained_assignment = None
train2 = pd.DataFrame(train)
for i in train2:
    for j in range(train2.shape[0]):
        if train2[i][j] > 0.01:
            train2[i][j] = 1
        else:
            train2[i][j] = 0
feat = list(train2.drop(['ham', 'Id'], axis=1).columns)
X_train = train2[feat]
y_train = train2['ham']
socres = []
classifier = naive_bayes.GaussianNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
classifier = naive_bayes.MultinomialNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
classifier = naive_bayes.ComplementNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
classifier = naive_bayes.BernoulliNB()
score = cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1')
scores.append(score.mean())
print(score.mean())
plt.bar(['Gaussian', 'Complement', 'Multinomial', 'Bernoulli'], height=[scores[0], scores[1], scores[2], scores[3]])
classifier.fit(X_train, y_train)

#Importamos o classificador e o treinamos no dataset de treino.

test = pd.read_csv('../input/datasetsnaive/test_features.csv')
test.shape[0] #número de e-mails

X_test = test[feat]
feat

y_test = classifier.predict(X_test)

#Importamos os dados que queremos classificar entre SPAM e HAM, e o classificamos usando nosso modelo e guardamos as respostas na variável *y_test*.



#No final criamos um dataframe das respostas com o formato para a submissão.

colham = {'ham': y_test}
colid = {'Id': test.Id}

prediction = pd.DataFrame()
prediction['Id'] = test.Id
prediction['ham'] = y_test

prediction.to_csv('predictions.csv', index = False)