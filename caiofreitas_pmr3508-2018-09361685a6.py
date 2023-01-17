import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import naive_bayes
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
data = pd.read_csv("../input/treino/train_data.csv")
test = pd.read_csv('../input/testee/test_data.csv')
data.shape
trainY = data.ham
trainX = data.drop('ham', axis=1)
trainX [48:59]
data[data.columns[58]]
# Com uma breve análise da variável ham na base de treino, percebe-se que há equilíbrio entre os casos Positivos e Negativos
plt.figure(figsize=(8,6))
plt.bar(["Ham", "Spam"], data.ham.value_counts())
plt.title('Variavel de Interesse')
valores_medios = np.zeros(shape=(48))
for i in range(48):
    name = data.columns[i]
    valores_medios[i] = data[name].mean()
    
desvio_padrao = np.zeros(shape=48)
for i in range(48):
    name = data.columns[i]
    desvio_padrao[i] = np.std(data[name])
    
plt.figure(figsize=(18,10))
plt.title("Valores Médios da Frequência de Cada Palavra")
plt.bar(data.columns[:48], valores_medios)
plt.xticks(data.columns[:48], rotation='vertical')
plt.show()

plt.figure(figsize=(18,10))
plt.title("Desvio Padrão de Cada Palavra")
plt.bar(data.columns[:48], desvio_padrao)
plt.xticks(data.columns[:48], rotation='vertical')
plt.show()
valores_medios = np.zeros(shape=8)
for i in range(49,57):
    name = data.columns[i]
    valores_medios[i-49] = data[name].mean()
    
plt.figure(figsize=(10,6))
plt.title("Valores Médios da Frequência de Caracteres/Sequências")
plt.bar(data.columns[49:57], valores_medios)
plt.xticks(data.columns[49:57], rotation='vertical')
plt.show()
data_spam = data.query('ham == 0')
data_ham = data.query('ham == 1')
pvham = np.mean(data_ham['char_freq_;'])
pvspam = np.mean(data_spam['char_freq_;'])

parham = np.mean(data_ham['char_freq_('])
parspam = np.mean(data_spam['char_freq_('])

brham = np.mean(data_ham['char_freq_['])
brspam = np.mean(data_spam['char_freq_['])
                                   
excham = np.mean(data_ham['char_freq_!'])
excspam = np.mean(data_spam['char_freq_!'])

cifraoham = np.mean(data_ham['char_freq_$'])
cifraospam = np.mean(data_spam['char_freq_$'])
                                   
hashham = np.mean(data_ham['char_freq_#'])
hashspam = np.mean(data_spam['char_freq_#'])
labels = [";", "(", "[", "!", "$", "#"]
heightspam = [pvspam, parspam, brspam, excspam, cifraospam, hashspam]
heightsham = [pvham, parham, brham, excham, cifraoham, hashham]
plt.figure(figsize=(18,10))
plt.bar(labels, heightspam, label='Spam')
plt.bar(labels, heightsham, label='Ham')
plt.title("Valores Médios da Frequência de Caracteres/Sequências", size=20)
plt.legend(prop={'size':20})
plt.show()
gaussiannaive = naive_bayes.GaussianNB()
bernoullinaive = naive_bayes.BernoulliNB()
gaussiannaive.fit(trainX, trainY)
bernoullinaive.fit(trainX, trainY)
# Função para plotar a curva de aprendizado
def plot_learning_curve(estimator, title, X, y):
    ylim=None
    cv=None
    n_jobs=None
    train_sizes=np.linspace(.1, 1.0, 5)
    plt.figure(figsize=(16,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Dados ultilizados", size=16)
    plt.ylabel("Acurácia", size=16)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation Score")

    plt.legend(loc="best")
    plt.show()
plot_learning_curve(bernoullinaive,"Curva de Aprendizado para o Bernoulli Naive Bayes", trainX, trainY)
plot_learning_curve(gaussiannaive,"Curva de Aprendizado para o Gaussian Naive Bayes", trainX, trainY)
bernoulliscores = cross_val_score(bernoullinaive, trainX, trainY, cv=10)
gaussianscores = cross_val_score(gaussiannaive, trainX, trainY, cv=10)
scores = bernoulliscores.mean(), gaussianscores.mean()
plt.figure(figsize=(8,6))
plt.bar(["Bernoulli", "Gaussian"], scores)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
p1 = cross_val_predict(bernoullinaive, trainX, trainY, cv=10, method = 'predict_proba')
fprb, tprb ,thresholds =roc_curve(trainY, p1[:,1])

plt.figure(figsize=(16,8))
plt.plot(fprb,tprb)
plt.plot([0, 1], [0, 1], color='red', linestyle='-.')
plt.xlabel('Especificidade', size=16)
plt.ylabel('Sensividade', size=16)
plt.title('Curva ROC para o Bernoulli Naive Bayes', size=20)
plt.show()

p2 = cross_val_predict(gaussiannaive, trainX, trainY, cv=10, method = 'predict_proba')
fprg, tprg ,thresholds =roc_curve(trainY, p2[:,1])

plt.figure(figsize=(16,8))
plt.plot(fprg,tprg)
plt.plot([0, 1], [0, 1], color='red', linestyle='-.')
plt.xlabel('Especificidade', size=16)
plt.ylabel('Sensividade', size=16)
plt.title('Curva ROC para o Gaussian Naive Bayes', size=20)
plt.show()
scoresb = cross_val_score(bernoullinaive, trainX, trainY, cv=10)
scoresg = cross_val_score(gaussiannaive, trainX, trainY, cv=10)
print("Bernoulli CV Score - ", scoresb.mean())
print("Gaussian CV Score - ", scoresg.mean())
# Cálculo da área de baixo das curvas ROC para comparação dos classificadores:
bernoulli_auc = auc(fprb, tprb)
gaussian_auc = auc(fprg, tprg)
print('Bernoulli - ', bernoulli_auc)
print('Gaussian - ', gaussian_auc)
predictions = bernoullinaive.predict(test)
predict = pd.DataFrame(index = test.Id)
predict["ham"] = predictions
predict.to_csv('predictions.csv')
predict