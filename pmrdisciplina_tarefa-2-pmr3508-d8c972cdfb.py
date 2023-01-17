import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import numpy as np
from os import listdir
listdir("../input/tarefa2")
train = pd.read_csv("../input/tarefa2/train_data.csv")
train.head()
train.shape #retorna o formato do dataset
train.nunique().sort_values()[0:10] #retorna o numero de valores únicos da mesma coluna em ordem crescente
train.mean().sort_values()[0:10] #retorna a media dos valores das colunas em ordem crescente
train['ham'] = train['ham'].astype(str)
spam = train[train['ham'] == 'False']
ham = train[train['ham'] == 'True']

#troco true e false por valores binarios
train = train.replace(['True', 'False'], [1, 0])
#obtenção das médias das frequências de cada palavra por categoria (spam ou não spam)
spam_mean = []
ham_mean = []
for item in spam.mean():
    spam_mean.append(item)
for item in ham.mean():
    ham_mean.append(item)

print("Média de frequência das 10 primeiras palavras para spam:", spam_mean[0:10])
print("\nMédia de frequência das 10 primeiras palavras para ham:", ham_mean[0:10])
plt.plot(spam_mean[0:-5], color = 'r', label = 'spam')
plt.plot(ham_mean[0:-5], color = 'b', label = 'ham')
plt.title('Frequencia das palavras')

red = mpatches.Patch(color='red', label='Spam')
blue = mpatches.Patch(color = 'blue', label = 'Ham')
plt.legend(handles=[red,blue])

plt.show()
plt.hist(spam_mean[0:-5], bins = 10, color = 'r')
plt.hist(ham_mean[0:-5], bins = 10, color = 'b')
plt.title('Histograma das frequências')

red = mpatches.Patch(color='red', label='Spam')
blue = mpatches.Patch(color = 'blue', label = 'Ham')
plt.legend(handles=[red,blue])

plt.show()
groupby_ham = train.groupby(['ham']) #criação de um objeto groupby a partir das colunas do dataframe
#permite a visualização de todas as colunas do meu dataset no jupyter
from IPython.display import display
pd.options.display.max_columns = None
display(groupby_ham.mean())
corr = train.corr()
corr.head()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool)) #correlação entre todas as variáveis
plt.title("Correlacao entre as variaveis")
plt.show()
corr['ham'].plot(color = 'b')
plt.title('Correlação das variáveis com o ham')
plt.show()
corr['ham'].sort_values()
#criando novas colunas no dataframe com esses valores binários
for column in train.columns:
    binary = []
    for item in train[column]:
        if item > train[column].mean():
            binary.append(1)
        else:
            binary.append(0)
    train[column + '_avg'] = binary
train_avg = train[train.columns[57:]] #seleciono apenas as novas colunas criadas com o codigo acima
train_avg = train_avg.drop(labels = ['Id_avg', 'ham_avg'], axis = 1)
train_avg.head()
sns.heatmap(train_avg.corr(), mask=np.zeros_like(train_avg.corr(), dtype=np.bool))
plt.title('Correlação entre as features binárias')
plt.show()
train_avg.corr()['ham'].sort_values()
train_c = train[['ham', 'char_freq_$_avg', 'char_freq_!_avg', 'word_freq_your_avg', 'word_freq_remove_avg', 'word_freq_free_avg', 'word_freq_money_avg', 'word_freq_000_avg', 'word_freq_our_avg', 'word_freq_receive_avg', 'word_freq_all_avg', 'word_freq_you_avg','word_freq_business_avg', 'word_freq_credit_avg', 'word_freq_internet_avg', 'word_freq_you_avg', 'word_freq_your', 'word_freq_000', 'char_freq_$', 'word_freq_remove']]
sns.heatmap(train_c.corr(), mask=np.zeros_like(train_c.corr(), dtype=np.bool))
plt.title('Maiores correlações entre as features binárias')

plt.show()
train_c.corr()['ham'].sort_values()
x_train = train_c.drop(['ham'], axis = 1)
y_train = train_c['ham']
from sklearn.metrics import fbeta_score, make_scorer
f3 = make_scorer(fbeta_score, beta=3)
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score

bernoulli = BernoulliNB(alpha = 0.000001)

print('Score F3 Bernoulli: ', np.mean(cross_val_score(bernoulli, x_train, y_train, scoring = f3, cv = 5)))

bernoulli.fit(x_train, y_train)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(bernoulli, x_train, y_train, cv=5)
confusion = confusion_matrix(y_train, y_pred)
print(confusion)
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train, y_pred)
print('False positive rate:', fpr[1])
print('True positive rate:', tpr[1])
from sklearn.model_selection import train_test_split

x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size = 0.25)
bernoulli.fit(x_train2, y_train2)
pred_prob = bernoulli.predict_proba(x_val)
scores = pd.DataFrame(pred_prob)
scores.columns = ['Prob Spam (0)', 'Prob Ham (1)']
scores['Real'] = y_val
scores = scores.fillna(value = 0)
scores.head()
fpr, tpr, thresholds = roc_curve(y_val, pred_prob[:, 1])
plt.figure()

plt.plot(fpr, tpr, color ='blue')
plt.plot([0,1], [0,1], color = 'red', linestyle = '--')
plt.plot([0,0], [0,1], [1,1], color = 'green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Classificador Naive Bayes')


red = mpatches.Patch(color='red', label='Pior classificador')
green = mpatches.Patch(color = 'green', label = 'Melhor classificador')
blue = mpatches.Patch(color = 'blue', label = 'Classificador obtido')
plt.legend(handles=[red, green, blue])

plt.show()

dist_min_sq = abs(1-tpr[1])*abs(1-tpr[1]) + (fpr[1])*(fpr[1])
print(dist_min_sq)
i_min = 0
for i in range(len(tpr)):
    dist_sq = abs(1-tpr[i])*abs(1-tpr[i]) + fpr[i]*fpr[i]
    if dist_sq < dist_min_sq:
        dist_min_sq = dist_sq
        i_min = i
print("Indice que me dá a menor distância: ", i_min)
print("TPR do índice: ", tpr[i_min])
print("FPR do índice: ", fpr[i_min])
print("Threshold do índice: ", thresholds[i_min])
bernoulli2 = BernoulliNB(class_prior = [1 - thresholds[i_min], thresholds[i_min]])

print('Score F3 Bernoulli: ', np.mean(cross_val_score(bernoulli2, x_train, y_train, scoring = f3, cv = 5)))

bernoulli2.fit(x_train, y_train)
from sklearn.neighbors import KNeighborsClassifier

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    print(i, np.mean(cross_val_score(knn, x_train, y_train, scoring = f3, cv = 5)))
knn = KNeighborsClassifier(n_neighbors = 37, p = 1)
knn.fit(x_train, y_train)
from sklearn.ensemble import VotingClassifier

combine = VotingClassifier(estimators=[('bernoulli2', bernoulli), ('knn', knn)], voting='soft', weights = [1,2])
print(np.mean(cross_val_score(combine, x_train, y_train, scoring = f3, cv = 5)))

combine.fit(x_train, y_train)

test = pd.read_csv('../input/tarefa2/test_features.csv')

for column in test.columns:
    binary = []
    for item in test[column]:
        if item > test[column].mean():
            binary.append(1)
        else:
            binary.append(0)
    test[column + '_avg'] = binary
test = test[['char_freq_$_avg', 'char_freq_!_avg', 'word_freq_your_avg', 'word_freq_remove_avg', 'word_freq_free_avg', 'word_freq_money_avg', 'word_freq_000_avg', 'word_freq_our_avg', 'word_freq_receive_avg', 'word_freq_all_avg', 'word_freq_you_avg','word_freq_business_avg', 'word_freq_credit_avg', 'word_freq_internet_avg', 'word_freq_you_avg', 'word_freq_your', 'word_freq_000', 'char_freq_$', 'word_freq_remove']]
x_train.shape
test.shape
predict_combine = combine.predict(test)
predict_bernoulli = bernoulli.predict(test)
predict_knn = knn.predict(test)
#Submissão do modelo conjunto
sample = pd.read_csv('../input/tarefa2/sample_submission_1.csv')
submit = sample.drop(['ham'], axis = 1)
submit['ham'] = predict_combine
submit = submit.set_index('Id')
submit.head()
submit.to_csv('submission_c4.csv')
#Submissao do modelo bernoulli
sample = pd.read_csv('../input/tarefa2/sample_submission_1.csv')
submit = sample.drop(['ham'], axis = 1)
submit['ham'] = predict_bernoulli
submit = submit.set_index('Id')
submit.head()
submit.to_csv('submission_b4.csv')
#Submissao do modelo KNN
sample = pd.read_csv('../input/tarefa2/sample_submission_1.csv')
submit = sample.drop(['ham'], axis = 1)
submit['ham'] = predict_knn
submit = submit.set_index('Id')
submit.head()
submit.to_csv('submission_k4.csv')