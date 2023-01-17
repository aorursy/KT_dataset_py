from IPython.display import HTML
HTML('''<script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
Para esconder/mostrar os erros de output do notebook, clique <a href="javascript:code_toggle_err()">aqui</a>.''')
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer, roc_curve, auc
import os
print(os.listdir('../input/pmr-3508-tarefa-2'))
test_d = pd.read_csv('../input/testefeat/test_features.csv',index_col='Id')
train_d = pd.read_csv('../input/train-data/train_data.csv', index_col='Id')
train_d.head()
train_d.info()
test_d.info()
ax = train_d['ham'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Classes'); plt.ylabel('Proporção');
plt.title('Distribuição de Classe dos dados na base treino');
#Função de avaliação entre classificadores para um determinado dataset
def ClassifierScores(data, labels, scorer = None, n_neighbors = 5):

    bnb = BernoulliNB()
    scores = cross_val_score(bnb, data, labels, cv=10, scoring = scorer)
    print('Bernoulli NB')
    print(f'Pontuação (10-fold CV) = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
    if scorer != None:
        scores = cross_val_score(bnb, data, labels, cv=10)
        print(f'Acurácia = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
        print()

#MultinomialNB() não pode receber valores negativos
    if np.sum((data < 0).values.ravel()) > 0:
        print('Dados possuem valores negativos, pulando Multinomial NB')
        print()
    else:
        mnb = MultinomialNB()
        scores = cross_val_score(mnb, data, labels, cv=10, scoring = scorer)
        print('Multinomial NB')
        print(f'Pontuação (10-fold CV) = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
        scores = cross_val_score(mnb, data, labels, cv=10)
        print(f'Acurácia = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
        print()
        
    gnb = GaussianNB()
    fbeta = make_scorer(fbeta_score,beta = 3)
    scores = cross_val_score(gnb, data, labels, cv=10, scoring = scorer)
    print('Gaussian NB')
    print(f'Pontuação (10-fold CV) = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
    if scorer != None:
        scores = cross_val_score(gnb, data, labels, cv=10)
        print(f'Acurácia = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
        print()
    
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, n_jobs = -1)
    scores = cross_val_score(knn, data, labels, cv=10, scoring = scorer)
    print(f'{n_neighbors}NN')
    print(f'Pontuação (10-fold CV) = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
    if scorer != None:
        scores = cross_val_score(knn, data, labels, cv=10)
        print(f'Acurácia = {round(scores.mean(), 4)}, com Desvio Padrão = {round(scores.std(), 4)}')
baseline = pd.DataFrame(np.zeros(train_d.drop('ham',axis=1).shape))
fbeta = make_scorer(fbeta_score, greater_is_better=True, beta = 3)
ClassifierScores(baseline,
                 train_d.loc[:,'ham'],
                 fbeta)
ClassifierScores(train_d.drop('ham', axis = 1),
                 train_d.loc[:,'ham'],
                 fbeta)
test_d['ham'] = np.nan
data = train_d.append(test_d, sort=False)
bin_train1 = pd.DataFrame()
feature = []
bin_train1['ham'] = data['ham']
for c in cols:
    for i, row in data.iterrows():
        if row[c] > 0:
            feature.append(1)
        else:
            feature.append(0)
    bin_train1[c] = feature
    feature.clear()

bin_train1.dropna(inplace = True)
bin_train1.head()
bin_train1.drop(labels = ['capital_run_length_average','capital_run_length_longest','capital_run_length_total'], axis= 1, inplace = True)
ClassifierScores(bin_train1.drop('ham', axis = 1),
                 bin_train1.loc[:,'ham'],
                 fbeta,
                 11)
spam = train_d[train_d['ham'] == 0]
mail = train_d[train_d['ham'] == 1]
cols = list(data.columns)
cols.pop()

spam_means = {col:spam[col].mean() for col in cols}
spam_stds = {col:spam[col].std() for col in cols}
mail_means = {col:mail[col].mean() for col in cols}
mail_stds = {col:mail[col].std() for col in cols}
means = {col:train_d[col].mean() for col in cols}
stds = {col:train_d[col].mean() for col in cols}
plt.figure(figsize = (20, 8))
ax = plt.subplot(111)
x = np.arange(len(cols))
bar_mail_means = ax.bar(x-0.2, list(mail_means.values()),width=0.2,color='b',align='center')
bar_all_means = ax.bar(x, list(means.values()),width=0.2,color='g',align='center')
bar_spam_means = ax.bar(x+0.2, list(spam_means.values()),width=0.2,color='r',align='center')
ax.set_xticks(x+0.2)
ax.set_xticklabels(cols)
ax.legend((bar_mail_means, bar_spam_means, bar_all_means), ('Média para ham', 'Média para spam',  'Média dos dados de treino'))
plt.xticks(rotation=90)

plt.xlabel('Coluna'); plt.ylabel('Valor médio');
plt.title('Comparação entre os valores médios das colunas para cada classe')
plt.show()
plt.figure(figsize = (20, 8))
ax = plt.subplot(111)
x = np.arange(len(cols)-3)
bar_mail_means = ax.bar(x-0.2, list(mail_means.values())[:-3],width=0.2,color='b',align='center')
bar_all_means = ax.bar(x, list(means.values())[:-3],width=0.2,color='g',align='center')
bar_spam_means = ax.bar(x+0.2, list(spam_means.values())[:-3],width=0.2,color='r',align='center')
ax.set_xticks(x+0.2)
ax.set_xticklabels(cols)
ax.legend((bar_mail_means, bar_spam_means, bar_all_means), ('Média para ham', 'Média para spam',  'Média dos dados de treino'))
plt.xticks(rotation=90)

plt.xlabel('Coluna'); plt.ylabel('Valor médio');
plt.title('Comparação entre os valores médios das colunas para cada classe')
plt.show()
plt.figure(figsize = (20, 8))
ax = plt.subplot(111)
x = np.arange(len(cols)-3)
bar_mail_stds = ax.bar(x-0.2, list(mail_stds.values())[:-3],width=0.2,color='b',align='center')
bar_all_stds = ax.bar(x, list(stds.values())[:-3],width=0.2,color='g',align='center')
bar_spam_stds = ax.bar(x+0.2, list(spam_stds.values())[:-3],width=0.2,color='r',align='center')
ax.set_xticks(x+0.2)
ax.set_xticklabels(cols)
ax.legend((bar_mail_stds, bar_spam_stds, bar_all_stds), ('Desvio Padrão para ham', 'Desvio Padrão para spam', 'Desvio Padrão dos dados de treino'))
plt.xticks(rotation=90)

plt.xlabel('Coluna'); plt.ylabel('Desvio Padrão');
plt.title('Comparação entre os valores médios das colunas para cada classe');
plt.show()
fit_norm = data[cols]

for c in cols:
        fit_norm.loc[:,c] = fit_norm.loc[:,c].subtract(means[c]).divide(stds[c])
        
fit_norm['ham'] = data['ham']

fit_norm.head()
train_norm = fit_norm.dropna()
ClassifierScores(train_norm.drop('ham', axis = 1),
                 train_norm.loc[:,'ham'],
                 fbeta,
                 11)
lim_spam = {col:(spam_means[col]-means[col])/stds[col] for col in cols}
lim_mail = {col:(mail_means[col]-means[col])/stds[col] for col in cols}

bin_train3 = pd.DataFrame()
feature = []
bin_train3['ham'] = data['ham']
for c in cols:
    for i, row in fit_norm.iterrows():
        if spam_means[c] > mail_means[c]:
            if row[c] > lim_mail[c]:
                feature.append(1)
            else:
                feature.append(0)
        else:
            if row[c] > lim_spam[c]:
                feature.append(1)
            else:
                feature.append(0)
        
    bin_train3[c] = feature
    feature.clear()

ClassifierScores(bin_train3.dropna().drop('ham', axis = 1),
                 bin_train3.dropna().loc[:,'ham'],
                 fbeta,
                 11)
lim_spam_std = {col:spam_stds[col]/stds[col] for col in cols}
lim_mail_std = {col:mail_stds[col]/stds[col] for col in cols}

bin_train4 = pd.DataFrame()
feature = []
bin_train4['ham'] = data['ham']
for c in cols:
    for i, row in fit_norm.iterrows():
        if spam_means[c] > lim_spam[c]:
            if row[c] > lim_spam[c]-0.2*lim_spam_std[c]:
                feature.append(1)
            else:
                feature.append(0)
        else:
            if abs(row[c]) > lim_mail[c]-0.2*lim_mail_std[c]:
                feature.append(1)
            else:
                feature.append(0)
        
    bin_train4[c] = feature
    feature.clear()

bin_train4.head()
ClassifierScores(bin_train4.dropna().drop('ham', axis = 1),
                 bin_train4.dropna().loc[:,'ham'],
                 fbeta,
                 11)
words = list(data.columns)[:-10]
word_table = pd.DataFrame()
word_table['ham'] = train_d['ham']

count = []
for c in words:
    for i, row in train_d.iterrows():
        if row[c] > 0:
            count.append(1)
        else:
            count.append(0)
    word_table[c] = count
    count.clear()
    
word_table.head()
spamicity = [word_table.loc[word_table['ham']==0].loc[:,word].sum()/word_table.loc[:,word].sum() for word in word_table.columns]
spamicity = pd.Series(spamicity,word_table.columns)
spamicity.drop('ham',inplace = True)
drop = list(spamicity.loc[abs(spamicity - 0.5) < 0.3].keys())
display(drop)
print(f'{len(drop)} a remover')
cols = list(data.drop(drop,axis=1).drop('ham',axis=1).columns)
bin_train6 = pd.DataFrame()
feature = []
bin_train6['ham'] = data['ham']
for c in cols:
    for i, row in fit_norm.iterrows():
        if row[c] > 0:
            feature.append(1)
        else:
            feature.append(0)
        
    bin_train6[c] = feature
    feature.clear()

ClassifierScores(bin_train6.dropna().drop('ham', axis = 1),
                 bin_train6.dropna().loc[:,'ham'],
                 fbeta,
                 11)
bin_train7 = pd.DataFrame()
feature = []
bin_train7['ham'] = data['ham']
for c in cols:
    for i, row in fit_norm.iterrows():
        if spam_means[c] > means[c]:
            if row[c] > lim_spam[c]:
                feature.append(2)
            elif row[c] > 0:
                    feature.append(1)
            else:
                feature.append(0)
        else:
            if row[c] > lim_mail[c]:
                feature.append(2)
            elif row[c] > 0:
                feature.append(1)
            else:
                feature.append(0)
    bin_train7[c] = feature
    feature.clear()

ClassifierScores(bin_train7.dropna().drop('ham', axis = 1),
                 bin_train7.dropna().loc[:,'ham'],
                 fbeta,
                 11)
bin_train8 = pd.DataFrame()
feature = []
bin_train8['ham'] = data['ham']
for c in cols:
    for i, row in fit_norm.iterrows():
        if spam_means[c] > means[c]:
            if row[c] > lim_spam[c]-0.25*lim_spam_std[c]:
                feature.append(2)
            elif row[c] > 0:
                feature.append(1)
            else:
                feature.append(0)
        else:
            if row[c] > lim_mail[c] - 0.25*lim_mail_std[c]:
                feature.append(2)
            elif row[c] > 0:
                feature.append(1)
            else:
                feature.append(0)
    bin_train8[c] = feature
    feature.clear()

ClassifierScores(bin_train8.dropna().drop('ham',axis=1),
                 bin_train8.dropna().loc[:,'ham'],
                 fbeta,
                 11)
X = bin_train8.dropna().drop('ham', axis = 1)
y = bin_train8.dropna().loc[:,'ham']
scores_array = []
for n in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn,
                             X,
                             y,
                             cv=10,
                             scoring = fbeta)
    scores_array.append(scores.mean())
    
plt.plot(range(1,100),scores_array, 'ro')
plt.xlabel('Fbeta'); plt.ylabel('Número de Vizinhos');
plt.title('Escolha de número de vizinhos para treinamento do KNN');
n = np.argmax(scores_array)+1
print(f'Pontuação Máxima:{round(max(scores_array),4)}')
print(f'Número de Vizinhos:{n}')
scores_array = []
priori = np.linspace(0.001,0.999,1000)
for p in priori:
    bnb = BernoulliNB(class_prior=[p,1-p])
    scores = cross_val_score(bnb,
                             X,
                             y,
                             cv=10,
                             scoring = fbeta)
    scores_array.append(scores.mean())
    
plt.plot(priori,scores_array, 'ro')
plt.xlabel('Fbeta'); plt.ylabel('Porcentagem a priori de Spam');
plt.title('Escolha porcentagens a priori para Bernoulli Naive-Bayes');
pb = priori[np.argmax(scores_array)]
print(f'Pontuação Máxima:{round(max(scores_array),4)}')
print(f'Parâmetro Probabilidade:{round(pb,4)}')
scores_array = []
priori = np.linspace(0.001,0.999,1000)
for p in priori:
    mnb = MultinomialNB(class_prior=[p,1-p])
    scores = cross_val_score(mnb,
                             X,
                             y,
                             cv=10,
                             scoring = fbeta)
    scores_array.append(scores.mean())
    
plt.plot(priori,scores_array, 'ro')
plt.xlabel('Fbeta'); plt.ylabel('Porcentagem a priori de Spam');
plt.title('Escolha porcentagens a priori para Multinomial Naive-Bayes');
pm = priori[np.argmax(scores_array)]
print(f'Pontuação Máxima:{round(max(scores_array),4)}')
print(f'Parâmetro Probabilidade:{round(pm,4)}')
bnb = BernoulliNB(class_prior=[pb,1-pb])

bnb.fit(X,y)

test_data = bin_train8.loc[bin_train8['ham'].isnull()].drop('ham',axis=1)

testPred = bnb.predict(test_data)
arq = open ("prediction_bnb.csv", "w")
arq.write("Id,ham\n")
for i, j in zip(test_data.index, testPred):
    arq.write(str(i)+ "," + str(int(j))+"\n")
arq.close()
classifier = MultinomialNB(class_prior=[pm,1-pm])

mnb.fit(X,y)

test_data = bin_train8.loc[bin_train8['ham'].isnull()].drop('ham',axis=1)

testPred = mnb.predict(test_data)
arq = open ("prediction_mnb.csv", "w")
arq.write("Id,ham\n")
for i, j in zip(test_data.index, testPred):
    arq.write(str(i)+ "," + str(int(j))+"\n")
arq.close()
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=7)
classifier = MultinomialNB(class_prior=[pm,1-pm])

thresholds = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize = (8, 6))

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thr = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    thresholds.append(interp(mean_fpr, fpr, thr)) 
    thresholds[-1][0] = 1.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Acaso', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_thresholds = np.mean(thresholds, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Curva ROC média (AUC = %0.2f$\pm$ %0.2f)' % (mean_auc,std_auc),
         lw=2, alpha=.8)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para o melhor classificador (MNB)')
plt.legend(loc="lower right")
plt.show()
pont = 0
for i in range(len(mean_fpr)):
    pont_p = 10*mean_tpr[i]/(10*mean_tpr[i]+9*(1-mean_tpr[i])+mean_fpr[i])
    if pont_p > pont:
        pont = pont_p
        maxindex = i
print(f'Segundo a curva, o limiar ideal para maximizar fbeta é {round(mean_thresholds[maxindex], 4)}.')
print(f'Para este limiar, a taxa de falsos positivos esperada é {round(mean_fpr[maxindex], 4)}.\nA taxa de verdadeiros positivos é {round(mean_tpr[maxindex], 4)}.')
