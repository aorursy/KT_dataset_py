import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
import os
print(os.listdir("../input"))
spam = pd.read_csv("../input/dataset/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
spam.shape
spam.head()
spam_OnlyTrue=spam[spam['ham']==1]
spam_OnlyTrue2=spam_OnlyTrue[['capital_run_length_average','capital_run_length_longest','capital_run_length_total']]
spam_OnlyTrue = spam_OnlyTrue.loc[:, spam_OnlyTrue.columns != 'ham']
spam_OnlyTrue=spam_OnlyTrue.loc[:, spam_OnlyTrue.columns != 'Id']
spam_OnlyTrue=spam_OnlyTrue.loc[:, spam_OnlyTrue.columns != 'capital_run_length_average']
spam_OnlyTrue=spam_OnlyTrue.loc[:, spam_OnlyTrue.columns != 'capital_run_length_longest']
spam_OnlyTrue=spam_OnlyTrue.loc[:, spam_OnlyTrue.columns != 'capital_run_length_total']
spam_OnlyTrue.head()
spamTrue_meanValues=spam_OnlyTrue.mean()
spamTrue_stdValues=spam_OnlyTrue.std()
spamTrue_meanValues2=spam_OnlyTrue2.mean()
spamTrue_stdValues2=spam_OnlyTrue2.std()
spam_OnlyFalse=spam[spam['ham']==0]
spam_OnlyFalse2=spam_OnlyFalse[['capital_run_length_average','capital_run_length_longest','capital_run_length_total']]
spam_OnlyFalse = spam_OnlyFalse.loc[:, spam_OnlyFalse.columns != 'ham']
spam_OnlyFalse=spam_OnlyFalse.loc[:, spam_OnlyFalse.columns != 'Id']
spam_OnlyFalse=spam_OnlyFalse.loc[:, spam_OnlyFalse.columns != 'capital_run_length_average']
spam_OnlyFalse=spam_OnlyFalse.loc[:, spam_OnlyFalse.columns != 'capital_run_length_longest']
spam_OnlyFalse=spam_OnlyFalse.loc[:, spam_OnlyFalse.columns != 'capital_run_length_total']
spamFalse_meanValues=spam_OnlyFalse.mean()
spamFalse_stdValues=spam_OnlyFalse.std()
diff=abs(spamTrue_meanValues-spamFalse_meanValues)
spamFalse_meanValues2=spam_OnlyFalse2.mean()
spamFalse_stdValues2=spam_OnlyFalse2.std()
diff2=abs(spamTrue_meanValues2-spamFalse_meanValues2)
from matplotlib.pyplot import figure

figure(figsize=(20,20))

ax1 = plt.subplot(321)
ax1.set_ylim([0, 2.5])
spamTrue_meanValues.plot(kind='bar')

ax2 = plt.subplot(322)
ax2.set_ylim([0, 4.5])
spamTrue_stdValues.plot(kind='bar')


ax3 = plt.subplot(323)
ax3.set_ylim([0, 2.5])
spamFalse_meanValues.plot(kind='bar')

ax4 = plt.subplot(324)
ax4.set_ylim([0, 4.5])
spamFalse_stdValues.plot(kind='bar')

ax5 = plt.subplot(325)
ax5.set_ylim([0, 2])
diff.plot(kind='bar')

from matplotlib.pyplot import figure

figure(figsize=(20,20))

ax1 = plt.subplot(321)
#ax1.set_ylim([0, 2.5])
spamTrue_meanValues2.plot(kind='bar')

ax2 = plt.subplot(322)
#ax2.set_ylim([0, 4.5])
spamTrue_stdValues2.plot(kind='bar')


ax3 = plt.subplot(323)
#ax3.set_ylim([0, 2.5])
spamFalse_meanValues2.plot(kind='bar')

ax4 = plt.subplot(324)
#ax4.set_ylim([0, 4.5])
spamFalse_stdValues2.plot(kind='bar')

ax5 = plt.subplot(325)
#ax5.set_ylim([0, ])
diff2.plot(kind='bar')
testspam = pd.read_csv("../input/dataset/test_features.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testspam.head()
Xspam = spam[['word_freq_our', 'word_freq_free', 'word_freq_you', 'word_freq_your', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'char_freq_!', 'capital_run_length_average','capital_run_length_longest','capital_run_length_total']]
Yspam=spam.ham
Xtestspam = testspam[['word_freq_our', 'word_freq_free', 'word_freq_you', 'word_freq_your', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'char_freq_!', 'capital_run_length_average','capital_run_length_longest','capital_run_length_total']]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
maxscore=0
n=0
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, Xspam, Yspam, cv=10)
    meanscore=np.mean(scores)
    if meanscore>maxscore:
        maxscore=meanscore
        n=i
maxscore
n
knn = KNeighborsClassifier(n_neighbors=n)
scores = cross_val_score(knn, Xspam, Yspam, cv=10)
scores
mean=np.mean(scores)
mean
knn.fit(Xspam,Yspam)
YtestPred = knn.predict(Xtestspam)
result = np.vstack((testspam["Id"], YtestPred)).T
x = ["id","ham"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
bnb = BernoulliNB()
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
f3_scorer = make_scorer(fbeta_score, beta=3)
scores = cross_val_score(bnb, Xspam, Yspam, cv=10, scoring = f3_scorer)
print("Score F3 atingido", scores.mean())
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(bnb, Xspam, Yspam, cv=10)
cm = confusion_matrix(Yspam, y_pred)
print(cm)
from sklearn.metrics import roc_curve, auc

proba = cross_val_predict(bnb, Xspam, Yspam, cv=10, method = 'predict_proba') 
fpr, tpr, threshold = roc_curve(Yspam, proba[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
dist = 1000
for i in range(len(tpr)):
    dist1 = math.sqrt((tpr[i]-1)**2 + (fpr[i])**2)
    if dist1 < dist:
        dist = dist1
        indice = i
print("melhor indice:",indice)
print("false positive rate deste indice:", fpr[indice])
print("melhor limite de probabilidades:", threshold[indice])
bnb = BernoulliNB(class_prior = [1-threshold[indice], threshold[indice]])
scores = cross_val_score(bnb, Xspam, Yspam, cv=10, scoring = f3_scorer)
print("Score F3 atingido", scores.mean())
bnb.fit(Xspam, Yspam)
YtestPred = bnb.predict(Xtestspam)
result = np.vstack((testspam["Id"], YtestPred)).T
x = ["id","ham"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado
