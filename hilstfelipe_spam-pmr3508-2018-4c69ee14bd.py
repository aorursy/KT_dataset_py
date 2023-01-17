import statistics as stt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as slm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB # "this class requires samples to be represented as binary-valued feature vectors"
from sklearn.naive_bayes import MultinomialNB # "implements the naive Bayes algorithm for multinomially distributed data"
testspam = pd.read_csv("../input/spambase/test_features.csv",
            sep = r'\s*,\s*',
            engine = 'python',
            na_values = "?")
testspam.shape
testspam.head(5)
trainspam = pd.read_csv("../input/spambase/train_data.csv",
            sep = r'\s*,\s*',
            engine = 'python',
            na_values = "?")
trainspam.shape
trainspam.head(5)
spam = trainspam.query("ham == 0")
fig, axs = plt.subplots(2,3, sharey=True)
fig.subplots_adjust(right = 1.75)
fig.subplots_adjust(top = 1.5)

axs[0, 0].boxplot(spam["char_freq_;"], 0, '')
axs[0, 0].set_title('especial ;')

axs[0, 1].boxplot(spam["char_freq_("], 0, '')
axs[0, 1].set_title('especial ( ... )')

axs[0, 2].boxplot(spam["char_freq_["], 0, '')
axs[0, 2].set_title('especial [ ... ]')

axs[1, 0].boxplot(spam["char_freq_!"], 0, '')
axs[1, 0].set_title('especial !')

axs[1, 1].boxplot(spam["char_freq_$"], 0, '')
axs[1, 1].set_title('especial $')

axs[1, 2].boxplot(spam["char_freq_#"], 0, '')
axs[1, 2].set_title('especial #')

plt.show()
plt.close()
nospam = trainspam.query("ham == 1")
fig, axs = plt.subplots(2,3, sharey=True)
fig.subplots_adjust(right = 1.75)
fig.subplots_adjust(top = 1.5)

axs[0, 0].boxplot(nospam["char_freq_;"], 0, '')
axs[0, 0].set_title('especial ;')

axs[0, 1].boxplot(nospam["char_freq_("], 0, '')
axs[0, 1].set_title('especial ( ... )')

axs[0, 2].boxplot(nospam["char_freq_["], 0, '')
axs[0, 2].set_title('especial [ ... ]')

axs[1, 0].boxplot(nospam["char_freq_!"], 0, '')
axs[1, 0].set_title('especial !')

axs[1, 1].boxplot(nospam["char_freq_$"], 0, '')
axs[1, 1].set_title('especial $')

axs[1, 2].boxplot(nospam["char_freq_#"], 0, '')
axs[1, 2].set_title('especial #')

plt.show()
plt.close()
fig, axs = plt.subplots(1,3,sharey=True)
fig.subplots_adjust(right = 3.0)
fig.subplots_adjust(top = 1.25)

axs[0].hist(trainspam["char_freq_!"])
axs[0].set_title("TOTAL")

axs[1].hist(spam["char_freq_!"])
axs[1].set_title("SPAM")

axs[2].hist(nospam["char_freq_!"])
axs[2].set_title("NÃO SPAM")

plt.show()
plt.close()
fig, axs = plt.subplots(1,3,sharey=True)
fig.subplots_adjust(right = 3.0)
fig.subplots_adjust(top = 1.25)

axs[0].hist(trainspam["char_freq_("])
axs[0].set_title("TOTAL")

axs[1].hist(spam["char_freq_("])
axs[1].set_title("SPAM")

axs[2].hist(nospam["char_freq_("])
axs[2].set_title("NÃO SPAM")

plt.show()
plt.close()
char = trainspam[["char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#","ham"]]
char["bin"] = char["char_freq_;"]+char["char_freq_("]+char["char_freq_["]+char["char_freq_!"]+char["char_freq_$"]+char["char_freq_#"]
char["bin"] = np.where(char["bin"]==0,0,1)
char["char_freq_;"] = np.where(char["char_freq_;"]==0,0,1)
char["char_freq_("] = np.where(char["char_freq_("]==0,0,1)
char["char_freq_["] = np.where(char["char_freq_["]==0,0,1)
char["char_freq_!"] = np.where(char["char_freq_!"]==0,0,1)
char["char_freq_$"] = np.where(char["char_freq_$"]==0,0,1)
char["char_freq_#"] = np.where(char["char_freq_#"]==0,0,1)
charnospam = char.query("ham == 1")
charspam = char.query("ham == 0")

plt.figure(1)
plt.subplots_adjust(right = 1.2, top = 1.2)
plt.subplot(121)
plt.title('NO SPAM')
charnospam["bin"].value_counts().plot(kind='pie')
plt.subplot(122)
plt.title('SPAM')
charspam["bin"].value_counts().plot(kind='pie')

plt.show()
plt.close()
run = trainspam[["capital_run_length_average","capital_run_length_longest","capital_run_length_total","ham"]]
runham = run.query('ham == 1')
run.head()
def percent(colum):
    return colum*100//float(sum(colum))
hamxaverage = pd.crosstab(run["capital_run_length_average"],run["ham"])
hamxaverage.apply(percent,axis=1).plot()
hamxlongest = pd.crosstab(run["capital_run_length_longest"],run["ham"])
hamxlongest.apply(percent,axis=1).plot()
hamxtotal = pd.crosstab(run["capital_run_length_total"],run["ham"])
hamxtotal.apply(percent,axis=1).plot()
word = trainspam
word = word.drop(["char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#","capital_run_length_average","capital_run_length_longest","capital_run_length_total","Id"],axis=1)
tryword = word[["word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab",
                "word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85",
                "word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct"]]
word.head()
wordcorr = word.corr()
tryword = word[["word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct"]]
trywordcorr = tryword.corr()
plt.matshow(wordcorr,121)
plt.matshow(trywordcorr,122)
tryword = tryword.drop(["word_freq_george","word_freq_data","word_freq_857","word_freq_1999","word_freq_parts","word_freq_pm"],axis=1)
usefulwords = word.drop(tryword,axis=1)
usefulwords = usefulwords.drop("ham",axis=1)
names = usefulwords.columns.tolist()
Xtrainrun = trainspam[["capital_run_length_average","capital_run_length_longest","capital_run_length_total"]]
Ytrain = trainspam.ham
Xtrainrun.head(5)
means =[]
for num in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = num, weights ='distance')
    scores = cross_val_score(knn, Xtrainrun, Ytrain, cv=10)
    mean = stt.mean(scores)
    means.append(mean) 
bestn = means.index(max(means))+1
bestn
knn = KNeighborsClassifier(n_neighbors = bestn, weights ='distance')
scores = cross_val_score(knn, Xtrainrun, Ytrain, cv=10)
print("media:",stt.mean(scores))
print("desvio:",stt.pstdev(scores))
knn.fit(Xtrainrun, Ytrain)
Xtest = testspam[["capital_run_length_average","capital_run_length_longest","capital_run_length_total"]]
Ypred = knn.predict(Xtest)
testspam["runham"] = Ypred
testspam.head(5)
Xtrainspc = trainspam[["char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#"]]
means = []
for i in range(100):
    b = i/100
    bnb = BernoulliNB(binarize=b)   
    scores = cross_val_score(bnb, Xtrainspc, Ytrain, cv=10)
    mean = stt.mean(scores)
    means.append(mean) 
besti = means.index(max(means))+1
bestb = besti/100
bestb
bnb = BernoulliNB(binarize = bestb)
scores = cross_val_score(bnb, Xtrainspc, Ytrain, cv=10)
print("media:",stt.mean(scores))
print("desvio:",stt.pstdev(scores))
bnb.fit(Xtrainspc, Ytrain)
Xtest = testspam[["char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#"]]
Ypred = bnb.predict(Xtest)
testspam["spcham"] = Ypred
testspam.head(5)
Xtrainwfq = trainspam[names]
mnb = MultinomialNB(alpha=0.0000000001)
scores = cross_val_score(mnb, Xtrainwfq, Ytrain, cv=10)
print("media:",stt.mean(scores))
print("desvio:",stt.pstdev(scores)) 
mnb.fit(Xtrainwfq, Ytrain)
Xtest = testspam[names]
Ypred = mnb.predict(Xtest)
testspam["wfqham"] = Ypred
testspam.head(5)
x1 = np.bitwise_and(testspam["runham"],testspam["spcham"])
x2 = np.bitwise_and(testspam["runham"],testspam["wfqham"])
x3 = np.bitwise_and(testspam["spcham"],testspam["wfqham"])
x4 = np.bitwise_or(x1,x2)
testspam["ham"] = np.bitwise_or(x3,x4)
testspam.head(5)
Prun = knn.predict(Xtrainrun)
Pspc = bnb.predict(Xtrainspc)
Pwfq = mnb.predict(Xtrainwfq)
x1 = np.bitwise_and(Prun,Pspc)
x2 = np.bitwise_and(Prun,Pwfq)
x3 = np.bitwise_and(Pwfq,Pspc)
x4 = np.bitwise_or(x1,x2)
Predito = np.bitwise_or(x3,x4)
Real = Ytrain
C = slm.confusion_matrix(Real, Predito)
pd.DataFrame(C,
            index = ["Class Positve","Class Negative"],
            columns = ["Real Positive","Real Negative"])
Precisao = C[0][0]/(C[0][0]+C[0][1])
Recall = C[0][0]/(C[0][0]+C[1][0])
Acuracia = (C[0][0]+C[1][1])/(C[0][0]+C[1][0]+C[1][0]+C[1][1])
print("Precision = %0.2f" %Precisao)
print("Recall = %0.2f" %Recall)
print("Acuracia = %0.2f" %Acuracia)
f1 = slm.f1_score(Real, Predito)
print("F1 = %0.2f" %f1)
f3 = slm.fbeta_score(Real, Predito, 3)
print("F3 = %0.2f" %f3)
fpr, tpr, thresholds = slm.roc_curve(Real, Predito)
area = slm.roc_auc_score(Real, Predito)

plt.figure()

plt.plot(fpr, tpr, color='red', label=area)
plt.plot([0,1],[0,1],color='blue', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
plt.close()

print("Curva ROC de area %0.2f" %area)
#answer = testspam[["Id","ham"]]
#answer.to_csv(r"C:\Users\User\Desktop\sem_4\PMR3508-2018\spambase\answer.csv",index=False)