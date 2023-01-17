import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
spambase = pd.read_csv("../input/spambase/train_data.csv")
spambase.head()
spambase.describe()
#splitting database between spam and ham
ham = spambase.query('ham == 1')
spam = spambase.query('ham == 0')
#individually relating attributes to class
head = ham.columns.values.tolist()
for i in range(57):
    ham_term_mean = np.mean(ham[head[i]])
    spam_term_mean = np.mean(spam[head[i]])
    loc = [1, 2]
    h = [ham_term_mean, spam_term_mean]
    labels = ["Ham", "Spam"]
    if ((ham_term_mean >= spam_term_mean and ham_term_mean >= 2*spam_term_mean)      #highlights attributes with a notable
        or (ham_term_mean <= spam_term_mean and ham_term_mean*2 <= spam_term_mean)): #correlation with class variable
            plt.bar(loc, h, tick_label = labels, color = 'g')
    elif ((ham_term_mean >= spam_term_mean*0.75 and ham_term_mean <= 1.25*spam_term_mean) #highlights attributes with weak
        or spam_term_mean >= ham_term_mean*0.75 and spam_term_mean <= 1.25*ham_term_mean):#correlation with class variable
            plt.bar(loc, h, tick_label = labels, color = 'r')
    else:
        plt.bar(loc, h, tick_label = labels)
    plt.title(head[i])
    plt.ylabel('Avg value')
    plt.ylim(top = max([1.0, spam_term_mean*1.05, ham_term_mean*1.05])) #fixes scale for better comparison between charts,
    plt.show()                                                          #except for extreme cases
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
print(ham.shape)
print (spam.shape)
print(2251/(2251+1429))
#defining methods to be used and formatting dataset
x_base_1 = spambase.drop(["ham", 'Id'], 1)
y_base = spambase['ham']
nb = [BernoulliNB(), GaussianNB(), MultinomialNB(), 'BernoulliNB', 'GaussianNB', 'MultinomialNB']
#estimating precision and f3 score for each method
for i in range (3):
    nb[i].fit(x_base_1, y_base)
    score1 = cross_val_score(nb[i], x_base_1, y_base, cv=10)
    p = np.mean(score1)
    print(nb[i+3])
    print("Estimated precision: ", p)
    pred = nb[i].predict(x_base_1)
    f3scr = sklearn.metrics.fbeta_score(y_base, pred, 3)
    print("F3 score: ", f3scr, "\n")
x_base_2 = spambase.drop(["ham", 'Id', 'word_freq_will'], 1)
for i in range (3):
    nb[i].fit(x_base_2, y_base)
    score1 = cross_val_score(nb[i], x_base_2, y_base, cv=10)
    p = np.mean(score1)
    print(nb[i+3])
    print("Estimated precision: ", p)
    pred = nb[i].predict(x_base_2)
    f3scr = sklearn.metrics.fbeta_score(y_base, pred, 3)
    print("F3 score: ", f3scr, "\n")
x_base_3 = spambase.drop(["ham", 'Id', 'word_freq_will', 
           'word_freq_address', 'word_freq_report', 'word_freq_you', 'char_freq_('], 1)
for i in range (3):
    nb[i].fit(x_base_3, y_base)
    score1 = cross_val_score(nb[i], x_base_3, y_base, cv=10)
    p = np.mean(score1)
    print(nb[i+3])
    print("Estimated precision: ", p)
    pred = nb[i].predict(x_base_3)
    f3scr = sklearn.metrics.fbeta_score(y_base, pred, 3)
    print("F3 score: ", f3scr, "\n")
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import figure

X, y = x_base_3[:], y_base[:]

cv = StratifiedKFold(n_splits=10)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

i = 0
for train, test in cv.split(X, y):
    probas_ = nb[0].fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#finding best threshold value by finding the point closest to (0; 1)
import math
dist = 1000
c = 0
for i in range(len(mean_tpr)):
    dist1 = math.sqrt((mean_tpr[i]-1)**2 + (mean_fpr[i])**2)
    if dist1 < dist:
        dist = dist1
        j = i
t = 1-(j+1)/100
print ("For best threshold value of", t, "\nTrue positive rate:", mean_tpr[j],"\nFalse Positive Rate:", mean_fpr[j])
#finding F3 score for best threshold value
bnb = BernoulliNB(class_prior = [1-t, t])
bnb.fit(x_base_3, y_base)
y_pred = bnb.predict(x_base_3)
f3scr = sklearn.metrics.fbeta_score(y_base, y_pred, 3)
print("Best F3 score:", f3scr)
#making predictions
test = pd.read_csv("../input/spambase/test_features.csv")
xtest = test.drop(['Id', 'word_freq_will', 
           'word_freq_address', 'word_freq_report', 'word_freq_you', 'char_freq_('], 1)
test_pred = bnb.predict(xtest)
sub = pd.DataFrame({"id":test.Id, "ham":test_pred})
sub.to_csv("submission.csv", index=False)
sub