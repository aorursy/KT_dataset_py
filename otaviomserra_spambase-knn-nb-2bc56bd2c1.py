import pandas as pd
import matplotlib.pyplot as plt
spamTrain = pd.read_csv("../input/spambase-pmr3508/train_data.csv")
spamTrain.head()
spamTrain.shape
corrData = spamTrain.corr()
corrData.head()
corrOrder = corrData["ham"]
corrOrder
xSpamTrain = spamTrain[["word_freq_all","word_freq_our","word_freq_over","word_freq_remove",
                         "word_freq_internet","word_freq_order","word_freq_receive","word_freq_addresses",
                         "word_freq_free","word_freq_email","word_freq_you","word_freq_credit","word_freq_your",
                         "word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george",
                         "word_freq_1999","char_freq_!","char_freq_$","capital_run_length_longest",
                         "capital_run_length_total"]]
ySpamTrain = spamTrain["ham"]
spamTest = pd.read_csv("../input/spambase-pmr3508/test_features.csv")[["word_freq_all","word_freq_our","word_freq_over","word_freq_remove",
                         "word_freq_internet","word_freq_order","word_freq_receive","word_freq_addresses",
                         "word_freq_free","word_freq_email","word_freq_you","word_freq_credit","word_freq_your",
                         "word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george",
                         "word_freq_1999","char_freq_!","char_freq_$","capital_run_length_longest",
                         "capital_run_length_total"]]
spamTest.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
values = [1,5,10,20,30,40,50,60]
scores = []
for k in values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(sum(cross_val_score(knn,xSpamTrain,ySpamTrain,cv=10))/10)
scores
new_values = [1,2,3,4,5]
new_scores = []
for k in new_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    new_scores.append(sum(cross_val_score(knn,xSpamTrain,ySpamTrain,cv=10))/10)
new_scores
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xSpamTrain,ySpamTrain)
target_pred = knn.predict(spamTest).tolist()
id_list = pd.read_csv("../input/spambase-pmr3508/test_features.csv")["Id"].tolist()
pd.DataFrame({"Id":id_list,"ham":target_pred}).to_csv("prediction.csv",index=False)
from sklearn.naive_bayes import GaussianNB
nbXSpamTrain = pd.read_csv("../input/spambase-pmr3508/train_data.csv").drop(["Id","ham"],axis=1)
nbSpamTest = pd.read_csv("../input/spambase-pmr3508/test_features.csv").drop("Id",axis=1)
nb_scores = []
nb = GaussianNB()
nb_scores.append(sum(cross_val_score(nb,nbXSpamTrain,ySpamTrain,cv=10))/10)
nb_scores.append(sum(cross_val_score(nb,xSpamTrain,ySpamTrain,cv=10))/10)
nb_scores
nbXSpamTrain = spamTrain[["word_freq_remove","word_freq_free","word_freq_you","word_freq_business",
                          "word_freq_your","word_freq_000","word_freq_money","word_freq_hp","char_freq_$",
                          "capital_run_length_total"]]
nbSpamTest = pd.read_csv("../input/spambase-pmr3508/test_features.csv")[["word_freq_remove","word_freq_free","word_freq_you","word_freq_business",
                          "word_freq_your","word_freq_000","word_freq_money","word_freq_hp","char_freq_$",
                          "capital_run_length_total"]]
nb_scores.append(sum(cross_val_score(nb,nbXSpamTrain,ySpamTrain,cv=10))/10)
nb_scores
nb.fit(xSpamTrain,ySpamTrain)
target_pred = nb.predict(spamTest).tolist()
pd.DataFrame({"Id":id_list,"ham":target_pred}).to_csv("nbprediction.csv",index=False)