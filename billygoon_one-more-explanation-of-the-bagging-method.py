import pandas as pd

from sklearn.datasets import make_classification



X, y = make_classification(n_samples=100000, 

                           n_features=30, 

                           n_redundant=0,

                           n_clusters_per_class=2, 

                           weights=[0.98], 

                           flip_y=0, 

                           random_state=12345)
cols = ["feature_"+str(i) for i in range(1, X.shape[1]+1)]

df = pd.DataFrame.from_records(X, columns=cols)

df["Class"] = y
pos_samples = df[df["Class"]==1].sample(frac=1) #extracting positive samples

neg_samples = df[df["Class"]==0].sample(frac=1) #extracting negative samples



# let's set aside 500 positive and 500 negative samples

test = pd.concat((pos_samples[:500], neg_samples[:500]), axis=0).sample(frac=1).reset_index(drop=True) #combine, shuffle, reset indices



# let's use the rest for training

train = pd.concat((pos_samples[500:], neg_samples[500:]), axis=0).sample(frac=1).reset_index(drop=True) #combine, shuffle, reset indices
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=1000, #the more the better, but slower

                               random_state=12345, #lucky number

                               class_weight="balanced",

                               verbose=1,

                               n_jobs=-1).fit(train.values[:,:-1], train["Class"])
from sklearn.metrics import classification_report, balanced_accuracy_score, plot_confusion_matrix



preds = model.predict(test.values[:,:-1])

print(classification_report(test["Class"], preds))

print("Accuracy: {}%".format(int(balanced_accuracy_score(test["Class"], preds)*100)))
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})



fig, ax = plt.subplots(figsize=(5, 5))

_=plot_confusion_matrix(model, test.values[:,:-1], test["Class"], values_format = '.0f', cmap=plt.cm.Blues, ax=ax)

pos_samples = train[train["Class"]==1].sample(frac=1)

neg_samples = train[train["Class"]==0].sample(frac=1)



#lets split into 5 bags

train_1 = pd.concat((pos_samples[:300], neg_samples[:3000]), axis=0)

train_2 = pd.concat((pos_samples[300:600], neg_samples[3000:6000]), axis=0)

train_3 = pd.concat((pos_samples[600:900], neg_samples[6000:9000]), axis=0)

train_4 = pd.concat((pos_samples[900:1200], neg_samples[9000:12000]), axis=0)

train_5 = pd.concat((pos_samples[1200:], neg_samples[12000:15000]), axis=0)
bag_1 = RandomForestClassifier(n_estimators=1000, random_state=12345, class_weight="balanced", n_jobs=-1).fit(train_1.values[:,:-1], train_1["Class"])

bag_2 = RandomForestClassifier(n_estimators=1000, random_state=12345, class_weight="balanced", n_jobs=-1).fit(train_2.values[:,:-1], train_2["Class"])

bag_3 = RandomForestClassifier(n_estimators=1000, random_state=12345, class_weight="balanced", n_jobs=-1).fit(train_3.values[:,:-1], train_3["Class"])

bag_4 = RandomForestClassifier(n_estimators=1000, random_state=12345, class_weight="balanced", n_jobs=-1).fit(train_4.values[:,:-1], train_4["Class"])

bag_5 = RandomForestClassifier(n_estimators=1000, random_state=12345, class_weight="balanced", n_jobs=-1).fit(train_5.values[:,:-1], train_5["Class"])
probs_1 = bag_1.predict_proba(test.values[:,:-1])[:,1]

probs_2 = bag_2.predict_proba(test.values[:,:-1])[:,1]

probs_3 = bag_3.predict_proba(test.values[:,:-1])[:,1]

probs_4 = bag_4.predict_proba(test.values[:,:-1])[:,1]

probs_5 = bag_5.predict_proba(test.values[:,:-1])[:,1]
probs = (probs_1+probs_2+probs_3+probs_4+probs_5)/5

preds = [1 if prob >= 0.5 else 0 for prob in probs]

print(classification_report(test["Class"], preds))

print("Accuracy: {}%".format(int(balanced_accuracy_score(test["Class"], preds)*100)))
from sklearn.metrics import confusion_matrix

import seaborn as sns



cm = confusion_matrix(test["Class"], preds)

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap=plt.cm.Blues)



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')

ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0'])

_=plt.tight_layout()
dts = [i/100 for i in range(10, 100, 5)]

accs = []

for dt in dts:

    probs = model.predict_proba(test.values[:,:-1])[:,1]

    preds = [1 if prob >= dt else 0 for prob in probs]

    acc = balanced_accuracy_score(test["Class"], preds)

    accs.append(acc)
fig=plt.figure(figsize=(12, 5))

_=plt.plot(accs)

_=plt.xticks([i for i in range(len(dts))], dts)

_=plt.grid()

_=plt.tight_layout()

_=plt.xlabel("Decision thresholds")

_=plt.ylabel("Accuracies")
probs = model.predict_proba(test.values[:,:-1])[:,1]

preds = [1 if prob >= 0.1 else 0 for prob in probs]

print(classification_report(test["Class"], preds))

print("Accuracy: {}%".format(int(balanced_accuracy_score(test["Class"], preds)*100)))

cm = confusion_matrix(test["Class"], preds)

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap=plt.cm.Blues)

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0'])

_=plt.tight_layout()
dts = [i/100 for i in range(10, 100, 5)]

accs = []

for dt in dts:

    probs_1 = bag_1.predict_proba(test.values[:,:-1])[:,1]

    probs_2 = bag_2.predict_proba(test.values[:,:-1])[:,1]

    probs_3 = bag_3.predict_proba(test.values[:,:-1])[:,1]

    probs_4 = bag_4.predict_proba(test.values[:,:-1])[:,1]

    probs_5 = bag_5.predict_proba(test.values[:,:-1])[:,1]

    probs = (probs_1+probs_2+probs_3+probs_4+probs_5)/5

    preds = [1 if prob >= dt else 0 for prob in probs]

    acc = balanced_accuracy_score(test["Class"], preds)

    accs.append(acc)
fig=plt.figure(figsize=(12, 5))

_=plt.plot(accs)

_=plt.xticks([i for i in range(len(dts))], dts)

_=plt.grid()

_=plt.tight_layout()

_=plt.xlabel("Decision thresholds")

_=plt.ylabel("Accuracies")
probs_1 = bag_1.predict_proba(test.values[:,:-1])[:,1]

probs_2 = bag_2.predict_proba(test.values[:,:-1])[:,1]

probs_3 = bag_3.predict_proba(test.values[:,:-1])[:,1]

probs_4 = bag_4.predict_proba(test.values[:,:-1])[:,1]

probs_5 = bag_5.predict_proba(test.values[:,:-1])[:,1]

probs = (probs_1+probs_2+probs_3+probs_4+probs_5)/5

preds = [1 if prob >= 0.1 else 0 for prob in probs]

print(classification_report(test["Class"], preds))

print("Accuracy: {}%".format(int(balanced_accuracy_score(test["Class"], preds)*100)))
cm = confusion_matrix(test["Class"], preds)

ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap=plt.cm.Blues)

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0'])

_=plt.tight_layout()