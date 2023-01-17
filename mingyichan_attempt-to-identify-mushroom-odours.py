import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline
mushrooms = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

mushrooms.info()
mushrooms.head()
mushrooms.describe()
mushrooms['odor'].value_counts()
pred_data = mushrooms.drop('odor',axis=1)

odours = mushrooms['odor']
from sklearn.preprocessing import LabelEncoder

Encoder_pred = LabelEncoder() 

for col in pred_data.columns:

    pred_data[col] = Encoder_pred.fit_transform(pred_data[col])

Encoder_odours = LabelEncoder()

odours = Encoder_odours.fit_transform(odours)
from sklearn.model_selection import train_test_split

pred_data_train, pred_data_test, odours_train, odours_test = train_test_split(pred_data, odours, test_size=0.2, random_state=1)
print("Before resampling:\n{}".format(np.asarray(np.unique(odours_train, return_counts=True)).T))



from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)

pred_data_train, odours_train = ros.fit_resample(pred_data_train, odours_train)



print("After resampling:\n{}".format(np.asarray(np.unique(odours_train, return_counts=True)).T))
from sklearn.naive_bayes import CategoricalNB



clf = CategoricalNB()

clf.fit(pred_data_train, odours_train)



print(clf.score(pred_data_train, odours_train))

print(clf.score(pred_data_test, odours_test))
import seaborn as sns

from sklearn.metrics import confusion_matrix



def visualize_confusion(classifier, pred_data_test, odours_test, encoder):

    conf = confusion_matrix(odours_test, classifier.predict(pred_data_test), normalize='true')

    fig, ax = plt.subplots(figsize=(10,10))

    labels = encoder.inverse_transform(classifier.classes_)

    sns.heatmap(conf, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    plt.show(block=False)
visualize_confusion(clf, pred_data_train, odours_train, Encoder_odours)
visualize_confusion(clf, pred_data_test, odours_test, Encoder_odours)
la_mushrooms = mushrooms[mushrooms.odor.isin(['l', 'a'])]

encoded_la = pd.DataFrame()

for col in la_mushrooms.columns:

    encoded_la[col] = LabelEncoder().fit_transform(la_mushrooms[col])



la_data = encoded_la.drop('odor', axis=1)

la_odours = encoded_la['odor']

from sklearn.feature_selection import chi2

_, pval = chi2(la_data, la_odours)

pval
for col_name in la_mushrooms.drop('odor', axis=1).columns:

    print("{}: {}\n".format(col_name, np.asarray(np.unique(la_mushrooms[col_name], return_counts=True)).T))
sy_mushrooms = mushrooms[mushrooms.odor.isin(['s', 'y'])]

encoded_sy = pd.DataFrame()

for col in sy_mushrooms.columns:

    encoded_sy[col] = LabelEncoder().fit_transform(sy_mushrooms[col])



sy_data = encoded_sy.drop('odor', axis=1)

sy_odours = encoded_sy['odor']

_, pval = chi2(sy_data, sy_odours)

pval
for col_name in sy_mushrooms.drop('odor', axis=1).columns:

    print("{}: {}\n".format(col_name, np.asarray(np.unique(sy_mushrooms[col_name], return_counts=True)).T))
# Get the 'l', 'a', 's', 'y' mushrooms

lasy_mushrooms = mushrooms[mushrooms.odor.isin(['l', 'a', 's', 'y'])]

lasy_data = lasy_mushrooms.drop('odor',axis=1)

lasy_odours = lasy_mushrooms['odor']



# Encoding categorical values into numerical ones

lasy_encoder_pred = LabelEncoder() 

for col in lasy_data.columns:

    lasy_data[col] = lasy_encoder_pred.fit_transform(lasy_data[col])

lasy_encoder_odours = LabelEncoder()

lasy_odours = lasy_encoder_odours.fit_transform(lasy_odours)



# Need to use one-hot encoding for classifiers that do not interpret categorical features correctly

# This will split all the categorical variables into binary ones - we will use PCA later to reduce dimensionality (while trying to retain variance information)

lasy_data = pd.get_dummies(lasy_data,columns=lasy_data.columns,drop_first=True)



# Split the dataset into training and test sets

lasy_data_train, lasy_data_test, lasy_odours_train, lasy_odours_test = train_test_split(lasy_data, lasy_odours, test_size=0.2, random_state=1)



# Oversample the training data for balance

ros = RandomOverSampler(random_state=1)

lasy_data_train, lasy_odours_train = ros.fit_resample(lasy_data_train, lasy_odours_train)



# PCA Step - Use Cumulative Summation of the Explained Variance to choose a good number of components

from sklearn.decomposition import PCA

pca = PCA().fit(lasy_data_train)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)')

plt.show()
pca = PCA(n_components=15)

lasy_data_train = pca.fit_transform(lasy_data_train)

lasy_data_test = pca.transform(lasy_data_test)
from sklearn.svm import SVC

train_acc = []

test_acc = []

c_range = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100]



for c in c_range:

    svc = SVC(C=c, kernel='rbf',random_state=1)

    svc.fit(lasy_data_train, lasy_odours_train)

    train_acc.append(svc.score(lasy_data_train, lasy_odours_train))

    test_acc.append(svc.score(lasy_data_test, lasy_odours_test))

    

plt.plot(c_range, train_acc, label="training accuracy")

plt.plot(c_range, test_acc, label="test accuracy")

plt.ylabel("Accuracy")

plt.xlabel("C")

plt.xscale("log")

plt.legend()
svc = SVC(C=0.1, kernel='rbf',random_state=1)

svc.fit(lasy_data_train, lasy_odours_train)

visualize_confusion(svc, lasy_data_train, lasy_odours_train, lasy_encoder_odours)
visualize_confusion(svc, lasy_data_test, lasy_odours_test, lasy_encoder_odours)
from sklearn.tree import DecisionTreeClassifier

train_acc_ent = []

test_acc_ent = []

train_acc_gini = []

test_acc_gini = []



depth_range = range(1, 31, 1)



for n in depth_range:

    dtc = DecisionTreeClassifier(max_depth=n, criterion='entropy', random_state=1)

    dtc.fit(lasy_data_train, lasy_odours_train)

    train_acc_ent.append(dtc.score(lasy_data_train, lasy_odours_train))

    test_acc_ent.append(dtc.score(lasy_data_test, lasy_odours_test))

    

    dtc = DecisionTreeClassifier(max_depth=n, criterion='gini', random_state=1)

    dtc.fit(lasy_data_train, lasy_odours_train)

    train_acc_gini.append(dtc.score(lasy_data_train, lasy_odours_train))

    test_acc_gini.append(dtc.score(lasy_data_test, lasy_odours_test))



plt.plot(depth_range, train_acc_ent, label="training accuracy ent")

plt.plot(depth_range, test_acc_ent, label="test accuracy ent")

plt.plot(depth_range, train_acc_gini, label="training accuracy gini")

plt.plot(depth_range, test_acc_gini, label="test accuracy gini")

plt.ylabel("Accuracy")

plt.xlabel("Max Depth")

plt.legend()
dtc = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)

dtc.fit(lasy_data_train, lasy_odours_train)

visualize_confusion(dtc, lasy_data_train, lasy_odours_train, lasy_encoder_odours)
visualize_confusion(dtc, lasy_data_test, lasy_odours_test, lasy_encoder_odours)
from sklearn.neighbors import KNeighborsClassifier

train_acc = []

test_acc = []



k_range = range(1, 51, 1)



for k in k_range:

    knc = KNeighborsClassifier(n_neighbors=k)

    knc.fit(lasy_data_train, lasy_odours_train)

    train_acc.append(knc.score(lasy_data_train, lasy_odours_train))

    test_acc.append(knc.score(lasy_data_test, lasy_odours_test))



plt.plot(k_range, train_acc, label="training accuracy")

plt.plot(k_range, test_acc, label="test accuracy")

plt.ylabel("Accuracy")

plt.xlabel("# of Neighbours")

plt.legend()
knc = KNeighborsClassifier(n_neighbors=45)

knc.fit(lasy_data_train, lasy_odours_train)

visualize_confusion(knc, lasy_data_train, lasy_odours_train, lasy_encoder_odours)
visualize_confusion(knc, lasy_data_test, lasy_odours_test, lasy_encoder_odours)