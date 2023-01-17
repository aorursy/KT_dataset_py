import pandas

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.metrics import SCORERS

from array import array



def scoreIt(a, b):

    from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

    try:

        v = accuracy_score(a, b)

        cut = -1

    except:

        #v = (mean_absolute_error(a, b), 0)

        scan_range = np.arange(0, 1.0, 0.01)

        l = [accuracy_score(a > x, b) for x in scan_range]

        v = np.amax(l, axis = None)

        try:

            if len(v) >= 1:

                v = v[0]

        except:

            pass

        cut = scan_range[np.where(l == v)[0][0]]

    return (v, cut)



full_metric = {}



dataset = pandas.read_csv("../input/data.csv")      # read dataset

dataset = dataset.drop(dataset.columns[-1], axis=1) # drop unnamed column that shows up last



# make test and training set

x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:,2:], dataset.iloc[:,1], test_size = 0.3, random_state = 0)
# make a violin plot of the features

nfeatures = len(x_train.columns)

for i in range(0, int(nfeatures/2)):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    for j in range(0, 2):

        sns.violinplot(y=x_train.columns[i*2+j], data = x_train, ax = ax[j])

plt.show()
# check how often one diagnosis happen and not the other

print(y_train.describe())

sns.countplot(x="diagnosis", data=pandas.DataFrame({"diagnosis": y_train}))

plt.show()
sns.heatmap(x_train.corr()) # show correlation plot

plt.show()
# encode labels in two numerical classes

labels = y_train.unique()

print(labels)

label_encoder = LabelEncoder()

label_encoder.fit(y_train)

y_train_trans = label_encoder.transform(y_train)

y_test_trans = label_encoder.transform(y_test)



# scale inputs so that they are within -1 and 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scale = scaler.transform(x_train)

x_test_scale = scaler.transform(x_test)
from sklearn.neural_network import MLPClassifier

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



nn_config = [(15, 10, 5, 2), (10, 5), (10, 2), (15,5)]



def classifyNN(c, x_train_scale, y_train_trans, x_test_scale, y_test_trans, regressor = True):

    if regressor:

        clf = MLPRegressor(solver='lbfgs', activation="tanh", alpha=1e-6, hidden_layer_sizes=c)

    else:

        clf = MLPClassifier(solver='lbfgs', activation="tanh", alpha=1e-6, hidden_layer_sizes=c)

    clf.fit(x_train_scale, y_train_trans)

    metric_result = scoreIt(clf.predict(x_test_scale), y_test_trans)

    fpr, tpr, thresholds = roc_curve(y_test_trans, clf.predict(x_test_scale) , pos_label=0)

    return fpr, tpr, thresholds, metric_result



def runAll(x_train_scale, y_train_trans, x_test_scale, y_test_scale):

    metric_result = {}

    fpr = {}

    tpr = {}

    thresholds = {}

    sty=["r", 'b', 'g', 'k', 'c', 'y', 'm']



    for c in nn_config:

        c_name = "_".join([str(x) for x in c])

        fpr[c_name], tpr[c_name], thresholds[c_name], metric_result['nn_%s'%c_name] = classifyNN(c, x_train_scale, y_train_trans, x_test_scale, y_test_trans, True)

        fpr["c_"+c_name], tpr["c_"+c_name], thresholds["c_"+c_name], metric_result['c_nn_%s'%c_name] = classifyNN(c, x_train_scale, y_train_trans, x_test_scale, y_test_trans, False)

    

    clf = LogisticRegression()

    clf.fit(x_train_scale, y_train_trans)

    metric_result['c_logistic'] = scoreIt(clf.predict(x_test_scale), y_test_trans)

    fpr['c_logistic'], tpr['c_logistic'], thresholds['c_logistic'] = roc_curve(y_test_trans, clf.predict(x_test_scale), pos_label=0)



    clf = DecisionTreeClassifier()

    clf.fit(x_train_scale, y_train_trans)

    metric_result['c_dt'] = scoreIt(clf.predict(x_test_scale), y_test_trans)

    fpr['c_dt'], tpr['c_dt'], thresholds['c_dt'] = roc_curve(y_test_trans, clf.predict(x_test_scale), pos_label=0)



    clf = RandomForestClassifier()

    clf.fit(x_train_scale, y_train_trans)

    metric_result['c_rf'] = scoreIt(clf.predict(x_test_scale), y_test_trans)

    fpr['c_rf'], tpr['c_rf'], thresholds['c_rf'] = roc_curve(y_test_trans, clf.predict(x_test_scale), pos_label=0)

    

    k = 0

    plt.figure(figsize=(15,20))

    for cname in sorted(thresholds):

        if "c_" in cname: continue

        plt.plot(tpr[cname], fpr[cname], sty[k]+'-', label = "%s (Area = %f)" % (cname, auc(tpr[cname], fpr[cname])))

        plt.plot(tpr["c_"+cname], fpr["c_"+cname], sty[k]+'v', label = "Classifier %s (Area = %f)" % (cname, auc(tpr['c_'+cname], fpr['c_'+cname])), markersize=10)

        k += 1

    plt.plot(tpr["c_logistic"], fpr["c_logistic"], sty[k]+'v', label = "Logistic Regression (Area = %f)" % (auc(tpr['c_logistic'], fpr['c_logistic'])), markersize=10)

    k+=1

    plt.plot(tpr["c_dt"], fpr["c_dt"], sty[k]+'v', label = "Decision Tree (Area = %f)" % (auc(tpr['c_dt'], fpr['c_dt'])), markersize=10)

    k+=1

    plt.plot(tpr["c_rf"], fpr["c_rf"], sty[k]+'v', label = "Random Forest (Area = %f)" % (auc(tpr['c_rf'], fpr['c_rf'])), markersize=10)

    k+=1

    plt.xlabel("Efficiency")

    plt.ylabel("Fake rate")

    plt.title("ROC")

    plt.legend(loc="best")

    plt.show()

    print(metric_result)

    return metric_result



tmp = runAll(x_train_scale, y_train_trans, x_test_scale, y_test_trans)

for k in tmp:

    full_metric["no_preprocessing_%s" % k] = tmp[k]
# now do PCA to find what are the variables that matter the most

sns.heatmap(x_train.corr()) # show correlation plot

plt.title("Pearson correlation of the train set before PCA")

plt.show()



from sklearn.decomposition import PCA

pca_trans = PCA()

pca_trans.fit(x_train)



# check impact of components

fig = plt.figure(figsize=(15,15))

plt.plot(range(0, len(pca_trans.explained_variance_)), pca_trans.explained_variance_/np.sum(pca_trans.explained_variance_))

plt.title("Fraction of the variance kept in the PCA transformed train set")

plt.xlabel("Number of variables used")

plt.ylabel("Cumulative fraction of variance")

plt.show()



# apply it

x_train_pca = pca_trans.transform(x_train)

x_test_pca = pca_trans.transform(x_test)



# check correlation matrix after transform: should be diagonal by construction

sns.heatmap(np.corrcoef(x_train_pca, rowvar=0))

plt.title("PCA transformed train set")

plt.show()



sns.heatmap(np.corrcoef(x_test_pca, rowvar=0))

plt.title("PCA transformed test set")

plt.show()



# make a violin plot of the transformed features

nfeatures_pca = 4 # these seem enough and contain most of the variance

for i in range(0, int(nfeatures_pca/2)):

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))

    for j in range(0, 2):

        sns.violinplot(data = x_train_pca[:, i*2+j], ax=ax[j])

    plt.show()



pca_scaler = StandardScaler()

pca_scaler.fit(x_train_pca)

x_train_scale_pca = pca_scaler.transform(x_train_pca)

x_test_scale_pca = pca_scaler.transform(x_test_pca)



# try classifying those variables instead

tmp = runAll(x_train_scale_pca, y_train_trans, x_test_scale_pca, y_test_trans)

for k in tmp:

    full_metric["PCA_%s" % k] = tmp[k]



# try doing it again, but using only 5 variables

tmp = runAll(x_train_scale_pca[:, 0:5], y_train_trans, x_test_scale_pca[:, 0:5], y_test_trans)

for k in tmp:

    full_metric["PCA5_%s" % k] = tmp[k]
from sklearn.decomposition import FastICA

ica_trans = FastICA(max_iter=100)

ica_trans.fit(x_train_scale)



# apply it

x_train_ica = ica_trans.transform(x_train_scale)

x_test_ica = ica_trans.transform(x_test_scale)



# check correlation matrix after transform

sns.heatmap(np.corrcoef(x_train_ica, rowvar=0))

plt.title("Pearson correlation of the train set after the ICA is applied")

plt.show()



# make a violin plot of the transformed features

nfeatures_ica = 6

for i in range(0, int(nfeatures_ica/2)):

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))

    for j in range(0, 2):

        sns.violinplot(data = x_train_ica[:, i*2+j], ax=ax[j])

    plt.show()



ica_scaler = StandardScaler()

ica_scaler.fit(x_train_ica)

x_train_scale_ica = ica_scaler.transform(x_train_ica)

x_test_scale_ica = ica_scaler.transform(x_test_ica)



tmp = runAll(x_train_scale_ica, y_train_trans, x_test_scale_ica, y_test_trans)

for k in tmp:

    full_metric["ICA_%s" % k] = tmp[k]
# now try NaiveBayes with independent components



from sklearn.naive_bayes import GaussianNB, MultinomialNB



nb = GaussianNB()

nb.fit(x_train_pca, y_train_trans)

x_train_pca_nb = nb.predict(x_train_pca)

x_test_pca_nb = nb.predict(x_test_pca)



full_metric["c_PCA_NB"] = scoreIt(nb.predict(x_test_pca), y_test_trans)



nb = GaussianNB()

nb.fit(x_train_ica, y_train_trans)

x_train_ica_nb = nb.predict(x_train_ica)

x_test_ica_nb = nb.predict(x_test_ica)



full_metric["c_ICA_NB"] = scoreIt(nb.predict(x_test_ica), y_test_trans)
from sklearn.cluster import KMeans



# try un-supervised learning



km = KMeans(n_clusters=2)

km.fit(x_train_scale)

x_train_km = km.predict(x_train_scale)

x_test_km = km.predict(x_test_scale)



#sns.distplot(-x_test_km + y_test_trans)

#plt.show()





print(accuracy_score(km.predict(x_test_scale), -y_test_trans))

full_metric["c_kmeans"] = scoreIt(km.predict(x_test_scale), -y_test_trans)
from sklearn.svm import SVC, SVR



for C in [0.5, 0.8, 1, 1.2, 1.5, 1.8, 2, 3, 10, 50, 100, 150, 200]:

    svm = SVR(C=C)

    svm.fit(x_train_scale, y_train_trans)

    #sns.distplot(svm.predict(x_test_scale) - y_test_trans)

    #plt.show()



    full_metric["SVM_%f" % C] = scoreIt(svm.predict(x_test_scale), y_test_trans)



    svc = SVC(C=C)

    svc.fit(x_train_scale, y_train_trans)

    #sns.distplot(svc.predict(x_test_scale) - y_test_trans)

    #plt.show()



    full_metric["c_SVC_%f" % C] = scoreIt(svc.predict(x_test_scale), y_test_trans)



    svc_ica = SVC(C=C)

    svc_ica.fit(x_train_ica, y_train_trans)

    #sns.distplot(svc_ica.predict(x_test_ica) - y_test_trans)

    #plt.show()



    full_metric["c_ICA_SVC_%f" % C] = scoreIt(svc_ica.predict(x_test_ica), y_test_trans)



    svc_pca = SVC(C=C)

    svc_pca.fit(x_train_pca, y_train_trans)

    #sns.distplot(svc_pca.predict(x_test_pca) - y_test_trans)

    #plt.show()



    full_metric["c_PCA_SVC_%f" % C] = scoreIt(svc_pca.predict(x_test_pca), y_test_trans)
print("Classifiers:")

for k in sorted(full_metric, key = lambda item: full_metric[item][0], reverse = True):

    if not "c_" in k: continue

    print("%50s     %20s" % (k, full_metric[k][0]))



print("Regressors:")

for k in sorted(full_metric, key = lambda item: full_metric[item][0], reverse = True):

    if "c_" in k: continue

    print("%50s     %20s     %20s" % (k, full_metric[k][0], full_metric[k][1]))