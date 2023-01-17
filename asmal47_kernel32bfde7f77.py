#Python3 environment setup for EDA AND DATA PRE-PROCESSING of KDDCup'99 dataset
#This analysis will be a part of IIDS being deveoped.

#importing libraries
import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
##NOTE:--
##Please extract the dataset from the archived file under the datasets directory

df = pd.read_csv('../input/kddcup.data.corrected')
##Change according to the directory of the cloned repo w.r.t dataset location.
df.shape

#Coloumns names were based on the previous analysis of this dataset from https://datahub.io/machine-learning/kddcup99
df.columns =["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


df.head
df.info()

df.dtypes
for each_index,each_line in enumerate(open('../input/kddcup.data.corrected')):
    if each_index < 5:
        print(each_line.strip())
df.head()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service']= le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])

X = df.iloc[:,:41]
y = df.iloc[:,-1]
X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 34,test_size = 0.3)

X = df.iloc[:,:41]
y = df.iloc[:,-1]
X.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
df.info()
df.head()
df.astype('float64')
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 34,test_size = 0.3)
X = df.iloc[:,:41]
y = df.iloc[:,-1]
X.head()
df.head()
df.astype('float64')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# Construct our Linear Regression model
#lr = LinearRegression(normalize=True)
#lr.fit(X_train,y_train)
#stop the search when only the last feature is left
#rfe = RFE(lr, n_features_to_select=1, verbose =3 )
#rfe.fit(X_train,y_train)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')
model1.fit(X_train_scaled,y_train)
model1.fit(X_train,y_train)
#stop the search when only the last feature is left
#rfe = RFE(model1, n_features_to_select=1, verbose =3 )
#rfe.fit(X_train,y_train)
predict = model1.predict(X_test_scaled)
model1.score(X_test_scaled,y_test)
model1.score(X_train_scaled,y_train)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix,recall_score,precision_score
print((confusion_matrix(y_test,predict)))

from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

import matplotlib.pyplot as plt
plt.show()
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,15))
class_distribution = df['label'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
print('Optimal number of features: {}'.format(rfe.n_features_))
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')

#stop the search when only the last feature is left
rfecv = RFECV(estimator=model1, step=1, cv=StratifiedKFold(10), scoring='accuracy' )
rfecv.fit(X_train_scaled,y_train)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,class_weight='balanced')

#stop the search when only the last feature is left
rfecv = RFECV(estimator=model1, step=1, cv=StratifiedKFold(2), scoring='accuracy' )
rfecv.fit(X_train_scaled,y_train)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df, hue = "label");
sns.pairplot(df, hue = "label");