print(__doc__)



import numpy as np

import pandas as pd

import seaborn as sns

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

%matplotlib inline
h = .02  # step size in the mesh



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1),

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



df = pd.read_csv("../input/loan.csv", low_memory=False, nrows=10000)

df=df.dropna(axis=1, how='all')

df = df.drop(['id','member_id','url','title','emp_title','desc','mths_since_last_record'], 1)

print(df)
#look at data

#print(df.columns)



for c in df.columns:

    #print(df[c].head())

    #if df[c].dtype=='object':

        #print "Unique in %s: %d"%(c,df[c].nunique())

        #print pd.get_dummies(df[c]).head()

    print("Nulls in %s: %d"%(c,df[c].isnull().sum()))



d=['installment','grade','sub_grade']

e=['home_ownership','verification_status','issue_d','loan_status']

print(df[d].head())

print(pd.get_dummies(df[d].head()))

print(df[e])

df_dummies = pd.get_dummies(df.iloc[:,:])

#df_dummies
df[['mths_since_last_delinq']]=df[['mths_since_last_delinq']].fillna(-999)

df[['revol_util','last_pymnt_d','next_pymnt_d']]=df[['revol_util','last_pymnt_d','next_pymnt_d']].fillna(0)



df[['mths_since_last_delinq',

    'next_pymnt_d',

    'revol_util',

    'last_pymnt_d']]
groupby_grade=df.groupby('grade').mean()

groupby_grade
groupby_sub_grade=df.groupby('sub_grade').mean()

groupby_sub_grade
#home_ownership verification_status   issue_d  loan_status

#groupby_=df.groupby('').mean()

#groupby_

groupby_home_ownership=df.groupby('home_ownership').mean()

groupby_home_ownership
groupby_verification_status=df.groupby('verification_status').mean()

groupby_verification_status
groupby_loan_status=df.groupby('loan_status').mean()

groupby_loan_status
correlation = df.corr()#df_dummies.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix',annot_kws={"size": 6})



plt.title('Correlation')
#df[['out_prncp','out_prncp_inv']]

#df[['total_pymnt','total_pymnt_inv']]

print(df.columns)

df[[u'loan_amnt', u'funded_amnt', u'funded_amnt_inv',u'installment',u'total_pymnt',

       u'total_pymnt_inv', u'total_rec_prncp', u'total_rec_int']]

#sns.countplot(df['term'])

sns.countplot(df['grade'])

#sns.countplot(df['loan_status'])
figure = plt.figure(figsize=(27, 9))

i = 1



# preprocess dataset, split into training and test part

X, y = pd.get_dummies(df.drop('term', 1)), pd.get_dummies(df['term']).iloc[:,1]



X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

y_min, y_max = X[:, 5].min() - .5, X[:, 5].max() + .5



print(x_min, x_max,y_min, y_max)

# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

#                      np.arange(y_min, y_max, h))

xx, yy = np.meshgrid(np.r_[x_min:x_max:100j],

                     np.r_[y_min:y_max:100j])



print(xx.shape,yy.shape)

print(xx,yy)

print(xx.ravel().shape, yy.ravel().shape)

print(np.c_[xx.ravel(), yy.ravel()].shape)

# just plot the dataset first

cm = plt.cm.RdBu

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

#ax.set_title("Input data")

# Plot the training points

ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)

# and testing points

ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

ax.set_xlim(xx.min(), xx.max())

ax.set_ylim(yy.min(), yy.max())

ax.set_xticks(())

ax.set_yticks(())

i += 1



plt.tight_layout()

plt.show()
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



    # Plot also the training points

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)

    # and testing points

    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,

               alpha=0.6)



    ax.set_xlim(xx.min(), xx.max())

    ax.set_ylim(yy.min(), yy.max())

    ax.set_xticks(())

    ax.set_yticks(())

    ax.set_title(name)

    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),

            size=15, horizontalalignment='right')

    i += 1
plt.tight_layout()

plt.show()