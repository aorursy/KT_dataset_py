# Scikit-learn

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.svm import SVC, SVR

from sklearn import datasets

from sklearn.metrics import balanced_accuracy_score, classification_report

# Other libraries

import numpy as np

import pandas as pd

import scipy.stats as sps

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# Use vector drawing inside jupyter notebook and Set matplotlib default axis font size

%config InlineBackend.figure_format = "svg"

plt.rcParams.update({'font.size': 8})
# quadratic function

def f2fun(x):

    return -1.3 + 0.4*x + 0.3 * x**2



# regular grid converted to features matrix

x0, y0 = np.meshgrid(np.linspace(-5.0,5.0,num=30), np.linspace(-2.0,2.8,num=20))



# Convert to feature matrix

X = np.concatenate((np.asanyarray(x0).reshape(x0.size,1),np.asanyarray(y0).reshape(y0.size,1)),axis=1)



# set labels (below quadratic line to 1)

y = np.zeros(y0.size);

y[f2fun(X[:,0]) < X[:,1]] = 1

y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)
# Here, the data is scaled prior train-test split. 

# In real applications, first split and scale afterwards, to simulate real-world scenario where we do not have the actual (out of sample) test set!

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,

                                                 train_size=0.7, test_size=0.3);
def plotsynth(X,y,t="Synthetic data input"):

    plt.scatter(X[:,0],X[:,1],c=y,marker=".",s=4);

    plt.title(t);

    plt.axis("tight")
cv = KFold(n_splits=5);

models = [GridSearchCV(SVC(kernel="linear"),

                       param_grid = {"C":np.logspace(-3, 3, 7)}, # controls hard-soft margin

                       cv=cv,scoring="balanced_accuracy"),

          GridSearchCV(SVC(kernel="rbf"),

                       param_grid = {"C":np.logspace(-3, 3, 7),

                                     "gamma":np.logspace(-3, 3, 7)},

                       cv=cv,scoring="balanced_accuracy"),

          GridSearchCV(SVC(kernel="poly"),

                       param_grid = {"C":np.logspace(-2, 2, 5),

                                     "gamma":np.logspace(-3, 0, 4), # set to max 1

                                     "coef0":np.logspace(-2, 2, 5),

                                     "degree":[2,3]}, # normally we would not know the degree

                       cv=cv,scoring="balanced_accuracy")]
# Plot input

plt.figure(figsize=(9,2.5));

plt.subplot(1,4,1);s = 2

plotsynth(X,y,t="Input");



# Fit and show result

for (n,m) in zip(["Linear  ", "RBF    ","Polynomial"],models):

    clf = m.fit(X_test,y_test);

    print(n,":\t model accuracy score train = {:.3f}, test = {:.3f}".format(

        balanced_accuracy_score(y_train,clf.predict(X_train)),

          balanced_accuracy_score(y_test,clf.predict(X_test))))

    plt.subplot(1,4,s); s+= 1;

    plotsynth(X,clf.predict(X),t=n);
# select rbf

rbf = models[1].best_estimator_;

print("RBF (non-margin) Support Vectors (SV) info:");

# Show number of SVs and corresponding indices (=X_train[rbf.support_,:])

for (i,v) in zip(["Nr. of class-specific (first,second) SVs","SV indices"],

                 [rbf.n_support_,rbf.support_]):

    print(i," = {}".format(v))
X = np.arange(0,21).reshape(21,1)

y = np.sin(2*np.pi*1/10.0*X[:,0]);
svm = GridSearchCV(SVR(kernel="rbf"),

                   param_grid={"C":np.logspace(-3, 3, 7),

                               "gamma":np.logspace(-3, 3, 7)},

                   cv = 3).fit(X,y);
plt.figure(figsize=(5,2.5))

plt.plot(X[:,0],y,"kx")

plt.plot(X[:,0],svm.predict(X),"g.--")

plt.legend(["input","rbf"])

plt.title("SVM Regression");