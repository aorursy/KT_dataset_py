import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
# This magic will ensure that the plots are in Jupyter in high quality
# see https://stackoverflow.com/questions/25412513/inline-images-have-low-quality
%config InlineBackend.figure_format = "svg"
iris = load_iris();
# use PCA as logistic regression that requires un-correlated features. 
X = StandardScaler().fit_transform(iris.data);
X = PCA(n_components="mle").fit_transform(X); 
y = LogisticRegression(solver="newton-cg",
                       multi_class="auto",random_state=0).fit(X,iris.target).predict(X)
X_train,y_train = shuffle(X,y,random_state=1)
np.random.seed(123)
nan_indices = np.random.binomial(True,0.2,X.size).reshape(X.shape).astype(bool);
X_train[nan_indices]= np.NaN;
X_test = X.copy()[nan_indices.any(axis=1),:]
y_test = y.copy()[nan_indices.any(axis=1)]
def fitmodel(xfit,yfit,xtest,ytest,text=""):
    modin=LogisticRegression(solver="newton-cg",multi_class="auto",random_state=0);
    modin.fit(xfit,yfit);
    # show accuracy for the missing data only
    print("{} model accuracy score = {:.3f}".format(text,
            balanced_accuracy_score(ytest,modin.predict(xtest))))
fitmodel(X,y,X_test,y_test,text="No missing")
X_del = X_train[~nan_indices.any(axis=1),:]
y_del = y_train[~nan_indices.any(axis=1)]
fitmodel(X_del,y_del,X_test,y_test,text="Deletion")
for strategy in ["mean","median"]:
    X_inp = SimpleImputer(strategy=strategy).fit_transform(X_train);
    fitmodel(X_inp,y_train,X_test,y_test,text=strategy)
# Fit distribution
param_fit = {}
plt.figure(figsize=(9,3))
for i in range(0,X.shape[1]):
    param_fit[i] = sp.norm.fit(X[:,i]);
    plt.subplot(1,X.shape[1],i+1)
    plt.hist(X[:,i],"sqrt",density=True);
    pdf = sp.norm.pdf(np.linspace(-4,4,100),loc=param_fit[i][0],scale=param_fit[i][1]);
    plt.plot(np.linspace(-4,4,100),pdf,"r-");
# Run model m-times and store each result in `y_m`
y_m = np.zeros((y_test.size,20));
np.random.seed(0)
for m in range(0,20):
    X_m = X_train.copy();
    # fill missing values
    for i in range(0,X.shape[1]):
        X_m[nan_indices[:,i],i] = sp.norm.rvs(size=np.sum(nan_indices[:,i]),
                                              loc=param_fit[i][0],scale=param_fit[i][1]);
    # fit model
    y_m[:,m] = LogisticRegression(solver="newton-cg",
                       multi_class="auto",random_state=0).fit(X_m,y_train).predict(X_test)
# Use most 
print("Multiple random imputation model accuracy score = {:.3f}".format(
            balanced_accuracy_score(y_test,sp.mode(y_m,axis=1)[0])))