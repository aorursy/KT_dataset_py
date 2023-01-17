# Scikit-learn 

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import roc_curve, classification_report

# Other libraries

import numpy as np

import matplotlib.pyplot as plt
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
# quadratic function

def f2fun(x):

    return -1.3 + 0.3 * x**2



# regular grid converted to features matrix

x0, y0 = np.meshgrid(np.linspace(-5.0,5.0,num=30), np.linspace(-2.0,2.8,num=20))



# Convert to feature matrix

X = np.concatenate((np.asanyarray(x0).reshape(x0.size,1),np.asanyarray(y0).reshape(y0.size,1)),axis=1)



# set labels (below quadratic line to 1)

y = np.zeros(y0.size);

y[f2fun(X[:,0]) < X[:,1]] = 1

y = LabelEncoder().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,train_size=0.67, test_size=0.33);
def plotsynth(X,y,t="Synthetic data input"):

    plt.figure(figsize=(6,4)) 

    plt.scatter(X[:,0],X[:,1],c=y,marker=".");

    plt.title(t);

    plt.axis("equal")
plotsynth(X,y);

plt.scatter(X_train[:,0],X_train[:,1],c=y_train,marker="s");
log_reg = GridSearchCV(LogisticRegression(penalty ="l2",max_iter=10000),

                {"C":[10**i for i in range(-5,6)],

              "solver":["newton-cg", "sag", "lbfgs"]},cv=3,scoring="balanced_accuracy");
log_reg.fit(X_train,y_train);

# Compute ROC curve (will be plotted later)

fpr1,tpr1,_ = roc_curve(y_train,log_reg.decision_function(X_train))
print("Train model (balanced) accuracy score = {:.3f}".format(log_reg.best_score_))

print("Test  model (balanced) accuracy score = {:.3f}".format(log_reg.score(X_test,y_test)))
plotsynth(X,y,t="Model output")

plt.scatter(X_train[:,0],X_train[:,1],c=log_reg.predict(X_train),marker="s");
from sklearn.preprocessing import PolynomialFeatures
Xp = PolynomialFeatures(degree=2).fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(Xp,y,stratify=y,train_size=0.66);
log_reg.fit(X_train,y_train);
print("Train model (balanced) accuracy score = {:.3f}".format(log_reg.best_score_))

print("Test  model (balanced) accuracy score = {:.3f}".format(log_reg.score(X_test,y_test)))
plotsynth(X,y,t="Model output")

plt.scatter(X_train[:,1],X_train[:,2],c=log_reg.predict(X_train),marker="s");
fpr2,tpr2, _ = roc_curve(y_train,log_reg.decision_function(X_train))
plt.figure(figsize=(3,3))

plt.plot(fpr1,tpr1,"b-",fpr2,tpr2,"r-");

plt.legend(["simple","polonomial"]);

plt.axis("equal");

plt.title("ROC for simple and polynomial model");
log_reg.estimator.fit(X_train,y_train).coef_
iris = datasets.load_iris()

X = iris.data;

y = LabelEncoder().fit_transform(iris.target);
scale = StandardScaler(with_mean=True,with_std=True);

X = scale.fit_transform(X);
X = PCA(n_components=0.99).fit_transform(X)
[X_train,X_test,y_train,y_test] = train_test_split(X,y,stratify=y,random_state=123)
lr = LogisticRegression(multi_class = "auto",penalty ="l2",max_iter=10000);

parameters = {"C":[10**i for i in range(-5,6)],

              "solver":["newton-cg", "sag", "lbfgs"]}
lro = GridSearchCV(lr,parameters,cv=5,scoring="accuracy");
lro.fit(X_train,y_train);
print("Model accuracy score for training = {:.3f}".format(lro.best_score_))
print("Classification report:\n",classification_report(y_test,lro.predict(X_test)))