import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
dataset = datasets.load_breast_cancer()
df = pd.DataFrame(np.c_[dataset['target'],dataset['data']], columns=np.r_[['target'],dataset['feature_names']])
df.head()
df.isnull().sum()
n_samples, n_features = df.shape
train_test_ratio = 0.7
n_training_samples = int(n_samples * 0.7)
n_test_samples = n_samples - n_training_samples
dfShuffled = df.sample(frac=1, random_state=42)
dfTrain = dfShuffled.iloc[0:n_training_samples,:]
dfTest = dfShuffled.iloc[n_training_samples:n_samples,:]
dfTrainX = dfShuffled.iloc[0:n_training_samples,1:n_features]
dfTestX = dfShuffled.iloc[n_training_samples:n_samples,1:n_features]
dfTrainY = dfShuffled['target'].iloc[0:n_training_samples]
dfTestY = dfShuffled['target'].iloc[n_training_samples:n_samples]
X_train = dfTrainX.values.astype(float)
X_test = dfTestX.values.astype(float)
y_train = dfTrainY.values.astype(int)
y_test = dfTestY.values.astype(int)
X_train.shape , X_test.shape
y_train.sum() + y_test.sum() == (df['target'] == 1).sum()
abs(X_train[:,0].sum() + X_test[:,0].sum() - df['mean radius'].sum()) < 0.0000000001
def plot_data_np(X1, X2, Y, x1Label, x2Label, ax):
    
    title = x1Label + ' vs ' + x2Label
    
    x1Min = X1.min()
    x1Max = X1.max()
    x2Min = X2.min()
    x2Max = X2.max()
    
    ax.plot(X1[Y==0], X2[Y==0], "bs")    
    ax.plot(X1[Y==1], X2[Y==1], "r^")
    ax.axis([x1Min, x1Max, x2Min, x2Max])
    ax.set_xlabel(x1Label)
    ax.set_ylabel(x2Label)
    ax.set_title(title)
    
def plot_data(df, x1Label, x2Label, ax):
    
    plot_data_np(df[x1Label].values, df[x2Label].values, (df['target'].values == 1)*1, x1Label, x2Label, ax)
plt.figure(1)
fig, axarr = plt.subplots(2, 2)
plot_data(df, 'mean radius', 'mean perimeter', axarr[0, 0])
plot_data(df, 'mean area', 'mean symmetry', axarr[0, 1])
plot_data(df, 'mean compactness', 'mean concavity', axarr[1, 0])
plot_data(df, 'mean radius', 'mean concave points', axarr[1, 1])
fig.set_size_inches(15, 15)
plt.show()
dfTrainX.append(dfTestX)['mean perimeter'].describe()
radii = df['mean radius'].values
mean = df['mean radius'].mean()
std = df['mean radius'].std()
minRadius, maxRadius = np.min(radii) , np.max(radii)
import random

actual_samples = df['mean radius'].values
norm_samples = [random.gauss(mean, std) for _ in range(n_samples)]

plt.figure(2)
plt.hist(actual_samples, 50, alpha=0.75, label='Actual', density=False, color='b')
plt.hist(norm_samples, 50, alpha=0.25, label='Normal Distribution', density=False, color='g')
plt.legend(loc='upper right')
plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.show()
df['mean radius'].skew()
stats.normaltest(radii)
beta = std ** 2 / mean
alpha = mean / beta
gamma_samples = [random.gammavariate(alpha, beta) for _ in range(n_samples)]
plt.figure(3)
plt.hist(actual_samples, 50, alpha=0.75, label='Actual', density=False, color='b')
plt.hist(gamma_samples, 50, alpha=0.25, label='Gamma Distribution', density=False, color='g')
plt.legend(loc='upper right')
plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.show()
benign_samples = df['mean radius'][df['target'] == 0]
malignant_samples = df['mean radius'][df['target'] == 1]
plt.figure(3)
plt.hist(benign_samples, 50, alpha=0.75, label='Benign', density=False, color='g')
plt.hist(malignant_samples, 50, alpha=0.25, label='Malignant', density=False, color='r')
plt.legend(loc='upper right')
plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.show()
benign_samples.skew()
stats.normaltest(benign_samples)
malignant_samples.skew()
stats.normaltest(malignant_samples)
y_pred_dumb = np.random.randint(2, size=y_test.shape[0])
from sklearn import metrics
metrics.f1_score(y_test, y_pred_dumb)
C = 20
from sklearn.linear_model import LogisticRegression
featureIndices = (0,7) # Radius and Concave Points
logisticClassifier = LogisticRegression(C=C, random_state=42)
logisticClassifier.fit(X_train[:,featureIndices], y_train)
y_pred_logistic = logisticClassifier.predict(X_test[:,featureIndices])
metrics.f1_score(y_test, y_pred_logistic)
def plot_boundary_np(w0, w1, w2, X1, X2, Y, ax):
       
    x1Min = X1.min()
    x1Max = X1.max()
    x2Min = X2.min()
    x2Max = X2.max()
    x1vals = np.linspace(x1Min, x1Max, 100)                  
    x2vals = -(w1 * x1vals + w0) / w2
    ax.axis([x1Min, x1Max, x2Min, x2Max])
    ax.plot(x1vals, x2vals, "k-")
w0 = logisticClassifier.intercept_
w = logisticClassifier.coef_[0]
X1 = X_test[:,featureIndices[0]]
X2 = X_test[:,featureIndices[1]]
Y = y_test
x1Label = dfTrainX.columns.values[featureIndices[0]]
x2Label = dfTrainX.columns.values[featureIndices[1]]

plt.figure(4)
fig, axarr = plt.subplots(1, 1)
plot_data_np(X1, X2, Y, x1Label, x2Label, axarr)
plot_boundary_np(w0, w[0], w[1], X1, X2, Y, axarr)
fig.set_size_inches(5, 5)
plt.show()
logisticClassifier = LogisticRegression(C=C, random_state=42)
logisticClassifier.fit(X_train, y_train)
y_pred_logistic = logisticClassifier.predict(X_test)
metrics.f1_score(y_test, y_pred_logistic)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
linearSVMPipeline = Pipeline([("scaler", StandardScaler()),('linearSVC',SVC(kernel="linear", C=1/C))])
linearSVMPipeline.fit(X_train[:,featureIndices], y_train)
y_pred_linear_SVM = linearSVMPipeline.predict(X_test[:,featureIndices])
metrics.f1_score(y_test, y_pred_linear_SVM)
def plot_svc_decision_boundary_two_features(scaler, classifier, X_test, Y_test, x1Label, x2Label, ax):
    
    X_test_scaled = scaler.fit_transform(X_test)
    X1 = X_test_scaled[:,0]
    X2 = X_test_scaled[:,1]
    x1min = X1.min()
    x1max = X1.max()
    x2min = X2.min()
    x2max = X2.max()
    Y = Y_test
    
    # Plot the data points first
    ax.plot(X1[Y==0], X2[Y==0], "bs")    
    ax.plot(X1[Y==1], X2[Y==1], "r^")
    
    # Now plot decision boundary
    w = classifier.coef_[0]
    w0 = classifier.intercept_[0]
    w1 = w[0]
    w2 = w[1]

    x1 = np.linspace(x1min, x1max, 200)
    x2 = -(w1 * x1 + w0) / w2
    gutter_up = (1 - w1 * x1 - w0) / w2
    gutter_down = (-1 - w1 * x1 - w0) / w2
    
    ax.plot(x1, x2, "k-", linewidth=2)
    ax.plot(x1, gutter_up, "k--", linewidth=2)
    ax.plot(x1, gutter_down, "k--", linewidth=2)
    
    ax.axis([x1min, x1max, x2min, x2max])
    ax.set_xlabel(x1Label)
    ax.set_ylabel(x2Label)
    ax.set_title(x1Label + ' vs ' + x2Label)
x1Label = dfTrainX.columns.values[featureIndices[0]]
x2Label = dfTrainX.columns.values[featureIndices[1]]
plt.figure(5)
fig, ax = plt.subplots(1, 1)
plot_svc_decision_boundary_two_features(linearSVMPipeline.named_steps['scaler'], 
                                        linearSVMPipeline.named_steps['linearSVC'], 
                                        X_test[:,featureIndices], y_test, x1Label, x2Label, ax)
linearSVMPipeline = Pipeline([("scaler", StandardScaler()),('linearSVC',SVC(kernel="linear", C=1/C, random_state=42))])
linearSVMPipeline.fit(X_train, y_train)
y_pred_linear_SVM = linearSVMPipeline.predict(X_test)
metrics.f1_score(y_test, y_pred_linear_SVM)
from sklearn.preprocessing import PolynomialFeatures
kernelSVMPipeline = Pipeline([("scaler", StandardScaler())
                                ,('KernelSVC',SVC(kernel="poly", C=1/C, degree=1, random_state=42))])
kernelSVMPipeline.fit(X_train, y_train)
y_pred_kernel_SVM = kernelSVMPipeline.predict(X_test)
metrics.f1_score(y_test, y_pred_kernel_SVM)
kernelSVMPipeline = Pipeline([("scaler", StandardScaler())
                                ,('KernelSVC',SVC(kernel="poly", C=1/C, degree=2, random_state=42))])
kernelSVMPipeline.fit(X_train, y_train)
y_pred_kernel_SVM = kernelSVMPipeline.predict(X_test)
metrics.f1_score(y_test, y_pred_kernel_SVM)