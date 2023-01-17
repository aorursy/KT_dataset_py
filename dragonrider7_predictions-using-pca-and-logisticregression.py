import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn.datasets
from sklearn.datasets import load_iris
iris = load_iris()
iris
#Create a concatenated dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
display(X.head())
X.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
display(X.head())
target = df['target']
target.head()
X.dropna(how='all', inplace=True)
X.head()
sns.scatterplot(x = X.sepal_length, y = X.sepal_width, style = df.target )
from sklearn.preprocessing import scale
x_train = scale(X)
#Find covariance matrix
covariancematrix = np.cov(x_train.T)
covariancematrix
eigenvalues, eigenvectors = np.linalg.eig(covariancematrix)
display(eigenvalues, eigenvectors)
#Alternatively, use Singular Value Decomposition (SVD)
eigenvec_svd, s, v = np.linalg.svd(x_train.T)
display(eigenvec_svd)
display(eigenvalues)
variance_accounted = []
for i in eigenvalues:
    va = (i/(eigenvalues.sum())*100)
    variance_accounted.append(va)
display(variance_accounted)
cumulative_variance = np.cumsum(variance_accounted)
cumulative_variance
sns.lineplot(x = [1,2,3,4], y = cumulative_variance);
plt.xlabel("No. of components")
plt.ylabel("Cumulative variance")
plt.title("Variance vs No. of components")
plt.show()
#Project data onto a lower dimensional plane
proj_vector = (eigenvectors.T[:][:])[:2].T
proj_vector
x_pca = np.dot(x_train, proj_vector)
#split the data set into train and test sets
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x_pca, target, test_size=0.2)
xtrain,xtest, ytrain, ytest = train_test_split(x_train, target, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model_pca = LogisticRegression()
model_pca.fit(xTrain, yTrain)
y_pred = model_pca.predict(xTest)
y_pred
model_original = LogisticRegression()
model_original.fit(xtrain, ytrain)
y_pred_original = model_original.predict(xtest)
y_pred_original
from sklearn.metrics import confusion_matrix
cm_pca = confusion_matrix(y_pred, yTest)
cm_pca
cm_original = confusion_matrix(y_pred_original, ytest)
cm_original
#Confusion Matrix showing percentages
print('Confusion Matrix for values predicted after selecting 2 components using principle component analysis' ,sns.heatmap((cm_pca/np.sum(cm_pca))*100, annot=True, cmap="GnBu"))

print('Confusion Matrix for values predicted using all 4 components from original standardised data', sns.heatmap((cm_original/np.sum(cm_original))*100, annot = True, cmap="Blues"))
from sklearn.metrics import classification_report
print('Classification Report for PCA data')
p = np.asarray(yTest)
p1 = pd.DataFrame(p, columns =['Actual'])
p2 = pd.DataFrame(y_pred, columns = ['Predictions_pca'])
pred = pd.concat([p1,p2], axis = 1)
print(classification_report(pred['Actual'], pred['Predictions_pca']))
print('Classification Report for values predicted using all 4 components')
q = np.asarray(ytest)
q1 = pd.DataFrame(q, columns=['Actual'])
q2 = pd.DataFrame(y_pred_original, columns=['Predictions_without_pca'])
pred_2 = pd.concat([q1,q2], axis=1)
print(classification_report(pred_2['Actual'], pred_2['Predictions_without_pca']))
