import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score#accuracy ölçmek için
from sklearn.naive_bayes import GaussianNB#one of our classifiers
from sklearn.neural_network import MLPClassifier#neural network
from sklearn.model_selection import train_test_split#datayı bölmek için
from sklearn.ensemble import RandomForestClassifier#another classifier
from sklearn.preprocessing import StandardScaler#datayı normalize etmek için(neural network daha iyi çalışır)
import matplotlib.pyplot as plt#grafik çizdirmek için
from sklearn import tree#ANOTHER classifier
import seaborn as sns#grafik çizdirmek için
from sklearn.decomposition import PCA#dimension reduction için(kullanmadık)
data=pd.read_csv('../input/train.csv')#train datası
test=pd.read_csv('../input/test.csv')#test datası
y=data.label#etiketler
x=data.drop('label',axis=1)#label kısmı drop edildi
x_test=test#test datası
print(data.isnull().any().describe())
print(test.isnull().any().describe())
g=sns.countplot(y)
plt.show(g)
pca=PCA(n_components=2, whiten=True)
pca.fit(x)
X_pca=pca.transform(x)
plt.figure(1, figsize=(12,8))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=10,cmap=plt.get_cmap('jet',10))
plt.colorbar()
pca=PCA()
pca.fit(x)
plt.figure(1,figsize=(12,8))
plt.xticks(np.arange(0, 800, 30.0))
plt.plot(pca.explained_variance_,linewidth=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_components=35
pca=PCA(n_components=n_components, whiten=True)
pca.fit(x_train)
X_train_pca=pd.DataFrame(pca.transform(x_train))
X_test_pca=pd.DataFrame(pca.transform(x_test))

clf=SVC(C=10, gamma=0.01, kernel="rbf")
clf.fit(X_train_pca, y_train)
competion_dataset=pd.read_csv("../input/test.csv",dtype="uint8")
competion_dataset=competion_dataset.values
competion_dataset=competion_dataset[:,0:]
competion_dataset_pca=pca.transform(competion_dataset)
y_pred2=clf.predict(competion_dataset_pca)
file_name="x_pca_{}_svc_mnist.csv".format(n_components)

model = MLPClassifier(solver='lbfgs', activation='logistic', learning_rate='adaptive')
clf = tree.DecisionTreeClassifier()
model1 = RandomForestClassifier(random_state=43)
gnb = GaussianNB()
model.fit(X_train_pca, y_train)
model1.fit(X_train_pca, y_train)
clf.fit(X_train_pca, y_train)
gnb.fit(X_train_pca, y_train)
y_pred1=clf.predict(X_test_pca)
y_pred2=model.predict(X_test_pca)
y_pred3=model1.predict(X_test_pca)
y_pred4=clf.predict(X_test_pca)
y_pred5=gnb.predict(X_test_pca)

print('Accuracy for support vector machines is: ',(accuracy_score(y_pred1, y_test)))
print('neural network accuracy:',accuracy_score(y_pred2, y_test))
print('random forest:' ,accuracy_score(y_pred3, y_test))
print('decision tree:' ,accuracy_score(y_pred4, y_test))
print('gaussian naive bayes:' ,accuracy_score(y_pred5, y_test))
results = pd.Series(y_pred1,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
#y_pred2=y_pred2.astype(int)
submission.to_csv(file_name,index=False)