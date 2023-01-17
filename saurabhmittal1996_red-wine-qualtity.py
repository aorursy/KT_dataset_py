import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data.shape
data.head()
data.info()
data.corr()['quality']
sns.heatmap(data.corr())
data[data==0].sum()
data.isnull().sum()
data.nunique()
sns.pairplot(data)
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol',]
for i in columns:
    sns.scatterplot(data = data[i])
    plt.xlabel(i)
    plt.show()
li = list(data['chlorides'].sort_values()[-4:].index)
data['chlorides'][li] = int(data.drop(li)['chlorides'].mode())
li = list(data['total sulfur dioxide'].sort_values()[-2:].index)
data['total sulfur dioxide'][li] = int(data.drop(li)['total sulfur dioxide'].mode())
li = list(data['sulphates'].sort_values()[-7:].index)
data['sulphates'][li] = int(data.drop(li)['sulphates'].mode())
li = list(data['residual sugar'].sort_values()[-11:].index)
data['residual sugar'][li] = int(data.drop(li)['residual sugar'].mean())
for i in columns:
    sns.scatterplot(data = data[i])
    plt.xlabel(i)
    plt.show()
for i in columns:
    sns.barplot(x='quality', y= i, data=data)
    plt.show()
for i in columns:    
    sns.boxplot(x='quality', y= i, data=data)
    plt.show()
def quality_index(x):
    if x > 6:
        return 1
    else:
        return 0
data['quality'] = data['quality'].apply(quality_index)
for i in columns:    
    sns.boxplot(x='quality', y= i, data=data)
    plt.show()
data.head()
data.quality.value_counts()
#Now seperate the dataset as response variable and feature variabes
X = data.drop('quality', axis = 1)
y = data['quality']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(X)
plt.figure(figsize=(5,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
#Train and Test splitting of data 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
lr = LogisticRegression()

svc = SVC(C=1.2, kernel='rbf')

rfc = RandomForestClassifier()

dtc = DecisionTreeClassifier()

knn = KNeighborsClassifier()

xgb = XGBClassifier()
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score
from sklearn.metrics import recall_score,f1_score, confusion_matrix, roc_curve, auc
def train_model(model):
    # Checking accuracy
    model = model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('accuracy_score',accuracy_score(y_test, pred)*100)
    print('precision_score',precision_score(y_test, pred)*100)
    print('recall_score',recall_score(y_test, pred)*100)
    print('f1_score',f1_score(y_test, pred)*100)
    print('roc_auc_score',roc_auc_score(y_test, pred)*100)
    # confusion matrix
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    fpr, tpr, threshold = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)*100

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
train_model(lr)
train_model(svc)
train_model(dtc)
train_model(knn)
train_model(xgb)
train_model(rfc)
