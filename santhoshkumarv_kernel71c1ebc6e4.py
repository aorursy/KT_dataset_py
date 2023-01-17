# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")

df.head()
df.info()
df.isnull().sum()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
plt.figure(figsize=(18,10))

sns.heatmap(df.corr(), annot=True, cmap='cool')

plt.show()
sns.countplot(df.target, palette=['green', 'red'])

plt.title("[0] == Not Disease, [1] == Disease");
plt.figure(figsize=(18, 10))

sns.countplot(x='age', hue='target', data=df, palette=['#1CA53B', 'red'])

plt.legend(["Haven't Disease", "Have Disease"])

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(12,12))

fs = ['cp', 'fbs', 'restecg','exang', 'slope', 'ca']

for i, axi in enumerate(axes.flat):

    sns.countplot(x=fs[i], hue='target', data=df, palette='bwr', ax=axi) 

    axi.set(ylabel='Frequency')

    axi.legend(["Haven't Disease", "Have Disease"])
plt.figure(figsize=(8,6))

sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='chol',y='thalach',data=df,hue='target')

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
from sklearn.preprocessing import StandardScaler



# Import tools

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc
# Define our feature and labels

X = df.drop(['target'], axis=1).values

y = df['target'].values
scale = StandardScaler()

X = scale.fit_transform(X)
class Model:

    def __init__(self, model, X, y):

        self.model = model

        self.X = X

        self.y = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)

        

        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

        

    def model_str(self):

        return str(self.model.__class__.__name__)

    

    def crossValScore(self, cv=5):

        print(self.model_str() + "\n" + "="*60)

        scores = ["accuracy", "precision", "recall", "roc_auc"]

        for score in scores:  

            cv_acc = cross_val_score(self.model, 

                                     self.X_train, 

                                     self.y_train, 

                                     cv=cv, 

                                     scoring=score).mean()

            

            print("Model " + score + " : " + "%.3f" % cv_acc)

        

    def accuracy(self):

        accuarcy = accuracy_score(self.y_test, self.y_pred)

        print(self.model_str() + " Model " + "Accuracy is: ")

        return accuarcy

        

    def confusionMatrix(self):        

        plt.figure(figsize=(6, 6))

        mat = confusion_matrix(self.y_test, self.y_pred)

        sns.heatmap(mat.T, square=True, 

                    annot=True, 

                    cbar=False, 

                    xticklabels=["Haven't Disease", "Have Disease"], 

                    yticklabels=["Haven't Disease", "Have Disease"])

        

        plt.title(self.model_str() + " Confusion Matrix")

        plt.xlabel('Predicted Values')

        plt.ylabel('True Values');

        plt.show();

        

    def classificationReport(self):

        print(self.model_str() + " Classification Report" + "\n" + "="*60)

        print(classification_report(self.y_test, 

                                    self.y_pred, 

                                    target_names=['Non Disease', 'Disease']))

    

    def rocCurve(self):

        y_prob = self.model.predict_proba(self.X_test)[:,1]

        fpr, tpr, thr = roc_curve(self.y_test, y_prob)

        lw = 2

        plt.figure(figsize=(10, 8))

        plt.plot(fpr, tpr, 

                 color='darkorange', 

                 lw=lw, 

                 label="Curve Area = %0.3f" % auc(fpr, tpr))

        plt.plot([0, 1], [0, 1], color='green', 

                 lw=lw, linestyle='--')

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title(self.model_str() + ' Receiver Operating Characteristic Plot')

        plt.legend(loc="lower right")

        plt.show()
from sklearn.ensemble import RandomForestClassifier



clf1 = Model(model=RandomForestClassifier(n_estimators=1000), X=X, y=y)
clf1.crossValScore(cv=10)
clf1.accuracy()
clf1.confusionMatrix()
import xgboost as xgb

clf2 = Model(model=xgb.XGBClassifier(), X=X, y=y)

clf2.crossValScore(cv=10)
clf2.accuracy()
clf2.confusionMatrix()
clf1.classificationReport()
clf2.classificationReport()
clf1.rocCurve()
clf2.rocCurve()
import warnings

warnings.simplefilter("ignore")



from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import QuantileTransformer



lr = LogisticRegression(C=10, n_jobs=-1)

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), lr)



pip = Model(model=pipeline, X=X, y=y)
pip.crossValScore()
pip.accuracy()
pip.confusionMatrix()
from sklearn.neighbors import KNeighborsClassifier



knn = Model(model=KNeighborsClassifier(n_neighbors=100), X=X, y=y)
knn.crossValScore()
knn.accuracy()
knn.confusionMatrix()
models = [clf1, clf2,Ann,pip, knn]

names = []

accs = []

for model in models:

    accs.append(model.accuracy())

    names.append(model.model_str())
sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,1.2,0.1))

plt.ylabel("Accuracy")

plt.xlabel("Algorithms")

sns.barplot(x=names, y=accs)

plt.savefig('models_accuracy.png')

plt.show()
from sklearn.neural_network import MLPClassifier

Ann = Model(model=MLPClassifier(hidden_layer_sizes=(300,150)), X=X, y=y)

Ann.crossValScore()
Ann.accuracy()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pl_mlp=Pipeline(steps=[('scaler',StandardScaler()),('mil_ann',MLPClassifier(hidden_layer_sizes=(1275,637)))])

scores=cross_val_score(pl_mlp,X_train,y_train,cv=10,scoring='accuracy')

print('ANN:',scores.mean())
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pl_xgb=Pipeline(steps=[('svm',xgb.XGBClassifier(objective='multi:softmax',num_class=2))])

scores=cross_val_score(pl_xgb,X_train,y_train,cv=10,scoring='accuracy')

print('xgb:',scores.mean())