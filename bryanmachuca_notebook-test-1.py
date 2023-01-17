#Arboles de decision
from sklearn.tree import DecisionTreeClassifier
#Regresion logistica
from sklearn.linear_model import LogisticRegression
#KNN
from sklearn.neighbors import KNeighborsClassifier
#SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#Niave Bayes
from sklearn.naive_bayes import GaussianNB
#Random forest 
from sklearn.ensemble import  RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

print("Librerias importadas")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.preprocessing import StandardScaler
# Z-normalize data
sc = StandardScaler()
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
# Load CSV
df = pd.read_csv("/kaggle/input/amazon-customer-service/Amazon_train.csv")
df_pred = pd.read_csv("/kaggle/input/amazon-customer-service/Amazon_test_atrib.csv")
# Datashape, Features, stat. description.
#print(df.shape)
#print("First 5 lines:")
#print(df.head(5))
#print("describe: ")
#print(df.describe())
#print("info: ")
#print(df.info())
import matplotlib.pyplot as plt
m = np.power(np.array(df["11"]),0.1)
n = np.array(df["Clase"])
#p = np.array(df["2"])
plt.scatter(m,n,alpha=0.05,color="red")
#plt.scatter(p,n,alpha=0.1)
#Pre procesamiento de la data

#Por correlacion
#df=df.drop(["1"],axis=1)#Mejora
#df=df.drop(["2"],axis=1)#Mejora
#df=df.drop(["3"],axis=1)#Mejora
#df=df.drop(["4"],axis=1)#Mejora

df=df.drop(["5"],axis=1)#Nada
#df=df.drop(["6"],axis=1)#Mejora
df=df.drop(["7"],axis=1)#Empeora
df=df.drop(["8"],axis=1)# Casi nada
df=df.drop(["9"],axis=1)#Empeora
#df=df.drop(["10"],axis=1)#Mejora
#df=df.drop(["11"],axis=1)#Empeora
df=df.drop(["12"],axis=1)#Empeora
df=df.drop(["13"],axis=1)#Nada
#df=df.drop(["14"],axis=1)#Mejora
df=df.drop(["15"],axis=1)#Nada
df=df.drop(["16"],axis=1)#Nada
df=df.drop(["17"],axis=1)#Nada
df=df.drop(["18"],axis=1)#Empeora
df=df.drop(["19"],axis=1)#Casi nada
df=df.drop(["20"],axis=1)#Nada
df=df.drop(["21"],axis=1)#Nada
df=df.drop(["22"],axis=1)#Nada
#df=df.drop(["23"],axis=1)#Mejora
df=df.drop(["24"],axis=1)#Nada
df=df.drop(["25"],axis=1)#Nada
df=df.drop(["26"],axis=1)#Nada
df=df.drop(["27"],axis=1)#Nada
df=df.drop(["28"],axis=1)#Nada
df=df.drop(["29"],axis=1)#Casi nada
df=df.drop(["30"],axis=1)#Casi nada
df=df.drop(["31"],axis=1)#Casi nada
df=df.drop(["32"],axis=1)#Casi nada
df=df.drop(["33"],axis=1)#Casi nada
df=df.drop(["34"],axis=1)#Empeora
df=df.drop(["35"],axis=1)#Casi nada
df=df.drop(["36"],axis=1)#Casi nada
df=df.drop(["37"],axis=1)#Casi nada
df=df.drop(["38"],axis=1)#Nada
df=df.drop(["39"],axis=1)#Nada
df=df.drop(["40"],axis=1)#Nada
df=df.drop(["41"],axis=1)#Nada
df=df.drop(["42"],axis=1)#Nada
df=df.drop(["43"],axis=1)#Nada
df=df.drop(["44"],axis=1)#Empeora
df=df.drop(["45"],axis=1)#Empeora
df=df.drop(["46"],axis=1)#Nada
df=df.drop(["47"],axis=1)#Empeora
df=df.drop(["48"],axis=1)#Nada
df=df.drop(["49"],axis=1)#Empeora
df=df.drop(["50"],axis=1)#Nada
df=df.drop(["51"],axis=1)#Empeora
df=df.drop(["52"],axis=1)#Empeora
df=df.drop(["53"],axis=1)#Empeora
df=df.drop(["54"],axis=1)#Empeora


#df_pred=df_pred.drop(["1"],axis=1)#Mejora
#df_pred=df_pred.drop(["2"],axis=1)#Mejora
#df_pred=df_pred.drop(["3"],axis=1)#Mejora
#df_pred=df_pred.drop(["4"],axis=1)#Mejora

df_pred=df_pred.drop(["5"],axis=1)#Nada
#df_pred=df_pred.drop(["6"],axis=1)#Mejora
df_pred=df_pred.drop(["7"],axis=1)#Empeora
df_pred=df_pred.drop(["8"],axis=1)# Casi nada
df_pred=df_pred.drop(["9"],axis=1)#Empeora
#df_pred=df_pred.drop(["10"],axis=1)#Mejora
df_pred=df_pred.drop(["11"],axis=1)#Empeora
df_pred=df_pred.drop(["12"],axis=1)#Empeora
df_pred=df_pred.drop(["13"],axis=1)#Nada
#df_pred=df_pred.drop(["14"],axis=1)#Mejora
df_pred=df_pred.drop(["15"],axis=1)#Nada
df_pred=df_pred.drop(["16"],axis=1)#Nada
df_pred=df_pred.drop(["17"],axis=1)#Nada
df_pred=df_pred.drop(["18"],axis=1)#Empeora
df_pred=df_pred.drop(["19"],axis=1)#Casi nada
df_pred=df_pred.drop(["20"],axis=1)#Nada
df_pred=df_pred.drop(["21"],axis=1)#Nada
df_pred=df_pred.drop(["22"],axis=1)#Nada
#df_pred=df_pred.drop(["23"],axis=1)#Mejora
df_pred=df_pred.drop(["24"],axis=1)#Nada
df_pred=df_pred.drop(["25"],axis=1)#Nada
df_pred=df_pred.drop(["26"],axis=1)#Nada
df_pred=df_pred.drop(["27"],axis=1)#Nada
df_pred=df_pred.drop(["28"],axis=1)#Nada
df_pred=df_pred.drop(["29"],axis=1)#Casi nada
df_pred=df_pred.drop(["30"],axis=1)#Casi nada
df_pred=df_pred.drop(["31"],axis=1)#Casi nada
df_pred=df_pred.drop(["32"],axis=1)#Casi nada
df_pred=df_pred.drop(["33"],axis=1)#Casi nada
df_pred=df_pred.drop(["34"],axis=1)#Empeora
df_pred=df_pred.drop(["35"],axis=1)#Casi nada
df_pred=df_pred.drop(["36"],axis=1)#Casi nada
df_pred=df_pred.drop(["37"],axis=1)#Casi nada
df_pred=df_pred.drop(["38"],axis=1)#Nada
df_pred=df_pred.drop(["39"],axis=1)#Nada
df_pred=df_pred.drop(["40"],axis=1)#Nada
df_pred=df_pred.drop(["41"],axis=1)#Nada
df_pred=df_pred.drop(["42"],axis=1)#Nada
df_pred=df_pred.drop(["43"],axis=1)#Nada
df_pred=df_pred.drop(["44"],axis=1)#Empeora
df_pred=df_pred.drop(["45"],axis=1)#Empeora
df_pred=df_pred.drop(["46"],axis=1)#Nada
df_pred=df_pred.drop(["47"],axis=1)#Empeora
df_pred=df_pred.drop(["48"],axis=1)#Nada
df_pred=df_pred.drop(["49"],axis=1)#Empeora
df_pred=df_pred.drop(["50"],axis=1)#Nada
df_pred=df_pred.drop(["51"],axis=1)#Empeora
df_pred=df_pred.drop(["52"],axis=1)#Empeora
df_pred=df_pred.drop(["53"],axis=1)#Empeora
df_pred=df_pred.drop(["54"],axis=1)#Empeora


df["Soles"]= np.power(df["Soles"],0.4)
df["Segundos"]= np.power(df["Segundos"],0.5)
df_pred["Soles"]= np.power(df_pred["Soles"],0.4)
df_pred["Segundos"]= np.power(df_pred["Segundos"],0.5)



#Aplicamos normalizacion para los datos mas relevantes
#df["1"]= ((df["1"])-df["1"].mean())/(df["1"].std())
#df["2"]= ((df["2"])-df["2"].mean())/(df["2"].std())
#df["3"]= ((df["3"])-df["2"].mean())/(df["3"].std())
#df["4"]= ((df["4"])-df["2"].mean())/(df["4"].std())
#df["6"]= ((df["6"])-df["6"].mean())/(df["6"].std())

#df["8"]= ((df["8"])-df["8"].mean())/(df["8"].std())
df["11"]= ((df["11"])-df["11"].mean())/(df["11"].std())
#df["10"]= ((df["10"])-df["10"].mean())/(df["10"].std())
#df["14"]= ((df["14"])-df["14"].mean())/(df["14"].std())
#df["23"]= ((df["23"])-df["23"].mean())/(df["23"].std())

#df["33"]= ((df["33"])-df["33"].mean())/(df["33"].std())

np.power(np.array(df["11"]),0.1)
print("describe: ")
print(df.describe())

import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics

# Z-normalize data
sc = StandardScaler()
Z = sc.fit_transform(df)
# Estimate the correlation matrix
R = np.dot(Z.T, Z) / df.shape[0]

sns.set(font_scale=1.0)

ticklabels = [s for s in df.columns]

hm = sns.heatmap(R,
            cbar=True,
            square=True,
            yticklabels=ticklabels,
            xticklabels=ticklabels)

plt.show()
#sns.pairplot(df)

from sklearn.model_selection import train_test_split

y = df['Clase'].values #target
X = df.drop(['Clase'],axis=1).values #features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=20, stratify=y)

print("trainset size: ", len(y_train),"\ntestset size: ", len(y_test))
# Classifier
#clf = LogisticRegression()
#clf = DecisionTreeClassifier(criterion='entropy', class_weight="balanced")
#clf =  KNeighborsClassifier(n_neighbors=3)
#clf = LinearSVC(C=7.0)
#clf = SVC()
#clf=GaussianNB()
#clf= RandomForestClassifier(criterion='entropy',class_weight="balanced_subsample"
#                           , n_estimators =25, bootstrap=False,max_features="auto")
#clf = AdaBoostClassifier(n_estimators=30)
clf = GradientBoostingClassifier(n_estimators=70)
#clf=BaggingClassifier (RandomForestClassifier(criterion='entropy',class_weight="balanced_subsample", n_estimators =15, bootstrap=False,max_features="auto"), max_samples=0.5, max_features=0.5)

#clf= SGDClassifier(loss='hinge')
# fit 
clf.fit(X_train,y_train)
# Predict
y_pred = clf.predict(X_test)
# Predicted probabilities
y_pred_prob = clf.predict_proba(X_test)
#trainset predictions
train_pred = clf.predict(X_train)
print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 

print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
print("recall score: ", recall_score(y_test,y_pred))
print("precision score: ", precision_score(y_test,y_pred))
print("f1 score: ", f1_score(y_test,y_pred))
print("accuracy score: ", accuracy_score(y_test,y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_prob[:,1]))

#train whole data
#y_train = df['Clase'].values 
#X_train = df.drop(['Clase'],axis=1).values 
X_test = df_pred.values 

#clf.fit(X_train,y_train)
# predict using test data
y_pred = clf.predict(X_test)
# Predicted probabilities
y_pred_prob = clf.predict_proba(X_test)
# Final Submission
Id=np.arange(1,393)
Id= Id.flatten()
my_submission = pd.DataFrame({'ID': Id, 'Clase': y_pred})

print(my_submission)
# you could use any filename. We choose submission here
my_submission.to_csv('submission_Sample_final.csv', index=False)