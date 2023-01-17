

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns
Data = pd.read_csv("../input/heart-disease-uci/heart.csv")

Data.head()
from sklearn.model_selection import train_test_split

train, test =train_test_split(Data,test_size=0.2,random_state=0)

print("train : {} , test : {}".format(train.shape, test.shape))
y_train = train["target"]

X_train = train.drop("target",axis=1)

y_test = test["target"]

X_test = test.drop('target',axis=1)
X_train.isnull().sum()
X_test.isnull().sum()
plt.figure(figsize=(20,20))

i=1

for elt in X_train.columns:

    plt.subplot(4,4,i)

    X_train[elt].hist(bins=20)

    plt.xlabel(elt)

    i+=1

plt.show()
features=["sex","cp","fbs","restecg", "exang","slope", 'ca', 'thal']

i=1

plt.figure(figsize=(10,20))

for feature in features:

    plt.subplot(4,2,i)

    vals = np.sort(X_train[feature].unique()).tolist()

    for v in vals:

        s_b = X_train[X_train[feature]==v]

        s_b2 = y_train[s_b.index]

        a=s_b2[s_b2==1].shape[0]

        b=s_b2[s_b2==0].shape[0]

        plt.bar(v - 0.125, a, color = 'r', width = 0.25)

        plt.bar(v + 0.125, b, color = 'g', width = 0.25)

    plt.xlabel(feature)

    plt.legend(["Has Heart Disease","Healthy"])

    i+=1

plt.show()
c_features=["age", "trestbps", "chol", "thalach", "oldpeak"]

plt.figure(figsize=(10,25))

i=1

for idx,elt in enumerate(c_features):

    f= idx + 1

    while f < len(c_features):

        plt.subplot(5,2,i)

        sns.scatterplot(X_train[c_features[f]],X_train[elt],hue = y_train,

                 palette=['green','red'],legend='full')

        plt.xlabel(c_features[f])

        plt.ylabel(elt)

        plt.legend()

        f+=1

        i+=1

plt.show()
from scipy.stats import skew
for elt in c_features: 

    print(elt, skew(X_train[elt]))
features_to_str=["cp","restecg", "slope", 'ca', 'thal']

for elt in features_to_str:

    X_train[elt] = X_train[elt].apply(str)

    X_test[elt] = X_test[elt].apply(str)
X_train = pd.get_dummies(data=X_train, prefix=features_to_str, 

                        prefix_sep='=', columns=features_to_str)

X_train
X_test = pd.get_dummies(data=X_test, prefix=features_to_str, 

                        prefix_sep='=', columns=features_to_str)

X_test
for elt in X_train.columns : 

    if elt not in X_test.columns: 

        X_test[elt]= np.zeros(len(X_test))

        print(elt)

print("X_train: {}, X_test: {}".format(X_train.shape, X_test.shape))
X_test = X_test[X_train.columns]

X_test
features_to_log = ["chol", "oldpeak"]

for elt in features_to_log:

    X_train[elt] = np.log(1+ X_train[elt])

    X_test[elt] = np.log(1+ X_test[elt])
from sklearn.preprocessing import StandardScaler

data_scaler = StandardScaler()

data_scaler.fit(X_train[c_features])

X_train[c_features] = data_scaler.transform(X_train[c_features])

X_test[c_features] = data_scaler.transform(X_test[c_features])

X_train.head()
X_test.head()
from sklearn.decomposition import PCA
pca = PCA(n_components=None)

pca.fit(X_train)

pca_components = pd.DataFrame(pca.explained_variance_ratio_,columns=['Data Variance per Component'])

pca_components['Total Captured Variance'] = pca_components['Data Variance per Component'].cumsum()
pca_components
pca= PCA(n_components=10)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel='rbf', n_components = 10)

X_train_kpca = kpca.fit_transform(X_train)

X_test_kpca = kpca.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)

lda.fit(X_train,y_train)

X_train_lda= lda.transform(X_train)

X_test_lda = lda.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score, recall_score, make_scorer
def CAP_performance(y_true,y_pred,p=0.5,plot=False):

    df =pd.DataFrame()

    df["GT"] = y_true

    df["Predictions"]=y_pred

    positive_percentage = df["GT"].sum() / len(df)

    df.sort_values(by=["Predictions"],axis=0,ascending=False, inplace=True)

    df.reset_index(inplace=True,drop=True)

    df['CumPredictions'] = df["GT"].cumsum()

    df['CumAcc']=df['CumPredictions'] / len(df)

    idx = int(np.trunc(p*len(df)))

    CAP=(df['CumAcc'].values[idx]+df['CumAcc'].values[idx+1])/2

    if plot: 

        plt.figure()

        #random line : 

        plt.plot([0,len(df)],[0,100],color='black',label="Random Model")

        #perfect model : 

        plt.plot([0,df['GT'].sum(),len(df)],[0,100,100],color='green',label="Cristal Ball Model")

        #our model:

        x=list(range(len(df)+1))

        y= [0]+ (df['CumAcc']*100/positive_percentage).values.tolist()

        plt.plot(x,y,color='red',label="Model Performance")

        plt.plot([p*len(df),p*len(df),0],[0,CAP*100/positive_percentage,CAP*100/positive_percentage],

                 color='blue',label='{} %'.format(CAP*100/positive_percentage))

        plt.xlim(0,len(df)+1)

        plt.ylim(0,101)

        plt.legend()

        plt.show()

    return CAP/positive_percentage
performances = {}

performances["Method"]=[]

performances["Healthy_Recall"]=[]

performances["Disease_Recall"]=[]

performances["CAP"]=[]
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=0)

LR_pca = LogisticRegression(random_state=0)

LR_kpca = LogisticRegression(random_state=0)

LR_lda = LogisticRegression(random_state=0)
LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("Logistic Regression")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
LR_pca.fit(X_train_pca,y_train)

y_pred = LR_pca.predict(X_test_pca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("Logistic Regression PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
LR_kpca.fit(X_train_kpca,y_train)

y_pred = LR_kpca.predict(X_test_kpca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("Logistic Regression K_PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
LR_lda.fit(X_train_lda,y_train)

y_pred = LR_lda.predict(X_test_lda)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("Logistic Regression LDA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
from sklearn.svm import SVC

SVM = SVC(C=0.5,gamma=0.1,random_state=0)

SVM_pca = SVC(C=0.5,gamma=0.1,random_state=0)

SVM_kpca = SVC(C=0.5,gamma=0.1,random_state=0)

SVM_lda = SVC(C=0.5,gamma=0.1,random_state=0)
SVM.fit(X_train,y_train)

y_pred = SVM.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("SVM")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
SVM_pca.fit(X_train_pca,y_train)

y_pred = SVM_pca.predict(X_test_pca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("SVM PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
SVM_kpca.fit(X_train_kpca,y_train)

y_pred = SVM_kpca.predict(X_test_kpca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("SVM K_PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
SVM_lda.fit(X_train_lda,y_train)

y_pred = SVM_lda.predict(X_test_lda)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("SVM LDA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
from sklearn.neighbors import KNeighborsClassifier

KNN= KNeighborsClassifier(n_neighbors=5)

KNN_pca= KNeighborsClassifier(n_neighbors=5)

KNN_kpca= KNeighborsClassifier(n_neighbors=5)

KNN_lda= KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,y_train)

y_pred = KNN.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("KNN 5")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
KNN_pca.fit(X_train_pca,y_train)

y_pred = KNN_pca.predict(X_test_pca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("KNN 5 PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
KNN_kpca.fit(X_train_kpca,y_train)

y_pred = KNN_kpca.predict(X_test_kpca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("KNN 5 K_PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
KNN_lda.fit(X_train_lda,y_train)

y_pred = KNN_lda.predict(X_test_lda)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("KNN 5 LDA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(criterion='entropy',random_state=0)

DTC_pca = DecisionTreeClassifier(criterion='entropy',random_state=0)

DTC_kpca = DecisionTreeClassifier(criterion='entropy',random_state=0)

DTC_lda = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTC.fit(X_train,y_train)

y_pred = DTC.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("DTC")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
DTC_pca.fit(X_train_pca,y_train)

y_pred = DTC_pca.predict(X_test_pca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("DTC PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
DTC_kpca.fit(X_train_kpca,y_train)

y_pred = DTC_kpca.predict(X_test_kpca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("DTC K_PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
DTC_lda.fit(X_train_lda,y_train)

y_pred = DTC_lda.predict(X_test_lda)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("DTC LDA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)

RFC_pca = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)

RFC_kpca = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)

RFC_lda = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)
RFC.fit(X_train,y_train)

y_pred = RFC.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("RF")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
RFC_pca.fit(X_train_pca,y_train)

y_pred = RFC_pca.predict(X_test_pca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("RF PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
RFC_kpca.fit(X_train_kpca,y_train)

y_pred = RFC_kpca.predict(X_test_kpca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("RF K_PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
RFC_lda.fit(X_train_lda,y_train)

y_pred = RFC_lda.predict(X_test_lda)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("RF LDA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
from xgboost import XGBClassifier

XGC = XGBClassifier(n_estimators = 150 , random_state=0)

XGC_pca = XGBClassifier(n_estimators = 150 , random_state=0)

XGC_kpca = XGBClassifier(n_estimators = 150 , random_state=0)

XGC_lda = XGBClassifier(n_estimators = 150 , random_state=0)
XGC.fit(X_train,y_train)

y_pred = XGC.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("XGB")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
XGC_pca.fit(X_train_pca,y_train)

y_pred = XGC_pca.predict(X_test_pca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("XGB PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
XGC_kpca.fit(X_train_kpca,y_train)

y_pred = XGC_kpca.predict(X_test_kpca)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("XGB K_PCA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
XGC_lda.fit(X_train_lda,y_train)

y_pred = XGC_lda.predict(X_test_lda)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

cap = CAP_performance(y_test,y_pred,plot=True)
performances["Method"].append("XGB LDA")

performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])

performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])

performances["CAP"].append(cap)
performances_df = pd.DataFrame(performances)

performances_df
plt.figure(figsize=(5,15))

plt.subplot(3,1,1)

plt.plot(performances_df["Healthy_Recall"])

plt.xticks(performances_df.index,performances_df["Method"],rotation='vertical')

plt.xlabel("Models")

plt.ylabel("Recall of Healthy Data Points")

plt.grid()

plt.subplot(3,1,2)

plt.plot(performances_df["Disease_Recall"])

plt.xticks(performances_df.index,performances_df["Method"],rotation='vertical')

plt.xlabel("Models")

plt.ylabel("Recall of Diseased Data Points")

plt.grid()

plt.subplot(3,1,3)

plt.plot(performances_df["CAP"])

plt.xticks(performances_df.index,performances_df["Method"],rotation='vertical')

plt.xlabel("Models")

plt.ylabel("Cumulative Accuracy Profile")

plt.grid()

plt.tight_layout(.5)

plt.show()