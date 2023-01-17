import pandas as pd

import numpy as np

from statsmodels.formula.api import logit,ols

from scipy.stats import uniform,norm,pearsonr,spearmanr

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,precision_score, recall_score,f1_score,accuracy_score, roc_curve

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.cluster import KMeans





def correlation(df,corr_value,p_v,function):

    rows_iterate=[]

    for row in df:

        for column in df:

            if (row != column) &(column not in rows_iterate) :

                corr,p_value,=function(df[row],df[column])

                if (abs(corr)>corr_value) & (p_value<=p_v):

                    print("Riga: ",row,", Colonna: ",column,", corr: ",corr,", p_value: ",p_value)

                    sns.scatterplot(df[row],df[column])

                    plt.show()

                rows_iterate.append(row)
df = pd.read_csv("../input/Dataset_spine.csv")

print(df.info())

df.groupby("Class_att").count()
df.describe()

df.groupby("Class_att")["Class_att"].count().plot.bar()

plt.grid()

plt.show()
print("range interquartile: \n", df.drop(columns="Class_att").quantile(0.75)-df.drop(columns="Class_att").quantile(0.25))
df.plot.box(figsize=(20,20))
df.plot.box(by ='Class_att')

df.boxplot(figsize=(15,25), by ='Class_att')

plt.show()

sns.boxplot(x="cervical_tilt", y ='Class_att',data=df )

plt.show()

sns.boxplot(x="pelvic_slope", y ='Class_att',data=df )

sns.boxplot(x="degree_spondylolisthesis", y ='Class_att',data=df )
data = df.copy()

data["Class_att"]=data["Class_att"].replace({"Abnormal":1,"Normal":0})

print(data.groupby("Class_att")["degree_spondylolisthesis"].mean())

print(data.groupby("Class_att")["pelvic_incidence"].mean())

print(data.groupby("Class_att")["lumbar_lordosis_angle"].mean())

print(data.groupby("Class_att")["sacral_slope"].mean())

print(data.groupby("Class_att")["pelvic_tilt"].mean())

plt.figure(figsize=(14,14))

sns.heatmap(data.corr(),linecolor="white",linewidths=0.1,annot=True)

plt.show()
def histPrinter(df):

    plt.figure(figsize=(10,7))

    i=1

    for column in df:

        plt.subplot(2,2,i)

        plt.title(column)

        plt.hist(df[column],density=True)

        df[column].plot.density()

        i=i+1

        if(i>4):

            i=1

            plt.show()

            plt.figure(figsize=(10,7))

histPrinter(df[df["Class_att"]=="Normal"].drop(columns="Class_att"))

histPrinter(df[df["Class_att"]=="Abnormal"].drop(columns="Class_att"))
print("pearson")

correlation(df.drop(columns="Class_att"),0.3,0.05,pearsonr)

print("spaerman")

correlation(df.drop(columns="Class_att"),0.3,0.05,spearmanr)
reg_log0=logit("Class_att~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+sacral_slope+pelvic_radius+degree_spondylolisthesis+pelvic_slope+direct_tilt+thoracic_slope+cervical_tilt+sacrum_angle+scoliosis_slope",data).fit()

reg_log0.summary()
reg_log01=logit("Class_att~pelvic_incidence+pelvic_tilt+pelvic_radius+degree_spondylolisthesis",data).fit()

reg_log01.summary()
train_set,test_set=train_test_split(data,test_size=0.20)

reg_log=logit("Class_att~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+sacral_slope+pelvic_radius+degree_spondylolisthesis+pelvic_slope+direct_tilt+thoracic_slope+cervical_tilt+sacrum_angle+scoliosis_slope",train_set).fit()

reg_log.summary()
reg_log2=logit("Class_att~pelvic_incidence+pelvic_tilt+pelvic_radius+degree_spondylolisthesis",train_set).fit()

reg_log2.summary()
test_mean=0

train_mean=0

data_mean=0

for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    reg_log2=logit("Class_att~pelvic_incidence+pelvic_tilt+pelvic_radius+degree_spondylolisthesis",train_set).fit()

    regLogPreds=reg_log2.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]])

    regLogPreds=regLogPreds.round()

    regLogTrainPreds = reg_log2.predict(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]])

    regLogTrainPreds=regLogTrainPreds.round()

    regLogDataPreds= reg_log2.predict(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]])

    regLogDataPreds=regLogDataPreds.round()

    test_mean+=accuracy_score(test_set["Class_att"],regLogPreds)

    train_mean+=accuracy_score(train_set["Class_att"],regLogTrainPreds)

    data_mean+=accuracy_score(data["Class_att"],regLogDataPreds)

    

test_matrix=confusion_matrix(test_set["Class_att"],reg_log2.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]).round())

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")

    

print(test_mean/10)

print(train_mean/10)

print(data_mean/10)
#classificatore a soglia

test_mean=0

plt.figure(figsize=(10,10))

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

for i in range(0,100):

    train_set,test_set=train_test_split(data,test_size=0.20)

    fpr, tpr, th = roc_curve(train_set["Class_att"], train_set["degree_spondylolisthesis"])

    score=tpr+1-fpr

    optimal_threshold = th[np.argmax(score)]

    prediction = test_set["degree_spondylolisthesis"]>optimal_threshold

    test_mean+=accuracy_score(test_set["Class_att"],prediction)

print(test_mean/100)

plt.plot(fpr,tpr)

fpr1, tpr1, th1 = roc_curve(data["Class_att"], data["pelvic_incidence"])

plt.plot(fpr1,tpr1)

fpr2, tpr2, th2 = roc_curve(data["Class_att"], data["pelvic_radius"])

plt.plot(tpr2,fpr2)

fpr3, tpr3, th3 = roc_curve(data["Class_att"], data["pelvic_tilt"])

plt.plot(fpr3,tpr3)

plt.legend(["degree_spondylolisthesis","pelvic_incidence","pelvic_radius","pelvic_tilt"])

test_matrix=confusion_matrix(test_set["Class_att"],prediction)

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

test_matrix
#classificatore bayesiano

test_mean=0

train_mean=0

data_mean=0



test_mean2=0

train_mean2=0

data_mean2=0

for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    naive = GaussianNB().fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    test_mean+=naive.score(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],test_set["Class_att"])

    train_mean+=naive.score(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    data_mean+=naive.score(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],data["Class_att"])

    

    naive2 = GaussianNB().fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

    test_mean2+=naive2.score(test_set.drop(columns="Class_att"),test_set["Class_att"])

    train_mean2+=naive2.score(train_set.drop(columns="Class_att"),train_set["Class_att"])

    data_mean2+=naive2.score(data.drop(columns="Class_att"),data["Class_att"])



test_matrix=confusion_matrix(test_set["Class_att"],naive.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]))

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")

print(test_mean/10)

print(train_mean/10)

print(data_mean/10)

print("\n")

print(test_mean2/10)

print(train_mean2/10)

print(data_mean2/10)
#SVM

param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},{'C': [1, 10, 100], 'gamma': [0.01, 0.05, 0.001], 'kernel': ['rbf']}]

gs = GridSearchCV(SVC(), param_grid, verbose=1, n_jobs=2)

gs.fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

print(gs.best_estimator_)



param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},{'C': [1, 10, 100], 'gamma': [0.01, 0.05, 0.001], 'kernel': ['rbf']}]

gs = GridSearchCV(SVC(), param_grid, verbose=1, n_jobs=2)

gs.fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

print(gs.best_estimator_)



test_mean=0

train_mean=0

data_mean=0



test_mean2=0

train_mean2=0

data_mean2=0

for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    svm = SVC(kernel="rbf",gamma=0.001).fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    test_mean+=svm.score(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],test_set["Class_att"])

    train_mean+=svm.score(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    data_mean+=svm.score(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],data["Class_att"])                                                                                  

    

    svm2=SVC(kernel="rbf",gamma=0.001).fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

    test_mean2+=svm2.score(test_set.drop(columns="Class_att"),test_set["Class_att"])

    train_mean2+=svm2.score(train_set.drop(columns="Class_att"),train_set["Class_att"])

    data_mean2+=svm2.score(data.drop(columns="Class_att"),data["Class_att"])



test_matrix=confusion_matrix(test_set["Class_att"],svm.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]))

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")



print(test_mean/10)

print(train_mean/10)

print(data_mean/10)

print("\n")

print(test_mean2/10)

print(train_mean2/10)

print(data_mean2/10)
#Decision tree

test_mean=0

train_mean=0

data_mean=0



test_mean2=0

train_mean2=0

data_mean2=0





for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    dt = DecisionTreeClassifier().fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    test_mean+=dt.score(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],test_set["Class_att"])

    train_mean+=dt.score(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    data_mean+=dt.score(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],data["Class_att"])

    

    dt2=DecisionTreeClassifier().fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

    test_mean2+=dt2.score(test_set.drop(columns="Class_att"),test_set["Class_att"])

    train_mean2+=dt2.score(train_set.drop(columns="Class_att"),train_set["Class_att"])

    data_mean2+=dt2.score(data.drop(columns="Class_att"),data["Class_att"])



test_matrix=confusion_matrix(test_set["Class_att"],dt.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]))

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")    



print(test_mean/10)

print(train_mean/10)

print(data_mean/10)

print("\n")

print(test_mean2/10)

print(train_mean2/10)

print(data_mean2/10)
#Random forest

test_mean=0

train_mean=0

data_mean=0



test_mean2=0

train_mean2=0

data_mean2=0





for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    df = RandomForestClassifier().fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    test_mean+=df.score(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],test_set["Class_att"])

    train_mean+=df.score(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    data_mean+=df.score(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],data["Class_att"])

    

    df2=RandomForestClassifier().fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

    test_mean2+=df2.score(test_set.drop(columns="Class_att"),test_set["Class_att"])

    train_mean2+=df2.score(train_set.drop(columns="Class_att"),train_set["Class_att"])

    data_mean2+=df2.score(data.drop(columns="Class_att"),data["Class_att"])



test_matrix=confusion_matrix(test_set["Class_att"],df.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]))

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")   

    

print(test_mean/10)

print(train_mean/10)

print(data_mean/10)

print("\n")

print(test_mean2/10)

print(train_mean2/10)

print(data_mean2/10)
#KNN



param_grid = [ {'n_neighbors':range(1,50)}]

gs = GridSearchCV(KNN(), param_grid)

gs.fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

print(gs.best_estimator_.n_neighbors)



param_grid = [ {'n_neighbors':range(1,50)}]

gs = GridSearchCV(KNN(), param_grid)

gs.fit(train_set.drop(columns="Class_att"),train_set["Class_att"],train_set["Class_att"])

print(gs.best_estimator_.n_neighbors)



test_mean=0

train_mean=0

data_mean=0



test_mean2=0

train_mean2=0

data_mean2=0

for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    knn = KNN(n_neighbors=(i%10+5))

    knn.fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    test_mean+=knn.score(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],test_set["Class_att"])

    train_mean+=knn.score(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    data_mean+=knn.score(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],data["Class_att"])

    

    knn2=KNN(n_neighbors=(i%10+5))

    knn2.fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

    test_mean2+=knn2.score(test_set.drop(columns="Class_att"),test_set["Class_att"])

    train_mean2+=knn2.score(train_set.drop(columns="Class_att"),train_set["Class_att"])

    data_mean2+=knn2.score(data.drop(columns="Class_att"),data["Class_att"])

    

test_matrix=confusion_matrix(test_set["Class_att"],knn.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]))

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")  

    

    

print(test_mean/10)

print(train_mean/10)

print(data_mean/10)

print("\n")

print(test_mean2/10)

print(train_mean2/10)

print(data_mean2/10)
#LDA

test_mean=0

train_mean=0

data_mean=0



test_mean2=0

train_mean2=0

data_mean2=0

for i in range(0,10):

    train_set,test_set=train_test_split(data,test_size=0.20)

    lda = LDA().fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    #transform=lda.transform(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]])

    #transform=pd.DataFrame(transform)

    #transform["c"]=train_set.reset_index(drop=True)["Class_att"]

    test_mean+=lda.score(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],test_set["Class_att"])

    train_mean+=lda.score(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],train_set["Class_att"])

    data_mean+=lda.score(data[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]],data["Class_att"])

    

    lda2= LDA().fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

    #transform2=lda2.transform(train_set.drop(columns="Class_att"),train_set["Class_att"])

    test_mean2+=lda2.score(test_set.drop(columns="Class_att"),test_set["Class_att"])

    train_mean2+=lda2.score(train_set.drop(columns="Class_att"),train_set["Class_att"])

    data_mean2+=lda2.score(data.drop(columns="Class_att"),data["Class_att"])



    

test_matrix=confusion_matrix(test_set["Class_att"],lda.predict(test_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]]))

test_matrix=test_matrix/test_matrix.sum(1).reshape(-1,1)

print(test_matrix,"\n")     

    

print(test_mean/10)

print(train_mean/10)

print(data_mean/10)

print("\n")

print(test_mean2/10)

print(train_mean2/10)

print(data_mean2/10)
#PCA

train_set,test_set=train_test_split(data,test_size=0.30)

pca = PCA(n_components=2).fit(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]])

pcaTransform= pca.transform(train_set[["pelvic_incidence","pelvic_tilt","pelvic_radius","degree_spondylolisthesis"]])

pcaTransform=pd.DataFrame(pcaTransform)

pcaTransform["c"]=train_set.reset_index(drop=True)["Class_att"]

sns.scatterplot(pcaTransform[0],pcaTransform[1],hue=pcaTransform["c"])

print(np.cumsum(pca.explained_variance_ratio_))
train_set,test_set=train_test_split(data,test_size=0.20)

pca2= PCA().fit(train_set.drop(columns="Class_att"),train_set["Class_att"])

print(np.cumsum(pca2.explained_variance_ratio_))

pcaTransform2=pca2.transform(data.drop(columns="Class_att"))

pcaTransform2=pd.DataFrame(pcaTransform)

pcaTransform2["c"]=train_set.reset_index(drop=True)["Class_att"]

sns.scatterplot(pcaTransform2[0],pcaTransform2[1],hue=pcaTransform2["c"])

plt.figure(figsize=(10,10))

plt.plot(np.cumsum(pca2.explained_variance_ratio_), 'ro-')

plt.grid()