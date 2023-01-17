import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
data = pd.read_csv("../input/heart.csv")

data.head(3)
print(data.info())
import seaborn as sns



corr_matrix = data.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

corr_matrix["target"].sort_values(ascending=False)

plt.figure(figsize=(10,5))

ax1 = plt.subplot(1, 2, 1)

ax = sns.scatterplot(x="thalach", y="trestbps", hue="target",data=data)



ax1 = plt.subplot(1, 2, 2)

ax = sns.scatterplot(x="thalach", y="oldpeak", hue="target",data=data)

plt.figure(figsize=(10,5))

ax1 = plt.subplot(1, 2, 1)

sns.distplot(data["thalach"][data["target"]==0], label="Negative")

sns.distplot(data["thalach"][data["target"]==1], label="Positive")

plt.ylabel("density")

plt.xlabel("maximum heart rate achieved")

plt.legend()

ax2 = plt.subplot(1, 2, 2)

ax = sns.scatterplot(x="thalach", y="target", hue="target",data=data)

plt.xlabel("maximum heart rate achieved")

plt.show()
sns.catplot(x="sex", hue = "target",kind="count", data=data);

plt.xlabel("Sex (0:Female, 1:Male)")

plt.title("Heart Disease (Target 0:Positive, 1:Negitave)")

plt.show()
sns.catplot(x="cp",hue="target",kind = "count",data=data)

plt.xlabel("Chest Pain (0:Female, 1:Male)")

plt.title("Heart Disease (Target 0:Positive, 1:Negitave)")

plt.show()
X = data.iloc[:,:-1].values

Y = data.iloc[:,-1].values



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 5)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

log_clf = LogisticRegression(solver="lbfgs")

log_clf.fit(x_train,y_train)

y_logclf_pred = log_clf.predict(x_test)

print(accuracy_score(y_test,y_logclf_pred))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,y_logclf_pred))

print(classification_report(y_test,y_logclf_pred))
#Encoding categorical features

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

fe1_1hot = encoder.fit_transform(data['thal'].values.reshape(-1,1))

fe2_1hot = encoder.fit_transform(data['ca'].values.reshape(-1,1))

fe3_1hot = encoder.fit_transform(data['slope'].values.reshape(-1,1))



datadrop = data.drop(columns=["thal","ca","slope"])

X_new = datadrop.iloc[:,:-1].values

Y_new = datadrop.iloc[:,-1].values

X_new = np.concatenate((X_new,fe1_1hot.toarray(),fe2_1hot.toarray(),fe3_1hot.toarray()),axis=1)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=3)

sgd_clf.fit(x_train,y_train)

y_sgd_clf_pred = sgd_clf.predict(x_test)

print(accuracy_score(y_test,y_sgd_clf_pred))
from sklearn.preprocessing import StandardScaler

stand_sca = StandardScaler()

X_trans = stand_sca.fit_transform(X_new)
x_train, x_test, y_train, y_test = train_test_split(X_trans,Y_new,test_size = 0.2, random_state = 5)

sgd_clf = SGDClassifier(random_state=3,n_jobs=-1)

sgd_clf.fit(x_train,y_train)

y_sgd_clf_pred = sgd_clf.predict(x_test)

print(accuracy_score(y_test,y_sgd_clf_pred))
plt.scatter([0,1],[100*0.704,100*0.868])

plt.ylabel("accuracy")

plt.annotate('Before Feature Scaling', xy=(0, 72), xytext=(0, 75), fontsize = 12,

            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))



plt.annotate('After Feature Scaling', xy=(1, 85), xytext=(0.6, 82.5), fontsize = 12,

            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))



plt.show()
def testclassfiersgd(max_in,tol_in):

    sgd_clf = SGDClassifier(random_state=3,n_jobs=-1,max_iter=max_in,tol=tol_in)

    sgd_clf.fit(x_train,y_train)

    y_sgd_clf_pred = sgd_clf.predict(x_test)

    acc = accuracy_score(y_test,y_sgd_clf_pred)

    return acc



max_iter_test = [10,50,100,500,1000,3000]

tol = [10,1,0.1,0.2,0.3]



acc_matrix_sgd = np.zeros((len(max_iter_test),len(tol)))



for i in range(len(max_iter_test)):

    #print(max_iter_test[i])

    for j in range(len(tol)):

        #print(tol[j])

        acc_matrix_sgd[i,j] = testclassfiersgd(max_iter_test[i],tol[j])



sns.heatmap(acc_matrix_sgd,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

plt.ylabel("max_iter")

plt.xlabel("tol")

plt.show()
log_clf = LogisticRegression(solver="lbfgs")

log_clf.fit(x_train,y_train)

y_logclf_pred = log_clf.predict(x_test)

print("Logistic Regression accuracy",100*accuracy_score(y_test,y_logclf_pred))
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100,random_state=1)

rf_clf.fit(x_train,y_train)

y_rf_clf_pred = rf_clf.predict(x_test)

print("Random Forest accuracy",100*accuracy_score(y_test,y_rf_clf_pred))
from sklearn.svm import SVC

svc_clf = SVC(kernel="rbf")

svc_clf.fit(x_train,y_train)

y_svc_clf_pred = svc_clf.predict(x_test)

print("SVM accuracy",100*accuracy_score(y_test,y_svc_clf_pred))
from sklearn.ensemble import GradientBoostingClassifier

grbt_clf = GradientBoostingClassifier(max_depth=2,n_estimators=100,random_state=1)

grbt_clf.fit(x_train,y_train)



errors = np.zeros((100,1))

i = 0

for y_pred in grbt_clf.staged_predict(x_test):

    errors[i] = accuracy_score(y_test,y_pred)

    i = i + 1

    #print(y_pred)



best_n_estimator = np.argmax(errors)



plt.plot(errors)

plt.xlabel('number of trees');plt.ylabel('accuracy');plt.show()



grbt_clf_best = GradientBoostingClassifier(max_depth=2,n_estimators=best_n_estimator+1)

grbt_clf_best.fit(x_train,y_train)

y_test_gbrt = grbt_clf_best.predict(x_test)



print("GBR accuracy is:",100*accuracy_score(y_test,y_test_gbrt))
plt.scatter([0,1,2,3,4],[86.8,93.4,86.8,90.1,91.8])

plt.ylabel("Accuracy");plt.title("Accuracy comparision of classifiers")

plt.xticks([0,1,2,3,4],("SGD","RF","LOG","GB","SVC"))

plt.show()
from sklearn.model_selection import cross_validate

sgd_cval=cross_validate(sgd_clf,X_trans,Y_new,cv=5)

rf_cval=cross_validate(rf_clf,X_trans,Y_new,cv=5)

log_cval=cross_validate(log_clf,X_trans,Y_new,cv=5)

grbt_cval=cross_validate(grbt_clf_best,X_trans,Y_new,cv=5)

svc_cval=cross_validate(svc_clf,X_trans,Y_new,cv=5)
all_mod_cval=np.concatenate((sgd_cval["test_score"],rf_cval["test_score"],log_cval["test_score"],

               grbt_cval["test_score"],svc_cval["test_score"]),axis=0)
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

sns.heatmap(all_mod_cval.reshape((5,5)),annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

plt.title("5 fold cross validation")

plt.yticks([0,1,2,3,4],("SGD","RF","LOG","GB","SVC"))

plt.subplot(1,2,2)

plt.scatter([0,1,2,3,4],[sgd_cval["test_score"].mean(),

                        rf_cval["test_score"].mean(),

                        log_cval["test_score"].mean(),

                        grbt_cval["test_score"].mean(),

                        svc_cval["test_score"].mean()])

plt.ylabel("Avg Accuracy");plt.title("Avg accuracy comparision of classifiers")

plt.xticks([0,1,2,3,4],("SGD","RF","LOG","GB","SVC"))

plt.show()