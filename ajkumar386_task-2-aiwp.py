import pandas as pd

import time

import seaborn as sns

data_txt = pd.read_csv('../input/income-data/income_data.txt',sep=",",header=None)

data_txt.head()
data_txt.columns = ["age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country","income"]

data_txt.head()
categorical = [var for var in data_txt.columns if data_txt[var].dtype=='O']

for var in categorical: 

       print(data_txt[var].value_counts())
data_txt.isnull().sum()
missing_values = [" ?"]

data_txt.to_csv('a.csv',index=None)

data = pd.read_csv("a.csv",na_values = missing_values)

data.head()
data.isnull().sum()
data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)

data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)

data['native.country'].fillna(data['native.country'].mode()[0], inplace=True)
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

data['workclass']=enc.fit_transform(data['workclass'])

data['marital.status']=enc.fit_transform(data['marital.status'])

data['occupation']=enc.fit_transform(data['occupation'])

data['relationship']=enc.fit_transform(data['relationship'])

data['race']=enc.fit_transform(data['race'])

data['sex']=enc.fit_transform(data['sex'])

data['native.country']=enc.fit_transform(data['native.country'])

data['education']=enc.fit_transform(data['education'])

data['income']=enc.fit_transform(data['income'])
data.head()
x = data.iloc[:, 0:14]

y = data.iloc[:, 14]
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y,train_size = 0.70, test_size = 0.30, random_state = 0) 
from sklearn.preprocessing import StandardScaler 

sc_x = StandardScaler() 

xtrain = sc_x.fit_transform(xtrain)  

xtest = sc_x.transform(xtest) 
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score

start = time.time()

classifier = LogisticRegression(random_state = 0) 

classifier.fit(xtrain, ytrain)

end = time.time()

t = end-start

y_pred = classifier.predict(xtest) 

acc=accuracy_score(ytest, y_pred)

print ("Accuracy : ", accuracy_score(ytest, y_pred))

print("Time elapsed: ",t)

cm = confusion_matrix(ytest, y_pred) 

sns.heatmap(cm,annot=True)
# predict probabilities

y_pred = classifier.predict(xtest)

#print(lr_probs)

# keep probabilities for the positive outcome only



#print(lr_probs)

ns_probs = [0 for _ in range(len(ytest))]

#print(ns_probs)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

lr_auc = roc_auc_score(ytest, y_pred)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Logistic: ROC AUC=%.3f' % (lr_auc))
import matplotlib.pyplot as plt

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(ytest, y_pred)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Logistic: ROC AUC=%.3f' % (lr_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.naive_bayes import GaussianNB

start =time.time()

gnb = GaussianNB()

gnb.fit(xtrain, ytrain)

end = time.time()

t1=end-start

y_pred1=gnb.predict(xtest)

acc1=accuracy_score(ytest, y_pred1)

print ("Accuracy : ", accuracy_score(ytest, y_pred1))

print("Time elapsed: ",t1)

cm1 = confusion_matrix(ytest, y_pred1)

sns.heatmap(cm1,annot=True)
# predict probabilities

y_pred1 = gnb.predict(xtest)

#print(lr_probs)

# keep probabilities for the positive outcome only



#print(lr_probs)

ns_probs = [0 for _ in range(len(ytest))]

#print(ns_probs)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

nb_auc = roc_auc_score(ytest, y_pred1)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Naive Bayes: ROC AUC=%.3f' % (nb_auc))
import matplotlib.pyplot as plt

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

nb_fpr, nb_tpr, _ = roc_curve(ytest, y_pred1)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Naive Bayes: ROC AUC=%.3f' % (nb_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.neighbors import  KNeighborsClassifier

start = time.time()

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(xtrain, ytrain)

end = time.time()

t2 = end - start

y_pred2 = knn.predict(xtest)

acc2=accuracy_score(ytest, y_pred2)

print ("Accuracy : ", accuracy_score(ytest, y_pred2))

print("Time elapsed: ",t2)

cm2 = confusion_matrix(ytest, y_pred2) 

sns.heatmap(cm2,annot=True)
# predict probabilities

y_pred2 = knn.predict(xtest)

#print(lr_probs)

# keep probabilities for the positive outcome only



#print(lr_probs)

ns_probs = [0 for _ in range(len(ytest))]

#print(ns_probs)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

kn_auc = roc_auc_score(ytest, y_pred2)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('KNN: ROC AUC=%.3f' % (kn_auc))
import matplotlib.pyplot as plt

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

kn_fpr, kn_tpr, _ = roc_curve(ytest, y_pred2)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='KNN: ROC AUC=%.3f' % (kn_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.tree import DecisionTreeClassifier

start = time.time()

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)

decision_tree = decision_tree.fit(xtrain, ytrain)

end = time.time()

t3 = end - start

y_pred3 = decision_tree.predict(xtest)

acc3=accuracy_score(ytest, y_pred3)

print ("Accuracy : ", accuracy_score(ytest, y_pred3)) 

print("Time elapsed: ",t3)

cm3 = confusion_matrix(ytest, y_pred3) 

sns.heatmap(cm3,annot=True)
# predict probabilities

y_pred3 = decision_tree.predict(xtest)

#print(lr_probs)

# keep probabilities for the positive outcome only



#print(lr_probs)

ns_probs = [0 for _ in range(len(ytest))]

#print(ns_probs)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

dt_auc = roc_auc_score(ytest, y_pred3)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Decision Tree: ROC AUC=%.3f' % (dt_auc))
import matplotlib.pyplot as plt

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

dt_fpr, dt_tpr, _ = roc_curve(ytest, y_pred3)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Decision Tree: ROC AUC=%.3f' % (dt_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.ensemble import RandomForestClassifier

start = time.time()

rfc = RandomForestClassifier()

rfc.fit(xtrain,ytrain)

end = time.time()

t4 = end - start

y_pred4 = rfc.predict(xtest)

acc4=accuracy_score(ytest, y_pred4)

print ("Accuracy : ", accuracy_score(ytest, y_pred4)) 

print("Time elapsed: ",t4)

cm4 = confusion_matrix(ytest, y_pred4) 

sns.heatmap(cm4,annot=True)
# predict probabilities

y_pred4 = rfc.predict(xtest)

#print(lr_probs)

# keep probabilities for the positive outcome only



#print(lr_probs)

ns_probs = [0 for _ in range(len(ytest))]

#print(ns_probs)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

rf_auc = roc_auc_score(ytest, y_pred4)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('Random Forest: ROC AUC=%.3f' % (rf_auc))
import matplotlib.pyplot as plt

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

rf_fpr, rf_tpr, _ = roc_curve(ytest, y_pred4)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Random Forest: ROC AUC=%.3f' % (rf_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.svm import SVC

start = time.time()

svm = SVC(kernel = 'rbf', random_state = 0)

svm.fit(xtrain, ytrain)

end = time.time()

t5 = end-start

y_pred5 = svm.predict(xtest)

acc5=accuracy_score(ytest, y_pred5)

print ("Accuracy : ", accuracy_score(ytest, y_pred5))

print("Time elapsed: ",t5)

cm5 = confusion_matrix(ytest, y_pred5) 

sns.heatmap(cm5,annot=True)
# predict probabilities

y_pred5 = svm.predict(xtest)

#print(lr_probs)

# keep probabilities for the positive outcome only



#print(lr_probs)

ns_probs = [0 for _ in range(len(ytest))]

#print(ns_probs)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

sv_auc = roc_auc_score(ytest, y_pred5)

# summarize scores

print('Random Prediction: ROC AUC=%.3f' % (ns_auc))

print('SVC: ROC AUC=%.3f' % (sv_auc))
import matplotlib.pyplot as plt

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

sv_fpr, sv_tpr, _ = roc_curve(ytest, y_pred5)

# plot the roc curve for the model



plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='SVC: ROC AUC=%.3f' % (sv_auc))

# axis labels

plt.title('ROC CURVE')

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

# show the legend

plt.legend()

# show the plot

plt.show()
plt.figure(figsize=(15,10))

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Prediction: ROC AUC=%.3f' % (ns_auc))

plt.plot(lr_fpr, lr_tpr, linestyle='--',marker='*', label='Logistic: ROC AUC=%.3f' % (lr_auc))

plt.plot(dt_fpr, dt_tpr, linestyle='--',marker='*',label='Deciison Tree: ROC AUC=%.3f' % (dt_auc))

plt.plot(nb_fpr, nb_tpr, linestyle='--',marker='*',label='Naive Bayes: ROC AUC=%.3f' % (nb_auc))

plt.plot(rf_fpr, rf_tpr, linestyle='--',marker='*',label='Random Forest: ROC AUC=%.3f' % (rf_auc))

plt.plot(sv_fpr, sv_tpr, linestyle='--',marker='*',label='Suppot Vector Machines: ROC AUC=%.3f' % (sv_auc))

plt.plot(kn_fpr, kn_tpr, linestyle='--',marker='*',label='KNN: ROC AUC=%.3f' % (kn_auc))

# axis labels

plt.xlabel('FALSE POSITIVE RATE')

plt.ylabel('TRUE POSITIVE RATE')

plt.title('ROC CURVES')

# show the legend

plt.legend()

# show the plot

plt.show()
ct = ['LR','NB','KNN','DT','RF','SVC']

ac = [acc,acc1,acc2,acc3,acc4,acc5]

fig = plt.figure(figsize = (10, 5)) 

plt.bar(ct, ac) 

plt.xlabel("Classification Technique") 

plt.ylabel("Accuracy Score") 

plt.title("Accuracy Analysis")

plt.show() 
ct = ['LR','NB','KNN','DT','RF','SVC']

tt = [t,t1,t2,t3,t4,t5]

fig = plt.figure(figsize = (10, 10)) 

plt.bar(ct,tt) 

plt.xlabel("Classification Technique") 

plt.ylabel("Time Elapsed in seconds") 

plt.title("Time Elapsed Analysis") 

plt.show() 