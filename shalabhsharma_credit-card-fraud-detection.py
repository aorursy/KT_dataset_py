print("CREDIT CARD FRAUD DETECTION")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Loading dataset
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

fraud = data[data['Class']==1]
normal = data[data['Class']==0]

outlierFraction = len(fraud)/float(len(normal))
print("outlier fraction is ",outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 
print("There is only 0.17% fraud transactions out all the transactions. The data is highly Unbalanced.")
import seaborn as sns
import matplotlib.pyplot as plt

LABELS = ["Normal", "Fraud"]
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(2), LABELS)
plt.title('Count of Fraud vs. Normal Transactions')
plt.ylabel('Count')
plt.xlabel('Class (0:Normal, 1:Fraud)')
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),cmap="Blues")
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
#scaling the amount column
data['Amount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data.sample(5)
#dropping old amount and time columns
data.drop(["Time"], axis = 1, inplace = True)
data.sample(5)
# feature data (predictors)
y = data['Class']
X = data.drop(columns=['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
from imblearn.over_sampling import RandomOverSampler, SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 

sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
#Build the model
clf = LogisticRegression()
# Train the classifier
clf.fit(X_train_res, y_train_res)
#test the model
y_pred = clf.predict(X_test)

print('the Model used is Logistic Regression')
lacc= accuracy_score(y_test,y_pred)
print('The accuracy is {}'.format(lacc))
lprec= precision_score(y_test,y_pred)
print('The precision is {}'.format(lprec))
lrec= recall_score(y_test,y_pred)
print('The recall is {}'.format(lrec))
lf1= f1_score(y_test,y_pred)
print('The F1-Score is {}'.format(lf1))
lMCC=matthews_corrcoef(y_test,y_pred)
print('The Matthews correlation coefficient is{}'.format(lMCC))

#confusion matrix
import matplotlib.pyplot as plt 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(6, 6)) 
sns.set(font_scale=1.5)
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, annot_kws={"size": 16}, fmt ="d");
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
clf.fit(X_train_res,y_train_res)
y_pred = clf.predict(X_test)

print('the Model used is Decison Tree')
dacc= accuracy_score(y_test,y_pred)
print('The accuracy is {}'.format(dacc))
dprec= precision_score(y_test,y_pred)
print('The precision is {}'.format(dprec))
drec= recall_score(y_test,y_pred)
print('The recall is {}'.format(drec))
df1= f1_score(y_test,y_pred)
print('The F1-Score is {}'.format(df1))
dMCC=matthews_corrcoef(y_test,y_pred)
print('The Matthews correlation coefficient is{}'.format(dMCC))

#confusion matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(6, 6)) 
sns.set(font_scale=1.5)
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, annot_kws={"size": 16}, fmt ="d");
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()
from sklearn. ensemble import BaggingClassifier
bg = BaggingClassifier()
bg_model = bg.fit(X_train_res,y_train_res)
y_pred = bg_model.predict(X_test)
 
print('the Model used is Bagging Classifier')
bacc= accuracy_score(y_test,y_pred)
print('The accuracy is {}'.format(bacc))
bprec= precision_score(y_test,y_pred)
print('The precision is {}'.format(bprec))
brec= recall_score(y_test,y_pred)
print('The recall is {}'.format(brec))
bf1= f1_score(y_test,y_pred)
print('The F1-Score is {}'.format(bf1))
bMCC=matthews_corrcoef(y_test,y_pred)
print('The Matthews correlation coefficient is{}'.format(bMCC))

#confusion matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test,y_pred) 
plt.figure(figsize =(6, 6)) 
sns.set(font_scale=1.5)
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, annot_kws={"size": 16}, fmt ="d");
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train_res,y_train_res)
y_pred = rfc.predict(X_test)

n_errors = (y_pred != y_test).sum()
print('The model used is Random Forest classifier')
racc= accuracy_score(y_test,y_pred)
print('The accuracy is {}'.format(racc))
rprec= precision_score(y_test,y_pred)
print('The precision is {}'.format(rprec))
rrec= recall_score(y_test,y_pred)
print('The recall is {}'.format(rrec))
rf1= f1_score(y_test,y_pred)
print('The F1-Score is {}'.format(rf1))
rMCC=matthews_corrcoef(y_test,y_pred)
print('The Matthews correlation coefficient is{}'.format(rMCC))

#confusion matrix
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(6, 6)) 
sns.set(font_scale=1.5)
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, annot_kws={"size": 16}, fmt ="d");
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()
import pandas as pd
x={'Logistic R':[lacc,lprec,lrec,lf1,lMCC], 'Decison Tree':[dacc,dprec,drec,df1,dMCC], 'Bagging':[bacc,bprec,brec,bf1,bMCC], 'Random Forest':[racc,rprec,rrec,rf1,rMCC]}
df= pd.DataFrame(x, columns=['Logistic R','Decison Tree', 'Bagging','Random Forest'])
df.index=['Accuracy','Precision','Recall','F1-score','MCC']
df