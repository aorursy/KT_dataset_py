import pandas as pd
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import fbeta_score, accuracy_score
from IPython.display import display
from sklearn.metrics import make_scorer
from IPython.core.display import HTML

## read CSV file
df = pd.read_csv("../input/creditcard.csv")
#is any NULL row ?
df.isnull().any().any(), df.shape
#is any NULL column ?
df.isnull().sum(axis=0)
## Show first 10 records
df.head(10)
## list columns
df.columns
## show columns data types
df.dtypes
## Statistical description
df.describe().transpose()
## return fraudulent transactions
fraudulent= len(df[df["Class"]==1])
## return genuine transactions
genuine = len(df[df["Class"]==0])
print ("Total number of fraudulent transactions : ", fraudulent)
print ("Total number of genuine transactions    : ", genuine)
## return total length "size"
total= len(df)
print ("Genuine percentage                      : ",(float(genuine) / float(total))*100,"%")
print ("Fraudulent percentage                   : ",(float(fraudulent) / float(total))*100,"%")
HTML("<div> <a href='https://plot.ly/~ali492/32/?share_key=Y5uuNKgdO2MVG8cgAo33a7' target='_blank' title='basic_pie_chart' style='display: block; text-align: center;'><img src='https://plot.ly/~ali492/32.png?share_key=Y5uuNKgdO2MVG8cgAo33a7' alt='basic_pie_chart' style='max-width: 100%;width: 600px;' width='600' onerror='this.onerror=null;this.src='https://plot.ly/404.png';' /></a> <script data-plotly='ali492:32' sharekey-plotly='Y5uuNKgdO2MVG8cgAo33a7' src='https://plot.ly/embed.js' async></script></div>")
HTML("<div><a href='https://plot.ly/~ali492/36/?share_key=6gOynd2Udby0UHcq8TLxbt' target='_blank' title='basic histogram' style='display: block; text-align: center;'><img src='https://plot.ly/~ali492/36.png?share_key=6gOynd2Udby0UHcq8TLxbt' alt='basic histogram' style='max-width: 100%;width: 600px;'  width='600' onerror='this.onerror=null;this.src='https://plot.ly/404.png';' /></a><script data-plotly='ali492:36' sharekey-plotly='6gOynd2Udby0UHcq8TLxbt' src='https://plot.ly/embed.js' async></script></div>")
df.drop(["Time","Amount"],axis=1 , inplace=True)
X = df.drop("Class",axis = 1)
y = df['Class']
from imblearn.under_sampling import RandomUnderSampler

ru  = RandomUnderSampler(random_state=422)
X_res, y_res = ru.fit_sample(X, y)
print ("Total number of fraudulent transactions : ", list(y_res).count(1) )
print ("Total number of genuine transactions    : ", list(y_res).count(0) )
total = len(y_res)
print ("Total transactions                      : ",total)
print ("Genuine percentage                      : ",(float(list(y_res).count(1)) / float(total))*100,"%")
print ("Fraudulent percentage                   : ",(float(list(y_res).count(1)) / float(total))*100,"%")
HTML("<div> <a href='https://plot.ly/~ali492/32/?share_key=Y5uuNKgdO2MVG8cgAo33a7' target='_blank' title='basic_pie_chart' style='display: block; text-align: center;'><img src='https://plot.ly/~ali492/32.png?share_key=Y5uuNKgdO2MVG8cgAo33a7' alt='basic_pie_chart' style='max-width: 100%;width: 600px;' width='600' onerror='this.onerror=null;this.src='https://plot.ly/404.png';' /></a> <script data-plotly='ali492:32' sharekey-plotly='Y5uuNKgdO2MVG8cgAo33a7' src='https://plot.ly/embed.js' async></script></div>")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size= 0.2, random_state= 430)
print("The split of the under_sampled data is as follows")
print("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

clf = SVC()
clf.fit(X_train, y_train.ravel())
from sklearn.metrics import f1_score, confusion_matrix,average_precision_score, precision_score, recall_score,roc_auc_score,accuracy_score
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision Recall:", average_precision_score(y_test, y_pred)*100)
clf= SVC(C= 23 ,kernel= 'rbf', random_state= 430, gamma = .001)
clf.fit(X_train, y_train.ravel())
from sklearn.metrics import f1_score, confusion_matrix,average_precision_score, precision_score, recall_score,roc_auc_score,accuracy_score
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision Recall:", average_precision_score(y_test, y_pred)*100)
cm = confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title("The Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
print("The Accuracy is : "+str((float(cm[1,1])+float(cm[0,0]))/(float(cm[0,0]) + float(cm[0,1])+float(cm[1,0]) + float(cm[1,1]))*100) + "%")
print("The Recall   is : "+ str(float(cm[1,1])/(float(cm[1,0]) + float(cm[1,1]))*100) +"%")
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()

