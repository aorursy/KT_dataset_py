import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#from tqdm import tnrange, tqdm_notebook
#from time import sleep  (working on how to implement this along with sklearn model fitting)
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
X = df.drop(['Outcome'], axis=1)
y = df.Outcome
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.28, random_state = 0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs' , max_iter = 400)
logreg.fit(Xtrain, ytrain)
ypred = logreg.predict(Xtest)
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(ytest,ypred)
cnf_matrix
class_names = [0,1]
fig, ax = plt.subplots()
fig
ax
tick_marks = np.arange(len(class_names))
tick_marks
plt.xticks(tick_marks , class_names)
plt.yticks(tick_marks, class_names)
ax.xaxis.set_label_position("top")
plt.tight_layout()
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.2)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest, ypred))
print("Precision:",metrics.precision_score(ytest, ypred))
print("Recall:",metrics.recall_score(ytest, ypred))
fpr, tpr , _ = metrics.roc_curve(ytest,ypred)
auc = metrics.roc_auc_score(ytest,ypred)
plt.plot(fpr,tpr,label = "auc : " + str(auc))
plt.plot([0,1],[0,1],'r--')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend(loc = 4)
plt.show()