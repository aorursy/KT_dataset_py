
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
dtc = tree.DecisionTreeClassifier(max_depth=4, min_impurity_decrease=0.002)

kf=KFold(n_splits=5, shuffle=True, random_state=0)
AUC=[]
Accuracy=[]
Confusion_Matrix=[]
for train_id, test_id in kf.split(Data_X,Data_y):
    X_train, X_test = Data_X.values[train_id], Data_X.values[test_id]
    y_train, y_test = Data_y.values[train_id], Data_y.values[test_id]
    dtc = dtc.fit(X_train,y_train)
    y_pred = dtc.predict_proba(X_test)[:,1]
    y_pred_binary=dtc.predict(X_test)
    accur=accuracy_score(y_test,y_pred_binary)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    AUC.append(roc_auc)
    #Confusion_Matrix.append(CM)
    Accuracy.append(accur)
print('Accuracy: ' + str((Accuracy)))

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Any results you write to the current directory are saved as output.
