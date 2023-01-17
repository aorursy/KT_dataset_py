#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Load data as dataframe
data=pd.read_csv("../input/creditcard.csv")
data.head()
#normalising data except time and class columns
data2= (data.iloc[:,1:-1] - data.iloc[:,1:-1].mean()) / (data.iloc[:,1:-1].max() - data.iloc[:,1:-1].min())
data2['Class']=data['Class']

#Converting time in seconds to hours 
data2['Hour'] = data['Time'] //3600
data2['Hour'].replace(-0,0,inplace=True)
data=data2
print("Normalised data :")
data.head()
from imblearn.over_sampling import SMOTE
smt = SMOTE()
def do_smote(data):
    names=list(data)
    y_t = data.Class
    X_t = data.drop('Class', axis=1)
    X_t, y_t = smt.fit_sample(X_t, y_t)
    #np.bincount(y_t)
    X=pd.DataFrame(X_t)
    X['Class']=y_t
    X.columns = names
    return X
smoted_data=do_smote(data)
for j in list(smoted_data):
    for i in range(2):
        sns.kdeplot(smoted_data[smoted_data.Class==i][j])
    plt.show()
keep=['V1','V2','V3','V4','V5','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','V21','V26','Amount','Hour','Class']
data=data[keep]
data.head()
data.Class.value_counts()
final_data=do_smote(data)
final_data.Class.value_counts()
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score
def get_performance_metrics(y_test,model_predictions):
    # Accuracy
    model_accuracy = accuracy_score(y_test,model_predictions)
    print("Accuracy is ", model_accuracy)

    # precision, recall, f1 score
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_test,model_predictions)
    print('Precision for each class is ', model_precision)
    print('Recall/sensitivity for each class is ', model_recall)
    print('F1 Score for each class is ', model_f1)

    # roc_auc
    model_roc_auc = roc_auc_score(y_test,model_predictions)
    print('AUC-ROC score is ', model_roc_auc)

    # confusion matrix
    model_confusion_matrix = confusion_matrix(y_test,model_predictions)
    print('confusion matrix is :-->')
    print(model_confusion_matrix)
# Separate input features (X) and target variable (y)

y = final_data.Class
X = final_data.drop('Class', axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
# Train model
reg = LogisticRegression().fit(X_train, y_train)
 
# Predict
pred_y = reg.predict(X_test)
 
get_performance_metrics(y_test, pred_y)
from sklearn.ensemble import RandomForestClassifier

# Train model
rf = RandomForestClassifier(class_weight={0: 100,1: 1})
rf.fit(X_train, y_train)
 # Predict on training set
pred_y = rf.predict(X_test)
get_performance_metrics(y_test, pred_y)
#from sklearn.svm import SVC
#svc_model = SVC()
#train
#svc_model.fit(X_train,y_train)
#predict
#svc_predictions= svc_model.predict(X_test)
#get_performance_metrics(y_test,svc_predictions)
