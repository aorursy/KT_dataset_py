import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
frames = list()
results = pd.read_csv("../input/train.csv")
for i in range(1,19):
    exp = '0' + str(i) if i < 10 else str(i)
    frame = pd.read_csv("../input/experiment_{}.csv".format(exp))
    row = results[results['No'] == i]
    frame['target'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    frames.append(frame)
df = pd.concat(frames, ignore_index = True)
df.head()
df.info()
df_correlation=df.corr()
df_correlation.dropna(thresh=1,inplace=True)
df_correlation.drop(columns=['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia','target'],inplace=True)
plt.figure(figsize=(20,20))
sns.heatmap(df_correlation)


from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import gc # for deleting unused variables
%matplotlib inline

#Creating Test Train Splits
x=df.drop(columns=['target','Machining_Process'],axis=1)
y=np.array(df['target'])
X_train,X_test,y_train,y_test =train_test_split(x,y,train_size=0.8,random_state=100)

#XgBoost

xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
# make predictions for test data
# use predict_proba since we need probabilities to compute auc
y_pred = xgb_model.predict(X_test)
y_pred[:10]
print("Trained on {0} observations and scoring with {1} test samples.".format(len(X_train), len(X_test)))
# roc_auc
#y_pred=y_pred.round()
auc = roc_auc_score(y_test, y_pred)
auc
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

accuracy=accuracy_score(y_pred=y_pred,y_true=y_test)
print("accuracy of Model to calculate Tool wear is:{0}%".format(round(accuracy,4)*100))
# Error terms
#Actual vs Predicted
#c = [i for i in range(1,(len(y_test)+1),1)]
start_point=100
count_points=130
c = [i for i in range(start_point,count_points+1,1)]
fig = plt.figure()
plt.plot(c,y_test[start_point:count_points+1], color="blue", linewidth=2.5, linestyle="-")#Actual Plot in blue
plt.plot(c,y_pred[start_point:count_points+1], color="red",  linewidth=2.5, linestyle="--")#predicted Plot in red
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Worn_status', fontsize=16) 
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds
draw_roc(y_test,y_pred)
# plot
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.show()
# Feature importances
features = [(df.columns[i], v) for i,v in enumerate(xgb_model.feature_importances_)]
features.sort(key=lambda x: x[1], reverse = True)
for item in features[:10]:
    print("{0}: {1:0.4f}".format(item[0], item[1]))
