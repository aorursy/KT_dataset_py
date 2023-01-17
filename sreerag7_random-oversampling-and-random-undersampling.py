import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()

from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

data.head()
data.isnull().sum().sum()
data.describe()
import seaborn as sns
print(data['Class'].value_counts())
target_count = data['Class'].value_counts()
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
sns.countplot(data['Class']);
count_class_0, count_class_1 = data['Class'].value_counts()
df_class_0 = data[data['Class'] == 0]
df_class_1 =data[data['Class'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.Class.value_counts())

df_test_under.Class.value_counts().plot(kind='bar', title='Count (Class)');
df_test_under.shape
X_under=df_test_under.drop(['Class'],axis=1)
y_under=df_test_under['Class']
from sklearn.model_selection import train_test_split

X_train_under, X_test_under, y_train_under,y_test_under = train_test_split(X_under,y_under,test_size=0.1)
X_train_under.shape, y_train_under.shape, X_test_under.shape
random_forest_under = RandomForestClassifier(n_estimators=100)
random_forest_under.fit(X_train_under, y_train_under)
y_pred_under = random_forest_under.predict(X_test_under)
acc_rf_under=accuracy_score(y_test_under,y_pred_under)
prec_rf_under,recall_rf_under,f1_rf_under,support_rf_under=precision_recall_fscore_support(y_test_under, y_pred_under, average='weighted')
cm_under=confusion_matrix(y_test_under,y_pred_under)
conf_matrix_under=pd.DataFrame(data=cm_under,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix_under, annot=True,fmt='d',cmap="YlGnBu")
prob_rf_under = random_forest_under.predict_proba(X_test_under)
fpr_rf_under ,tpr_rf_under, thresh_rf_under = roc_curve(y_test_under, prob_rf_under[:,1], pos_label=1)
random_probs_rf_under = [0 for i in range(len(y_test_under))]
p_fpr_rf_under, p_tpr_rf_under, _ = roc_curve(y_test_under, random_probs_rf_under, pos_label=1)
plt.title('ROC Random Forest Classifier UnderSampling ')
plt.plot(fpr_rf_under,tpr_rf_under)
plt.show()
auc_rf_under = roc_auc_score(y_test_under, prob_rf_under[:,1])
print('Accuracy:',acc_rf_under, 
      '\n' 'Precison:',prec_rf_under,
     '\n' 'Recall:',recall_rf_under,
     '\n' 'F1 Score:',f1_rf_under,
     '\n' 'AUC Score:',auc_rf_under)
     
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Class.value_counts())

df_test_over.Class.value_counts().plot(kind='bar', title='Count (Class)');
X_over=df_test_over.drop(['Class'],axis=1)
y_over=df_test_over['Class']
from sklearn.model_selection import train_test_split

X_train_over, X_test_over, y_train_over,y_test_over = train_test_split(X_over,y_over,test_size=0.1)
X_train_over.shape, y_train_over.shape, X_test_over.shape
random_forest_over = RandomForestClassifier(n_estimators=100)
random_forest_over.fit(X_train_over, y_train_over)
y_pred_over = random_forest_over.predict(X_test_over)
acc_rf_over=accuracy_score(y_test_over,y_pred_over)
prec_rf_over,recall_rf_over,f1_rf_over,support_rf_over=precision_recall_fscore_support(y_test_over, y_pred_over, average='weighted')
cm_over=confusion_matrix(y_test_over,y_pred_over)
conf_matrix_over=pd.DataFrame(data=cm_over,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix_over, annot=True,fmt='d',cmap="YlGnBu")
prob_rf_over = random_forest_over.predict_proba(X_test_over)
fpr_rf_over ,tpr_rf_over, thresh_rf_over = roc_curve(y_test_over, prob_rf_over[:,1], pos_label=1)
random_probs_rf_over = [0 for i in range(len(y_test_over))]
p_fpr_rf_over, p_tpr_rf_over, _ = roc_curve(y_test_over, random_probs_rf_over, pos_label=1)
plt.title('ROC RF Classifier Random Oversampling')
plt.plot(fpr_rf_over,tpr_rf_over)
plt.show()
auc_rf_over = roc_auc_score(y_test_over, prob_rf_over[:,1])
print('Accuracy:',acc_rf_over, 
      '\n' 'Precison:',prec_rf_over,
     '\n' 'Recall:',recall_rf_over,
     '\n' 'F1 Score:',f1_rf_over,
     '\n' 'AUC Score:',auc_rf_over)