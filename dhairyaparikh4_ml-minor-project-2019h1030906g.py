import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
import itertools
%matplotlib inline
# Define file directories
INPUT_DIR = '/kaggle/input/minor-project-2020/'
OUTPUT_DIR = '/kaggle/working/'

# Define csv files to be saved into
TRAIN_CSV_FILE = 'train.csv'
TEST_CSV_FILE = 'test.csv'
OUTPUT_CSV_FILE = 'output.csv'
# Reading the train file
column_names = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12',
                'col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25',
                'col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38',
                'col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49', 'col_50','col_51',
                'col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64',
                'col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77',
                'col_78','col_79','col_80','col_81','col_82','col_83','col_84','col_85','col_86','col_87','target']

train_df = pd.read_csv(os.path.join(INPUT_DIR, TRAIN_CSV_FILE), header=0, delimiter=r",",names=column_names)

# Reading the test file
column_names = ['id','col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12',
                'col_13','col_14','col_15','col_16','col_17','col_18','col_19','col_20','col_21','col_22','col_23','col_24','col_25',
                'col_26','col_27','col_28','col_29','col_30','col_31','col_32','col_33','col_34','col_35','col_36','col_37','col_38',
                'col_39','col_40','col_41','col_42','col_43','col_44','col_45','col_46','col_47','col_48','col_49', 'col_50','col_51',
                'col_52','col_53','col_54','col_55','col_56','col_57','col_58','col_59','col_60','col_61','col_62','col_63','col_64',
                'col_65','col_66','col_67','col_68','col_69','col_70','col_71','col_72','col_73','col_74','col_75','col_76','col_77',
                'col_78','col_79','col_80','col_81','col_82','col_83','col_84','col_85','col_86','col_87']

test_df = pd.read_csv(os.path.join(INPUT_DIR, TEST_CSV_FILE), header=0, delimiter=r",",names=column_names)

train_df.head()
train_df.info()
test_df.head()
test_df.info()
count_labels = train_df.value_counts(train_df['target'], sort = True).sort_index()
count_labels.plot(kind = 'bar')
plt.title("Label Count")
plt.xlabel("Label")
plt.ylabel("Frequency")
corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(100, 100))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.show()
X = train_df.iloc[:,1:-1]
y = train_df.iloc[:,-1:]
number_records_one = len(train_df[train_df.target == 1])
one_indices = np.array(train_df[train_df.target == 1].index)
zero_indices = train_df[train_df.target == 0].index

random_zero_indices = np.random.choice(zero_indices, number_records_one, replace = False)
random_zero_indices = np.array(random_zero_indices)

under_sample_indices = np.concatenate([one_indices,random_zero_indices])
under_sample_data = train_df.iloc[under_sample_indices,:]

X_undersample = under_sample_data.iloc[:,1:-1]
y_undersample = under_sample_data.iloc[:,-1:]

print("Percentage of zero records: ", len(under_sample_data[under_sample_data.target == 0])/len(under_sample_data))
print("Percentage of one records: ", len(under_sample_data[under_sample_data.target == 1])/len(under_sample_data))
print("Total number of records in resampled data: ", len(under_sample_data))
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.3, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)

negative = train_data[train_data.target==0]
positive = train_data[train_data.target==1]

neg_downsampled = resample(negative,replace=True, n_samples=len(positive),random_state=27) 

downsampled = pd.concat([positive, neg_downsampled])

downsampled.target.value_counts()
X_train_undersample = downsampled.iloc[:,0:-1]
y_train_undersample = downsampled.iloc[:,-1:]
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = ['balanced']
solver = ['liblinear', 'saga']

param_grid = dict(penalty=penalty, C=C, class_weight=class_weight, solver=solver)
grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train_undersample, y_train_undersample.values.ravel())

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
lr = LogisticRegression(C = 0.001, class_weight = 'balanced' , penalty = 'l2', solver = 'saga', random_state = 0, max_iter= 1000)
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_val.values)

cnf_matrix = confusion_matrix(y_val,y_pred)
np.set_printoptions(precision=2)

print("Accuracy for the validation dataset: ", (cnf_matrix[0,1]+cnf_matrix[1,0])/(cnf_matrix[0,0]+cnf_matrix[0,1]+cnf_matrix[1,0]
                                                                                                                  +cnf_matrix[1,1]))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()
lr = LogisticRegression(C = 0.001, class_weight = 'balanced' , penalty = 'l2', solver = 'saga', random_state = 0, max_iter= 1000)
y_pred_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_val.values)

fpr, tpr, thresholds = roc_curve(y_val.values.ravel(),y_pred_score)
roc_auc = auc(fpr,tpr)

plt.title('ROC-AUC Curve')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
X_test = test_df.iloc[:,1:]
lr = LogisticRegression(C = 0.001, class_weight = 'balanced' , penalty = 'l2', solver = 'saga', random_state = 0, max_iter= 1000)
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_proba = lr.predict_proba(X_test.values)
output=pd.DataFrame(data={"id":test_df["id"],"target":y_pred_proba[:,1]}) 
output.to_csv(path_or_buf=os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE),index=False,quoting=3,sep=',')