import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import timeit
import seaborn as sns
from pylab import rcParams

pd.set_option('display.max_columns', 500) # to display all the columns
sns.set(style='darkgrid', palette='dark', font_scale=2)
rcParams['figure.figsize'] = 10, 6
# to display plots inline

%matplotlib inline 
#Importing data

try:
    df = pd.read_csv('../input/breastCancer.csv')
except Exception as e:
    print(e)
    gc.collect()
#Lookigng into the data
df.head(5)
# Checking the dataframe shape
df.shape
df['bare_nucleoli'] = df['bare_nucleoli'].replace('?', np.NaN)
#Function to find % of missing values in df 
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns 
missing_values_table(df)
# Replacing the Nan Values with 0 and converting it to int dtype
df.bare_nucleoli.fillna(0,inplace=True)
df.bare_nucleoli = df.bare_nucleoli.astype(int)
# dropping the id column from df
columns = ['id']
df.drop(columns, inplace=True, axis=1)
df.head(2)
print(df['class'].value_counts())
sns.countplot(df['class'])
plt.xlabel('Labels')
plt.title('Target Value Counts')
# mapping 2 , 4 values to 0 , 1 classes
values = {2: 0, 4: 1}
df['class'] = df['class'].map(values) 
df['class'].value_counts()
# Separating Target Variable
y = df['class'].copy()
X = df.drop(labels = ['class'],axis = 1)
#Looking into the shape of features & target
y.shape , X.shape
#Minority Resampling - SMOTE

from imblearn.over_sampling import SMOTE
from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
sm = SMOTE()
X , y = sm.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y)))

X = pd.DataFrame(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
#Looking into the shape of features & target after split
X_train.shape , y_train.shape , X_test.shape , y_test.shape
# Importing the SVM classfier 
from sklearn.svm import SVC
# Function to plaot confusion matrix
def plot_conf_matrix (confusion_matrix):
    class_names = [0,1]
    fontsize=14
    df_conf_matrix = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names, 
        )
    fig = plt.figure()
    heatmap = sns.heatmap(df_conf_matrix, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# function to plot ROC Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def plot_roc_curve(roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
X_train.shape, X_test.shape
start_time = timeit.default_timer()

# Creating the /linear classifier

svclassifier_linear = SVC(kernel='linear' , class_weight = 'balanced' , verbose = True)  
svclassifier_linear.fit(X_train, y_train)

svclassifier_linear_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_linear = svclassifier_linear.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_linear)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred_linear))


print("Recall:",metrics.recall_score(y_test, y_pred_linear))
from sklearn.metrics import confusion_matrix
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_linear)
fpr, tpr, threshold = roc_curve(y_test, y_pred_linear)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.transform(X_test)
start_time = timeit.default_timer()
# Creating the Sigmoid classifier

svclassifier_sigmoid = SVC(kernel='sigmoid', verbose = True )  
svclassifier_sigmoid.fit(X_train_std, y_train)

svclassifier_sigmoid_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_sigmoid = svclassifier_sigmoid.predict(X_test_std)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_sigmoid)
acc_score
from sklearn.metrics import classification_report, confusion_matrix  
print(classification_report(y_test, y_pred_sigmoid)) 
from sklearn.metrics import confusion_matrix
conf_matrix_sigmoid = confusion_matrix(y_test, y_pred_sigmoid, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_sigmoid)
fpr, tpr, threshold = roc_curve(y_test, y_pred_sigmoid)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
start_time = timeit.default_timer()

# Creating the Gaussian/RBF classifier

svclassifier_rbf = SVC(kernel='rbf' , class_weight = 'balanced' ,verbose = True)  
svclassifier_rbf.fit(X_train_std, y_train)

svclassifier_rbf_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_rbf = svclassifier_rbf.predict(X_test_std)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_rbf)
acc_score
print(classification_report(y_test, y_pred_rbf)) 
from sklearn.metrics import confusion_matrix
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_rbf)
fpr, tpr, threshold = roc_curve(y_test, y_pred_rbf)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
start_time = timeit.default_timer()

# Creating the Sigmoid classifier

svclassifier_poly = SVC(kernel='poly',verbose = True)  
svclassifier_poly.fit(X_train_std, y_train)

svclassifier_poly_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_poly = svclassifier_poly.predict(X_test_std)
#from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_poly)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred_poly))


print("Recall:",metrics.recall_score(y_test, y_pred_poly))
from sklearn.metrics import confusion_matrix
conf_matrix_poly = confusion_matrix(y_test, y_pred_poly, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_poly)
fpr, tpr, threshold = roc_curve(y_test, y_pred_poly)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ANOVA SVM-C
# anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)

start_time = timeit.default_timer()

clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)

anova_svm_time = timeit.default_timer() - start_time

y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))
acc_score = accuracy_score(y_test, y_pred)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn.metrics.pairwise import laplacian_kernel
def l_kernel(X, y):
    return laplacian_kernel(X, y)
start_time = timeit.default_timer()

clf_l_kernel = svm.SVC(kernel=l_kernel)
clf_l_kernel.fit(X_train_std, y_train)

clf_l_kernel_time = timeit.default_timer() - start_time

y_pred_l_kernel = clf_l_kernel.predict(X_test_std)
y_pred_l_kernel = clf_l_kernel.predict(X_test_std)
acc_score = accuracy_score(y_test, y_pred_l_kernel)
acc_score
from sklearn.metrics import confusion_matrix
conf_matrix_l_kernel = confusion_matrix(y_test, y_pred_l_kernel, labels=None, sample_weight=None)

plot_conf_matrix(conf_matrix_l_kernel)
fpr, tpr, threshold = roc_curve(y_test, y_pred_l_kernel)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
print("Precision:",metrics.precision_score(y_test, y_pred_l_kernel))


print("Recall:",metrics.recall_score(y_test, y_pred_l_kernel))
from sklearn.svm import NuSVC
start_time = timeit.default_timer()

# Creating the linear classifier

svclassifier_linear = NuSVC(nu= 0.2 , kernel='linear' , class_weight = 'balanced' , verbose = True)  
svclassifier_linear.fit(X_train, y_train)

nu_svclassifier_linear_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_linear = svclassifier_linear.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_linear)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred_linear))


print("Recall:",metrics.recall_score(y_test, y_pred_linear))
from sklearn.metrics import confusion_matrix
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_linear)
fpr, tpr, threshold = roc_curve(y_test, y_pred_linear)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.transform(X_test)
start_time = timeit.default_timer()

# Creating the Sigmoid classifier

svclassifier_sigmoid = NuSVC(nu= 0.2 , kernel='sigmoid', verbose = True )  
svclassifier_sigmoid.fit(X_train_std, y_train)

nu_svclassifier_sigmoid_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_sigmoid = svclassifier_sigmoid.predict(X_test_std)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_sigmoid)
acc_score
from sklearn.metrics import classification_report, confusion_matrix  
print(classification_report(y_test, y_pred_sigmoid)) 
from sklearn.metrics import confusion_matrix
conf_matrix_sigmoid = confusion_matrix(y_test, y_pred_sigmoid, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_sigmoid)
fpr, tpr, threshold = roc_curve(y_test, y_pred_sigmoid)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
start_time = timeit.default_timer()

# Creating the Gaussian/RBF classifier

svclassifier_rbf = NuSVC(nu= 0.2 , kernel='rbf' , class_weight = 'balanced' ,verbose = True)  
svclassifier_rbf.fit(X_train_std, y_train)

nu_svclassifier_rbf_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_rbf = svclassifier_rbf.predict(X_test_std)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_rbf)
acc_score
print(classification_report(y_test, y_pred_rbf)) 
from sklearn.metrics import confusion_matrix
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_rbf)
fpr, tpr, threshold = roc_curve(y_test, y_pred_rbf)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
start_time = timeit.default_timer()

# Creating the Sigmoid classifier

svclassifier_poly = NuSVC(nu= 0.2 , kernel='poly',verbose = True)  
svclassifier_poly.fit(X_train_std, y_train)

nu_svclassifier_poly_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_poly = svclassifier_poly.predict(X_test_std)
#from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_poly)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred_poly))


print("Recall:",metrics.recall_score(y_test, y_pred_poly))
from sklearn.metrics import confusion_matrix
conf_matrix_poly = confusion_matrix(y_test, y_pred_poly, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_poly)
fpr, tpr, threshold = roc_curve(y_test, y_pred_poly)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn import svm 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ANOVA SVM-C
# anova filter, take 3 best ranked features

start_time = timeit.default_timer()

anova_filter = SelectKBest(f_regression, k=3)

clf = NuSVC(nu= 0.2 , kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)

nu_svclassifier_poly_time = timeit.default_timer() - start_time

y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))
acc_score = accuracy_score(y_test, y_pred)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn.metrics.pairwise import laplacian_kernel
def l_kernel(X, y):
    return laplacian_kernel(X, y)
start_time = timeit.default_timer()

clf_l_kernel = NuSVC(nu= 0.2 ,  kernel=l_kernel)
clf_l_kernel.fit(X_train_std, y_train)

nu_svclassifier_laplacian_time = timeit.default_timer() - start_time

y_pred_l_kernel = clf_l_kernel.predict(X_test_std)
y_pred_l_kernel = clf_l_kernel.predict(X_test_std)
acc_score = accuracy_score(y_test, y_pred_l_kernel)
acc_score
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred_l_kernel, labels=None, sample_weight=None)

plot_conf_matrix(conf_matrix)
fpr, tpr, threshold = roc_curve(y_test, y_pred_l_kernel)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
print("Precision:",metrics.precision_score(y_test, y_pred_l_kernel))

print("Recall:",metrics.recall_score(y_test, y_pred_l_kernel))
start_time = timeit.default_timer()

# Creating the linear classifier

svclassifier_linear = SVC(kernel='linear' , C=1.0 , class_weight = 'balanced' , verbose = True)  
svclassifier_linear.fit(X_train, y_train)

c_svclassifier_linear_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_linear = svclassifier_linear.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_linear)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred_linear))


print("Recall:",metrics.recall_score(y_test, y_pred_linear))
from sklearn.metrics import confusion_matrix
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_linear)
fpr, tpr, threshold = roc_curve(y_test, y_pred_linear)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_test_std = scalar.transform(X_test)
start_time = timeit.default_timer()

# Creating the Sigmoid classifier

svclassifier_sigmoid = SVC(C = 1.0 , kernel='sigmoid', verbose = True )  
svclassifier_sigmoid.fit(X_train_std, y_train)

c_svclassifier_sigmoid_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_sigmoid = svclassifier_sigmoid.predict(X_test_std)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_sigmoid)
acc_score
from sklearn.metrics import classification_report, confusion_matrix  
print(classification_report(y_test, y_pred_sigmoid)) 
from sklearn.metrics import confusion_matrix
conf_matrix_sigmoid = confusion_matrix(y_test, y_pred_sigmoid, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_sigmoid)
fpr, tpr, threshold = roc_curve(y_test, y_pred_sigmoid)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
start_time = timeit.default_timer()

# Creating the Gaussian/RBF classifier

svclassifier_rbf = SVC(C = 1.0 , kernel='rbf' , class_weight = 'balanced' ,verbose = True)  
svclassifier_rbf.fit(X_train_std, y_train)

c_svclassifier_rbf_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_rbf = svclassifier_rbf.predict(X_test_std)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_rbf)
acc_score
print(classification_report(y_test, y_pred_rbf)) 
from sklearn.metrics import confusion_matrix
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_rbf)
fpr, tpr, threshold = roc_curve(y_test, y_pred_rbf)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
start_time = timeit.default_timer()

# Creating the Sigmoid classifier

svclassifier_poly = SVC(C= 1.0 , kernel='poly',verbose = True)  
svclassifier_poly.fit(X_train_std, y_train)

c_svclassifier_poly_time = timeit.default_timer() - start_time
# Making Predictions
y_pred_poly = svclassifier_poly.predict(X_test_std)
#from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_poly)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred_poly))


print("Recall:",metrics.recall_score(y_test, y_pred_poly))
from sklearn.metrics import confusion_matrix
conf_matrix_poly = confusion_matrix(y_test, y_pred_poly, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_poly)
fpr, tpr, threshold = roc_curve(y_test, y_pred_poly)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn import svm 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ANOVA SVM-C
# anova filter, take 3 best ranked features
start_time = timeit.default_timer()

anova_filter = SelectKBest(f_regression, k=3)

clf = SVC(C = 1.0 , kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)

c_svclassifier_annova_time = timeit.default_timer() - start_time

y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))
acc_score = accuracy_score(y_test, y_pred)
acc_score
print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)

plot_conf_matrix(conf_matrix)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
from sklearn.metrics.pairwise import laplacian_kernel
def l_kernel(X, y):
    return laplacian_kernel(X, y)
start_time = timeit.default_timer()

clf_l_kernel = SVC(C = 1.0 ,  kernel=l_kernel)
clf_l_kernel.fit(X_train_std, y_train)

c_svclassifier_laplacian_time = timeit.default_timer() - start_time

y_pred_l_kernel = clf_l_kernel.predict(X_test_std)
y_pred_l_kernel = clf_l_kernel.predict(X_test_std)
acc_score = accuracy_score(y_test, y_pred_l_kernel)
acc_score
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred_l_kernel, labels=None, sample_weight=None)

plot_conf_matrix(conf_matrix)
fpr, tpr, threshold = roc_curve(y_test, y_pred_l_kernel)
roc_auc = auc(fpr, tpr)
plot_roc_curve(roc_auc)
print("Precision:",metrics.precision_score(y_test, y_pred_l_kernel))


print("Recall:",metrics.recall_score(y_test, y_pred_l_kernel))
# Creating Dict to capture the time taken by different classifiers 

time = dict( (name,eval(name)) for name in ['c_svclassifier_annova_time','c_svclassifier_laplacian_time','c_svclassifier_linear_time',
'c_svclassifier_poly_time','c_svclassifier_rbf_time','c_svclassifier_sigmoid_time','clf_l_kernel_time',
'svclassifier_linear_time','svclassifier_poly_time','svclassifier_rbf_time','svclassifier_sigmoid_time',
'nu_svclassifier_laplacian_time','nu_svclassifier_linear_time','nu_svclassifier_poly_time','nu_svclassifier_rbf_time',
 'nu_svclassifier_sigmoid_time'] )
# Sorting the list and displaying the time taken in sorted order

sorted(time.items(), key=lambda x: x[1])
#Visualizing the dataset of time consumed 

names = list(time.keys())
values = list(time.values())

plt.bar(range(len(time)),values,tick_label=names)
plt.xticks(fontsize=14, rotation=90)
plt.ylim(0, 8)
plt.show()
svVclassifier_linear = NuSVC(nu= 0.2, kernel='linear' , class_weight = 'balanced', probability=True)  
svVclassifier_sigmoid = NuSVC(nu= 0.2, kernel='sigmoid', probability=True)  
svVclassifier_rbf = NuSVC(nu= 0.2, kernel='rbf', class_weight = 'balanced', probability=True)
svVclassifier_poly = NuSVC(nu= 0.2, kernel='poly', probability=True)

anova_filter = SelectKBest(f_regression, k=3)
clf = NuSVC(nu= 0.2 , kernel='linear', probability=True)
svVclassifier_anova = make_pipeline(anova_filter, clf)

svVclassifier_laplace = NuSVC(nu= 0.2, kernel=l_kernel, probability=True)
svclassifier_linear = SVC(kernel='linear' , C=1.0 , class_weight = 'balanced', probability=True)  
svclassifier_sigmoid = SVC(C = 1.0 , kernel='sigmoid', probability=True)  
svclassifier_rbf = SVC(C = 1.0 , kernel='rbf' , class_weight = 'balanced', probability=True)  
svclassifier_poly = SVC(C= 1.0 , kernel='poly', probability=True) 

anova_filter = SelectKBest(f_regression, k=3)
clf = SVC(C = 1.0 , kernel='linear', probability=True)
svclassifier_anova = make_pipeline(anova_filter, clf)

svclassifier_laplace = SVC(C = 1.0 ,  kernel=l_kernel, probability=True)
algorithms = [svVclassifier_linear,svVclassifier_sigmoid,svVclassifier_rbf,svVclassifier_poly,svVclassifier_anova,
              svVclassifier_laplace,svclassifier_linear,svclassifier_sigmoid,svclassifier_rbf,svclassifier_poly,
              svclassifier_anova,svclassifier_laplace]
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
predictions_prob = np.matlib.zeros((len(y_test), len(algorithms)))
predictions = np.matlib.zeros((len(y_test), len(algorithms)))
training_acc = []
roc_score = []
start_time = timeit.default_timer()
for i,algorithm in enumerate(algorithms):
    clf = algorithm.fit(X_train_std, y_train)
    predictions[:,i] = clf.predict(X_test_std).reshape(X_test_std.shape[0],1)
    predictions_prob[:,i] = clf.predict_proba(X_test_std)[:,1].reshape(X_test_std.shape[0],1)
    #training_acc.append(clf.score(X_train_std,y_train))
    #roc_score.append(roc_auc_score(clf.predict(X_train_std),y_train))
    acc = cross_val_score(clf, X_train_std, y_train, scoring='accuracy', cv=10)
    training_acc.append(acc.mean())
    roc = cross_val_score(clf, X_train_std, y_train, scoring='roc_auc', cv=10)
    roc_score.append(roc.mean())
training_time = timeit.default_timer() - start_time
method_ensemble = []
acc_ensemble = []
roc_ensemble = []
f1_ensemble = []
start_time = timeit.default_timer()
final_predictions = []
for row_number in range(len(predictions_prob)):
    final_predictions.append(np.max(predictions_prob[row_number, ]))

preds = []
THRESHOLD = 0.7
for i in final_predictions:
    if i > THRESHOLD:
        preds.append(1)
    else:
        preds.append(0)
        
maxe_time = training_time + (timeit.default_timer() - start_time)
plot_conf_matrix(confusion_matrix(preds,y_test))
method_ensemble.append('MaxE')
acc_ensemble.append(accuracy_score(preds,y_test))
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
start_time = timeit.default_timer()

final_predictions = []
for row_number in range(len(predictions_prob)):
    final_predictions.append(np.min(predictions_prob[row_number, ]))

preds = []
THRESHOLD = 0.7
for i in final_predictions:
    if i > THRESHOLD:
        preds.append(1)
    else:
        preds.append(0)   
        
mine_time = training_time + (timeit.default_timer() - start_time)
plot_conf_matrix(confusion_matrix(preds,y_test))
method_ensemble.append('MinE')
acc_ensemble.append(accuracy_score(preds,y_test))
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
start_time = timeit.default_timer()
preds = []
for row_number in range(len(predictions)):
    (values,counts) = np.unique(np.array(predictions[row_number, ]),return_counts=True)
    ind=np.argmax(counts)
    preds.append(values[ind])

mve_time = training_time + (timeit.default_timer() - start_time)
plot_conf_matrix(confusion_matrix(preds,y_test))
method_ensemble.append('MVE')
acc_ensemble.append(accuracy_score(preds,y_test))
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
start_time = timeit.default_timer()

sum = 0
for acc in training_acc:
    sum = sum + acc

training_acc = training_acc/sum

final_predictions = []
for row_number in range(len(predictions_prob)):
    sum = 0
    for i in range(predictions_prob[0, ].shape[1]):
        sum = sum + (predictions_prob[row_number, ][0,i] * training_acc[i])
    final_predictions.append(sum)

preds = []
THRESHOLD = 0.7
for i in final_predictions:
    if i > THRESHOLD:
        preds.append(1)
    else:
        preds.append(0) 
        
wae_time = training_time + (timeit.default_timer() - start_time)
plot_conf_matrix(confusion_matrix(preds,y_test))
method_ensemble.append('WAE')
acc_ensemble.append(accuracy_score(preds,y_test))
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
start_time = timeit.default_timer()
clf_bg = BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=200)
clf_bg.fit(X_train_std,y_train)

bct_time = timeit.default_timer() - start_time
y_pred = clf_bg.predict(X_test_std)
method_ensemble.append('BCT')
acc_ensemble.append(accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
conf_matrix_bct = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)

plot_conf_matrix(conf_matrix_bct)
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

start_time = timeit.default_timer()
clf_ad = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                          n_estimators=300, random_state=1)
clf_ad.fit(X_train_std,y_train)

adaboost_time = timeit.default_timer() - start_time
# Making Predictions
boosting_pred = clf_ad.predict(X_test_std)
method_ensemble.append('Adaboost')
#from sklearn.metrics import accuracy_score
acc_ensemble.append(accuracy_score(y_test, boosting_pred))
from sklearn.metrics import confusion_matrix
conf_matrix_boosting = confusion_matrix(y_test, boosting_pred, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_boosting)
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
start_time = timeit.default_timer()
sum = 0
for score in roc_score:
    sum = sum + score

roc_score = roc_score/sum

final_predictions = []
for row_number in range(len(predictions_prob)):
    sum = 0
    for i in range(predictions_prob[0, ].shape[1]):
        sum = sum + (predictions_prob[row_number, ][0,i] * roc_score[i])
    final_predictions.append(sum)

preds = []
THRESHOLD = 0.7
for i in final_predictions:
    if i > THRESHOLD:
        preds.append(1)
    else:
        preds.append(0) 
        
wauce_time = training_time + (timeit.default_timer() - start_time)
plot_conf_matrix(confusion_matrix(preds,y_test))
method_ensemble.append('WAUCE')
acc_ensemble.append(accuracy_score(preds,y_test))
roc_ensemble.append(roc_auc_score(preds,y_test))
f1_ensemble.append(f1_score(preds,y_test))
# Creating Dict to capture the time taken by different ensemble methods 

time = dict((name,eval(name)) for name in ['maxe_time','mine_time','mve_time','wae_time',
                                           'bct_time','adaboost_time','wauce_time'])
# Sorting the list and displaying the time taken in sorted order

sorted(time.items(), key=lambda x: x[1])
#Visualizing the dataset of time consumed 

names = list(time.keys())
values = list(time.values())

plt.bar(range(len(time)),values,tick_label=names)
plt.xticks(fontsize=14, rotation=90)
plt.ylim(0, 8)
plt.show()
method_ensemble
plt.bar(method_ensemble,acc_ensemble,tick_label=method_ensemble)
plt.xticks(fontsize=14, rotation=90)
plt.ylim(0, 2)
plt.show()
plt.bar(method_ensemble,roc_ensemble,tick_label=method_ensemble)
plt.xticks(fontsize=14, rotation=90)
plt.ylim(0.9, 1)
plt.show()
plt.bar(method_ensemble,f1_ensemble,tick_label=method_ensemble)
plt.xticks(fontsize=14, rotation=90)
plt.ylim(0.9, 1)
plt.show()
