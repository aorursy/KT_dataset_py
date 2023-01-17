import numpy as np # linear algebra
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import math
import random
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tqdm import tqdm_notebook as tqdm
from sklearn import svm
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve

import warnings
warnings.filterwarnings('ignore')

# for inline plots
%matplotlib inline

# initialise figure sizes
mpl.rcParams['figure.figsize'] = (10,5)
mpl.rc('xtick', labelsize = 15) 
mpl.rc('ytick', labelsize = 15)

#initialise figure sizes
font = {'size'   : 15}
mpl.rc('font', **font)
print('running')
# get data from csv files
test  = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

# divide into X and y data
X_all = pd.DataFrame(train.iloc[:,1:train.shape[1]])
y_all = pd.DataFrame(train.iloc[:,0])

X_test = test

#slpit into, these will be chnges later as is approriate 
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size = 0.25)

#determine sizes of datasets
n_all,m_all = X_train.shape
n_train,m_train = X_train.shape
n_val, m_val = X_val.shape
n_test,m_test = X_test.shape

# print a summary of loaded results
print('FULL DATA')
print('Number of features (m): %.0f'%(m_all))
print('Number of traiing samples (n): %.0f'%(n_all))

print('TRAINING DATA')
print('Number of features (m): %.0f'%(m_train))
print('Number of traiing samples (n): %.0f'%(n_train))

print('\nVALIDATION DATA')
print('Number of features (m): %.0f'%(m_val))
print('Number of traiing samples (n): %.0f'%(n_val))

print('\nTEST DATA')
print('Number of features (m): %.0f'%(m_test))
print('Number of traiing samples (n): %.0f'%(n_test))

# Plot images in an O x P = N_samples subplot
def PlotSamples(N_samples, X, y, Label):
    n, m = X.shape
        
    rows = int(round(np.sqrt(N_samples)))
    columns = math.ceil(N_samples/rows)
    
    sample_i = random.sample(range(0, n), N_samples)
    mpl.rcParams['figure.figsize'] = [9,9]
        
    f, ax = plt.subplots(rows, columns)
    plt.tight_layout(pad = 0.2, w_pad = .1, h_pad=.1)
   
    for i in range(0, rows * columns):
    
        if i < N_samples:
                
                data = X.iloc[sample_i[i],:].values #this is the first number
                pix_rows, pix_cols = int(math.sqrt(m)), int(math.sqrt(m))
                grid = data.reshape((pix_rows, pix_cols))
                ax[i // columns, i % columns].imshow(grid)
                ax[i // columns, i % columns].axis('off') 
                ax[i // columns, i % columns].set_title(str(Label) + str(y['label'][sample_i[i]]))
                
        else:
                ax[i // columns, i % columns].axis('off')

# Plot images in an O x P = N_samples subplot of error made in prediction
def PlotError(N, X, False_pred_t, False_pred_f):

    n, m = X.shape   
    rows = int(round(np.sqrt(N)))
    columns = math.ceil(N/rows)
    
    sample_i = random.sample(list(False_pred_f.index), N)
    
    mpl.rcParams['figure.figsize'] = [9,9]    
    f, ax = plt.subplots(rows, columns)
    plt.tight_layout(pad = 0.2, w_pad = .1, h_pad = .1)
       
    for i in range(0, rows * columns):
    
        if i < N:
    
            False_label = 'False Prediction: ' + str(False_pred_f['label'][sample_i[i]]) + '  '
            True_label = 'True Label: '  + str(False_pred_t['label'][sample_i[i]]) + '  '
            label1 = False_label + '\n' + True_label
            
            data = X.loc[sample_i[i],:].values #this is the first number
            
            pix_rows, pix_cols = int(math.sqrt(m)), int(math.sqrt(m))
            grid = data.reshape((pix_rows, pix_cols))
            ax[i // columns, i % columns].imshow(grid)
            ax[i // columns, i % columns].axis('off') 
            ax[i // columns, i % columns].set_title(label1, fontsize = 12)
            
        else:
            ax[i // columns, i % columns].axis('off')
            
    return

#plot the training and validation confusion matrix in subplots
def Confusion(confusion_train, confusion_val):
    font = {'size'   : 15}
    mpl.rc('font', **font)
    
    class_names = [0,1,2,3,4,5,6,7,8,9]
    
    c_train = pd.DataFrame(confusion_train, index = class_names, columns = class_names )
    c_val = pd.DataFrame(confusion_val, index = class_names, columns = class_names )
          
    fig = plt.figure(figsize = (14,7))
    plt.subplot(121)
    heatmap = sns.heatmap(c_train, annot = True, cmap="YlGnBu", fmt = "d", annot_kws={"size": 12})
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Training Results', fontsize = 15)
    plt.subplot(122)
    
    heatmap = sns.heatmap(c_val, annot = True, cmap="YlGnBu", fmt = "d",annot_kws={"size": 12})
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Validation Results', fontsize = 15)
    
    return fig

# Reduce the size of data by R (0 - 1)
def ReduceData(R, X, y):
    n,m = X.shape
    
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    n_reduce = math.ceil(R * n)
    
    X_red = X.iloc[0:n_reduce,:]
    y_red = y.iloc[0:n_reduce]
    
    return(X_red, y_red)

# Normalisation:
def Norm(X):
    X_std = StandardScaler().fit_transform(X)
    return(X_std)
# plot a sample of figures with teh label printed over the number
print('Randomly selected samples with their assigned labels:')
PlotSamples(36, X_all, y_all, 'Label: ')


# plot a barchart of the label counts
plt.figure(figsize = [9,5])
sns.countplot(x = 'label', data = train);
plt.ylabel('Counts')
plt.xlabel("Number")
plt.title('Reoccurance of the Specific Numbers')
plt.grid(True)

mpl.rcParams['figure.figsize'] = (8,5)
pca_model = PCA().fit(X_all)

explain_ratio = pca_model.explained_variance_ratio_ * 100 #calculate variance ratios
var  = np.cumsum((pca_model.explained_variance_ratio_) * 100)

mpl.rcParams['figure.figsize'] = (15,6)
plt.figure()
plt.subplot(121)
plt.bar(range(len(explain_ratio)), explain_ratio)
plt.xlabel('Individual Component')
plt.ylabel('Individual Explain Variance (%))')
plt.title('Individual contribution of Components')
plt.xlim(0, 50)
plt.grid(True)

plt.subplot(122)
plt.plot(var, linewidth = 2)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)');
plt.title('Cumulative Contribution of Components')
plt.grid(True)
plt.xlim(0, 400)
plt.ylim(0,100)
plt.grid(True)
# determines the PCA adn SVM models 
def PCA_SVM_Model(X_std, y, N):
    
    #perform PCA feature reduction
    pca_model = PCA(n_components = N, svd_solver = 'randomized', whiten = True) 
    pca_model.fit(X_std)
    variance_N = np.sum(explain_ratio[0:N]) # determine the cotribution ofthe first N components
    
    X_pca = pd.DataFrame(pca_model.transform(X_std))
    
    svm_model = svm.SVC()
    svm_model.fit(X_pca, y)
    
    return(pca_model, svm_model, variance_N)
     
def Apply_PCA_SVM(pca_model, svm_model, X_std):
    X_pca = pd.DataFrame(pca_model.transform(X_std))
    # predict y-values training
    y_pred = pd.DataFrame(svm_model.predict(X_pca))
    y_pred.columns = ['label']
    
    return(y_pred)

def Error(y, y_pred):
    f1_sc = f1_score(y, y_pred, average='macro')
    f1_err = 1 - f1_sc
    return(f1_sc, f1_err)
#Initialise settings
N_comp_max = 70
N_range = range(5, N_comp_max,5)
Reduction = 1
split = 0.25

#Initialise arrays
f1_train_error = np.zeros(len(N_range))
f1_val_error = np.zeros(len(N_range))
variance_N = np.zeros(len(N_range))
cnt = 0

# Reduce data and split int training and validation data
X_red, y_red = ReduceData(Reduction, X_all, y_all) # reduction is only for saving time
X_train, X_val, y_train, y_val = train_test_split(X_red, y_red, test_size = split)

# determine size of new data sets
n_all, m_all = X_all.shape
n_red, m_red = X_red.shape
n_train,m_train = X_train.shape
n_val, m_val = X_val.shape

#Normalisation of data
X_train_std = Norm(X_train)
X_val_std = Norm(X_val)
    
for i in tqdm(N_range): #run for all selected components number
    #print('Determining PCA and SVM models based on training set...')
    pca_model, svm_model, variance_N = PCA_SVM_Model(X_train_std, y_train, i)

    #print('Applying PCA and SVM model based on training set...')
    y_train_pred = Apply_PCA_SVM(pca_model, svm_model, X_train_std)
    y_val_pred = Apply_PCA_SVM(pca_model, svm_model, X_val_std)

    #print('Calculating Errors...')
    f1_train_score, f1_train_error[cnt] = Error(y_train, y_train_pred)
    f1_val_score, f1_val_error[cnt] = Error(y_val, y_val_pred)
    
    cnt = cnt + 1

print('SUMMARY')
print('Size of Original data: %.0f x %.0f'%(n_all, m_all))
print('Reduction of input data: %.2f'%(Reduction))
print('Size of Reduced data: %.0f x %.0f'%(n_red, m_red))
print('Training-Validation Split: %.2f'%(split))
print('Size of Training data: %.0f x %.0f'%(n_train, m_train))
print('Size of Validation data: %.0f x %.0f'%(n_val, m_val))
print('Training-Validation Split: %.2f'%(split))
    
plt.figure(figsize = (9,5))
plt.plot(N_range, f1_train_error, linewidth = 2)
plt.plot(N_range,f1_val_error, linewidth = 2)
plt.xlabel('PCA Components')
plt.ylabel('Error')
plt.grid(True)
plt.xlim(0,N_comp_max)
plt.legend(['Train', 'Validation'])

# initial settings
Reduction = 1 # reduces teh size of the data (to speed things up)
split = 0.25 # the split fot he training and validation data
N_components = 50 # Number of PCA components

#reduce the data (if you want) and split into training and validatino
X_std_red, y_red = ReduceData(Reduction, X_all, y_all) # reduce the size of the data
X_train, X_val, y_train, y_val = train_test_split(X_std_red, y_red, test_size = split) 

#Normalisation of data
X_train_std = Norm(X_train)
X_val_std = Norm(X_val)

n_train,m_train = X_train.shape
n_val,m_val = X_val.shape

print('Determining PCA and SVM models based on training set...')
pca_model, svm_model, variance_N = PCA_SVM_Model(X_train_std, y_train, N_components)

print('Applying PCA and SVM model based on training set...')
y_train_pred = Apply_PCA_SVM(pca_model, svm_model, X_train_std)
y_val_pred = Apply_PCA_SVM(pca_model, svm_model, X_val_std)

#a Apply the same index as the actual values
y_train_pred.index = y_train.index
y_val_pred.index = y_val.index

print('Calculating Errors...')
f1_train_score, f1_train_error = Error(y_train, y_train_pred)
f1_val_score, f1_val_error = Error(y_val, y_val_pred)

print('Done...')
confusion_train = confusion_matrix(y_train, y_train_pred)
confusion_val = confusion_matrix(y_val, y_val_pred) 

print('\nPCA-SVM SUMMARY')
print('Number of components: %.2f'%(N_components))
print('Explained Variance Ratio: %.2f %%'%(variance_N))

print('\nTRAINING RESULTS')
print('Number of training samples: %.0f'%(m_train))
print('Training f1 score: %.2f'%(f1_train_score))
print(classification_report(y_train_pred, y_train))

print('\nVALIDATION RESULTS:')
print('Validation f1 score: %.2f'%(f1_val_score))
print('Number of validation samples: %.0f'%(m_val))
print(classification_report(y_val_pred, y_val))

#plot Confusion tables
Confusion(confusion_train, confusion_val)

# Determine Errors
False_train_pred_f = y_train_pred[y_train['label'].values!= y_train_pred['label'].values]
False_train_pred_t = y_train[y_train['label'].values!= y_train_pred['label'].values]
True_train_pred = y_train[y_train['label'].values == y_train_pred['label'].values]

False_val_pred_f = y_val_pred[y_val['label'].values!= y_val_pred['label'].values]
False_val_pred_t = y_val[y_val['label'].values != y_val_pred['label'].values]
True_val_pred = y_val[y_val['label'].values == y_val_pred['label'].values]

plt.figure(figsize = (14,5))
plt.subplot(121)
sns.countplot(x = 'label', data = False_train_pred_f)
plt.grid(True)
plt.xlabel('Number')
plt.ylabel('Counts')
plt. title('Incorrect Predictions in Training')

plt.subplot(122)
sns.countplot(x = 'label', data = True_train_pred)
plt.grid(True)
plt.xlabel('Number')
plt.ylabel('Counts')
plt. title('Correct Predictions in Training')

plt.figure(figsize = (14,5))
plt.subplot(121)
sns.countplot(x = 'label', data = False_val_pred_f)
plt.grid(True)
plt.xlabel('Number')
plt.ylabel('Counts')
plt. title('Incorrect Predictions in Validation')

plt.subplot(122)
sns.countplot(x = 'label', data = True_val_pred)
plt.grid(True)
plt.xlabel('Number')
plt.ylabel('Counts')
plt. title('Correct Predictions in Validation')
print('Randomly selected training samples and their incorreclty predicted values using the PCA and SVM')

PlotError(25, X_train, False_train_pred_t, False_train_pred_f)

split = 0.25 # the split fot he training and validation data
N_components = 50 # Number of PCA components
m_max = n_all
m_nstep = 10
m_range = np.linspace(500, m_max, m_nstep)

f1m_train_error = np.zeros(len(m_range))
f1m_val_error = np.zeros(len(m_range))

cnt = 0

for i in tqdm(m_range):
    i = int(i)
    X_m = X_all.iloc[0:i,:]
    y_m = y_all.iloc[0:i]
    
    #slpit into 
    Xm_train, Xm_val, ym_train, ym_val = train_test_split(X_m, y_m, test_size = split)
    
    #Normalisation of data
    Xm_train_std = Norm(Xm_train)
    Xm_val_std = Norm(Xm_val)
    
    #print('Determining PCA and SVM models based on training set...')
    pca_model, svm_model, variance_N = PCA_SVM_Model(Xm_train_std, ym_train, N_components)

    #print('Applying PCA and SVM model based on training set...')
    ym_train_pred = Apply_PCA_SVM(pca_model, svm_model, Xm_train_std)
    ym_val_pred = Apply_PCA_SVM(pca_model, svm_model, Xm_val_std)

    #print('Calculating Errors...')
    f1m_train_score, f1m_train_error[cnt] = Error(ym_train, ym_train_pred)
    f1m_val_score, f1m_val_error[cnt] = Error(ym_val, ym_val_pred)
    
    cnt = cnt + 1

plt.figure(figsize = (8,5))
plt.plot(m_range, f1m_train_error)
plt.plot(m_range, f1m_val_error)
plt.xlabel('Number of Training Examples (m)')
plt.ylabel('Error')
plt.grid(True)
N_components = 50 # Number of PCA components

#Normalisation of data
X_all_std = Norm(X_all)
n_all, m_train = X_all.shape

print('Determining PCA and SVM models based on training set...')
pca_model_FINAL, svm_model_FINAL, variance_N = PCA_SVM_Model(X_all_std, y_all, N_components)

print('Applying PCA and SVM model based on training set...')
y_all_pred = Apply_PCA_SVM(pca_model_FINAL, svm_model_FINAL, X_all_std)

#a Apply the same index as the actual values
y_all_pred.index = y_all.index

print('Calculating Errors...')
f1_all_score, f1_all_error = Error(y_all, y_all_pred)

print('Done...')
confusion_all = confusion_matrix(y_all, y_all_pred)

print('\nPCA-SVM SUMMARY')
print('Number of components: %.2f'%(N_components))
print('Explained Variance Ratio: %.2f %%'%(variance_N))

print('\nALL RESULTS')
print('Number of training samples: %.0f'%(m_all))
print('Training f1 score: %.2f'%(f1_all_score))
print(classification_report(y_all_pred, y_all))

#plot Confusion tables
font = {'size'   : 15}
mpl.rc('font', **font)

class_names = [0,1,2,3,4,5,6,7,8,9]

confuse_all = pd.DataFrame(confusion_all, index = class_names, columns = class_names )

fig = plt.figure(figsize = (9,9))
heatmap = sns.heatmap(confuse_all, annot = True, cmap="YlGnBu", fmt = "d", annot_kws={"size": 12})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('All Results', fontsize = 15)
        
# Determine Errors
False_all_pred_f = y_all_pred[y_all['label'].values!= y_all_pred['label'].values]
False_all_pred_t = y_all[y_all['label'].values!= y_all_pred['label'].values]
True_all_pred = y_all[y_all['label'].values == y_all_pred['label'].values]

plt.figure(figsize = (14,5))
plt.subplot(121)
sns.countplot(x = 'label', data = False_all_pred_f)
plt.grid(True)
plt.xlabel('Number')
plt.ylabel('Counts')
plt. title('Incorrect Predictions in Training')

plt.subplot(122)
sns.countplot(x = 'label', data = True_all_pred)
plt.grid(True)
plt.xlabel('Number')
plt.ylabel('Counts')
plt. title('Correct Predictions in Training')


print('Randomly selected y_test samples and their predicted values using the PCA and SVM')
font = {'size': 12}
mpl.rc('font', **font)

X_test_std = Norm(X_test)
X_test_pca = pca_model_FINAL.transform(X_test_std)

y_test_pred = pd.DataFrame(svm_model_FINAL.predict(X_test_pca))
y_test_pred.columns = ['label']

PlotSamples(25, X_test, y_test_pred, 'Predict: ')

submission = pd.concat([pd.Series(range(1, (n_test + 1)), name = "ImageId"), y_test_pred],axis = 1)

submission.to_csv("Digits_PCA_SVM.csv",index = False)

print('Done')
