#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report,f1_score
from lightgbm import LGBMClassifier
import itertools
import scipy.stats as ss
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



RS=410 #Random State
data=pd.read_csv('/kaggle/input/caravan-insurance-challenge/caravan-insurance-challenge.csv')
data.head()
data.shape
data.columns
data.ORIGIN.value_counts()
data.info()
data.CARAVAN.value_counts()
plt.subplots(figsize=(10,8))
sns.heatmap(data.drop(columns=['ORIGIN']).corr());
fig,axes=plt.subplots(1,2,figsize=(12,8))
sns.heatmap(data.drop(columns=["ORIGIN"]).iloc[:,:43].corr(),vmin=-1,vmax=1,cmap='coolwarm',ax=axes[0])
sns.heatmap(data.drop(columns=['ORIGIN']).iloc[:,43:].corr(),vmin=-1,vmax=1,cmap='coolwarm',ax=axes[1])
axes[0].set_title("Upper-left Corrplot")
axes[1].set_title("Bottom-right Corrplot")
#Drop percentage representations
data_np=data.drop(columns=data.loc[:,(data.columns.str.startswith('p'))]).copy()
data_np.to_feather('reduced_cmbd.df')
!pip install pyarrow
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False, cf_report=False,
                          title='Confusion matrix', ax=None, cmap=plt.cm.Blues, cbar=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    if cf_report:
        print(classification_report(y_true,y_pred))
    
    fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(1,1)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    
    if cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # "Magic" numbers (https://stackoverflow.com/a/26720422/10939610)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
def plot_roc(y_true, y_pred, ax=None):
    """Plot ROC curve""" 
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_true, y_pred)
    roc_score = roc_auc_score(y_true,y_pred)
    
    fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(1,1)

    ax.set_title("Receiver Operating Characteristic")
    ax.plot(false_positive_rate, true_positive_rate)
    ax.plot([0, 1], ls="--")
    ax.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    ax.annotate('ROC: {:.5f}'.format(roc_score), [0.75,0.05])
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    fig.tight_layout()
    return roc_score
def feat_imps(model, X_train, plot=False, n=None):
    """ Dataframe containing each feature with its corresponding importance in the given model
    
    Args
    ----
        model : model, classifier that supports .feature_importances_ (RandomForest, AdaBoost, ect..)
        X_train : array like, training data object
        plot : boolean, if True, plots the data in the form of a bargraph
        n : int, only applicable if plot=True, number of features to plot, (default=15)
        
    Returns
    -------
        pandas DataFrame : columns = feature name, importance
    """
    
    fi_df = pd.DataFrame({'feature':X_train.columns,
                          'importance':model.feature_importances_}
                        ).sort_values(by='importance', ascending=False)
    if plot:
        fi_df[:(n if n is not None else 15)].plot.bar(x='feature',y='importance')
    else:
        return fi_df
def plot_cmroc(y_true, y_pred, classes=[0,1], normalize=True, cf_report=False):
    """Convenience function to plot confusion matrix and ROC curve """
    fig,axes = plt.subplots(1,2, figsize=(9,4))
    plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=normalize, cf_report=cf_report, ax=axes[0])
    roc_score = plot_roc(y_true, y_pred, ax=axes[1])
    fig.tight_layout()
    plt.show()
    return roc_score
train_df=data.query("ORIGIN=='train'").iloc[:,1:].copy()
test_df=data.query("ORIGIN=='test'").iloc[:,1:].copy()
X,y=train_df.drop(columns='CARAVAN'),train_df.CARAVAN
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=RS)
!pip install imblearn
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
ros=RandomOverSampler(random_state=RS)
rus=RandomUnderSampler(random_state=RS)
smt=SMOTE(random_state=RS,n_jobs=-1)
X_under,y_under=rus.fit_sample(X_train,y_train)
X_over,y_over=ros.fit_sample(X_train,y_train)
X_smote,y_smote=smt.fit_sample(X_train,y_train)
pd.DataFrame([*map(lambda x:ss.describe(x)._asdict(),[y_train,y_under,y_over,y_smote])],index=['Unbalanced','Undersample','Oversample','SMOTE'])
#Define baseline models
bc=BaggingClassifier(n_estimators=53,random_state=RS,n_jobs=-1)
ada=AdaBoostClassifier(n_estimators=53,random_state=RS)
rfc=RandomForestClassifier(n_estimators=53,random_state=RS,n_jobs=-1)
lgbm=LGBMClassifier(n_estimators=53,random_state=RS,n_jobs=-1)
bc_unbal=plot_cmroc(y_val,bc.fit(X_train,y_train).predict(X_val))
ada_unbal=plot_cmroc(y_val,ada.fit(X_train,y_train).predict(X_val))
lgbm_unbal=plot_cmroc(y_val,lgbm.fit(X_train,y_train).predict(X_val))
rfc_unbal=plot_cmroc(y_val,rfc.fit(X_train,y_train).predict(X_val))
models=[bc,ada,rfc,lgbm]
unbal_scores=[bc_unbal,ada_unbal,rfc_unbal,lgbm_unbal]
for model,score in zip(models,unbal_scores):
    print('{:25s}:{:.5f}'.format(model.__class__.__name__,score))
bc_under=plot_cmroc(y_val,bc.fit(X_under,y_under).predict(X_val))
ada_under=plot_cmroc(y_val,ada.fit(X_under,y_under).predict(X_val))
lgbm_under=plot_cmroc(y_val,lgbm.fit(X_under,y_under).predict(X_val))
rfc_under=plot_cmroc(y_val,rfc.fit(X_under,y_under).predict(X_val))
models=[bc,ada,rfc,lgbm]
under_scores=[bc_under,ada_under,rfc_under,lgbm_under]
for model,score in zip(models,under_scores):
    print('{:25s}:{:.5f}'.format(model.__class__.__name__,score))
bc_over=plot_cmroc(y_val,bc.fit(X_over,y_over).predict(X_val))
ada_over=plot_cmroc(y_val,ada.fit(X_over,y_over).predict(X_val))
lgbm_over=plot_cmroc(y_val,lgbm.fit(X_over,y_over).predict(X_val))
rfc_over=plot_cmroc(y_val,rfc.fit(X_over,y_over).predict(X_val))
models=[bc,ada,rfc,lgbm]
over_scores=[bc_over,ada_over,rfc_over,lgbm_over]
for model,score in zip(models,over_scores):
    print('{:25s}:{:.5f}'.format(model.__class__.__name__,score))
bc_smote=plot_cmroc(y_val,bc.fit(X_smote,y_smote).predict(X_val
                                                         ))
ada_smote=plot_cmroc(y_val,ada.fit(X_smote,y_smote).predict(X_val))
lgbm_smote=plot_cmroc(y_val,lgbm.fit(X_smote,y_smote).predict(X_val))
rfc_smote=plot_cmroc(y_val,rfc.fit(X_smote,y_smote).predict(X_val))
models=[bc,ada,rfc,lgbm]
smote_scores=[bc_smote,ada_smote,rfc_smote,lgbm_smote]
for model,score in zip(models,smote_scores):
    print('{:25s}:{:.5f}'.format(model.__class__.__name__,score))
X_test,y_test=test_df.iloc[:,:-1],test_df.iloc[:,-1]
bc=BaggingClassifier(n_estimators=53,n_jobs=-1)
ada=AdaBoostClassifier(n_estimators=53,random_state=RS)
rfc=RandomForestClassifier(n_estimators=53,n_jobs=-1,random_state=RS)
lgbm=LGBMClassifier(n_estimators=53,random_state=RS)

models=[bc,ada,rfc,lgbm]
for model in models:
    model.fit(X_under,y_under)
    tpreds=model.predict(X_test)
    print('{:25s}:{:.5f}'.format(model.__class__.__name__,roc_auc_score(y_test,tpreds)))
from sklearn.model_selection import GridSearchCV
param_grid={
    'learning_rate':[0.01,0.05,0.1,1],
    'n_estimators':[20,40,60,80,100],
    'num_values':[3,7,17,31],
    'max_bin':[4,8,16,32,64],
    'min_child_samples':[3,5,10,20,30],
}
lgbm_gs=GridSearchCV(LGBMClassifier(),param_grid,n_jobs=-1,scoring='roc_auc',verbose=2,iid=False,cv=5)
lgbm_gs.fit(X_under,y_under)
print('Best parameters:',lgbm_gs.best_params_)
plot_cmroc(y_val,lgbm_gs.predict(X_val))
plot_cmroc(y_test,lgbm_gs.predict(X_test))
param_grid_rf={
    'n_estimators':[40,60,100,128,256],
    'min_samples_leaf':[3,7,17,31],
    'max_leaf_nodes':[4,8,16,32,64],
    'min_samples_split':[3,5,10,20,30],
}
rfc_gs=GridSearchCV(RandomForestClassifier(),param_grid_rf,n_jobs=-1,scoring='roc_auc',verbose=2,iid=False,cv=5)
rfc_gs.fit(X_under,y_under)
print('Best parameters:',rfc_gs.best_params_)
plot_cmroc(y_val,rfc_gs.predict(X_val))
plot_cmroc(y_test,rfc_gs.predict(X_test))
lgbm_gs_ub=GridSearchCV(LGBMClassifier(),param_grid,n_jobs=-1,scoring='roc_auc',verbose=1,iid=False,cv=5)
lgbm_gs_ub.fit(X_train,y_train)
print('Best parameters:',lgbm_gs_ub.best_params_)
plot_cmroc(y_val,lgbm_gs_ub.predict(X_val))
plot_cmroc(y_test,lgbm_gs_ub.predict(X_test))
import warnings
from sklearn.feature_selection import RFE,SelectKBest,chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import scipy.stats as ss
import joblib
from mlxtend.feature_selection import SequentialFeatureSelector


fi_data=data.drop(columns=['ORIGIN']).copy()
X,y=fi_data.drop(columns='CARAVAN'),fi_data.CARAVAN
#plotting function
def scatter_density(data,labels,sca_title='',den_title="",**kwargs):
    """plot a scatter plot and a density plot Args:
             data:2-d array ,shape (n_samples,2)
             labels:array-like,class labels to be used for coloring scatterplot
              sca_title:str,scatter plot title
              den_title:str,density plot title
              **kwargs:keyword arguments passed to seaborn.
              Kdeplot
              Returns:
                     ax,matplotlib axis object"""
    fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True,sharex=True)
    #,gridspec_kw={'width_ratios':[50,50,4]}
    dataneg=data[labels==0]
    datapos=data[labels==1]
    sns.scatterplot(data[:,0],data[:,1],hue=labels,ax=ax[0])
    #sns.scatterplot(dataneg[:,0],dataneg[:,1],palette='Blues',ax=ax[0],alpha=0.06)
    #sns.scatterplot(datapos[:,0],datapos[:,1],palette='Oranges',ax=ax[0],alpha=1)
    sns.kdeplot(datapos[:,0],datapos[:,1],ax=ax[1],cmap='Oranges',**kwargs)
    sns.kdeplot(dataneg[:,0],dataneg[:,1],ax=ax[1],map='Blues',nlevels=30,**kwargs,shade=True,shade_lowest=False)#,cbar=True,cbar_ax=ax[2])
    ax[0].set_title(sca_title)
    ax[1].set_title(den_title)
    fig.tight_layout()
    plt.show()
    return ax
from sklearn.decomposition import PCA
Xs=pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
pca=PCA(random_state=RS)
Xpca=pca.fit_transform(Xs)

pca=PCA(random_state=RS)
_Xpca_raw=PCA(n_components=2,random_state=RS).fit_transform(X)
scatter_density(_Xpca_raw,y,'PCA Scatter Unscaled','PCA Density UnScaled');

from sklearn.decomposition import PCA
Xs=pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
pca=PCA(random_state=RS)
Xpca=pca.fit_transform(Xs)

Xpca=pca.fit_transform(Xs)
scatter_density(Xpca,y,'PCA Scaled:Scatter','PCA Scaled:Density');
pca.explained_variance_ratio_[:3]
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.annotate('(64,0.993)',xy=(64,0.993),xytext=(64,0.8),fontsize='medium',arrowprops={'arrowstyle':'->','mutation_scale':15})
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Explained variance')
plt.show()
!pip install openTSNE
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
tsne=TSNE(perplexity=75,learning_rate=500,n_iter=1000,metric='euclidean',negative_gradient_method='bh',n_jobs=4,callbacks=ErrorLogger(),random_state=RS)
Xembd=tsne.fit(Xs)
scatter_density(Xembd,y,'t-SNE scatter','t-SNE density');
!pip install 'umap-learn==0.3.10'
import umap.umap_ as umap

ump=umap.UMAP(n_neighbors=30,min_dist=0.2,random_state=RS,verbose=True)
Xumap=ump.fit_transform(Xs,y)
scatter_density(Xumap,y,'UMAP:Scatter','UMAP:Density')
ump=umap.UMAP(n_neighbors=30,min_dist=0.2,random_state=RS,verbose=False)
Xumap=ump.fit_transform(Xs)
ump=umap.UMAP(n_neighbors=30,min_dist=0.2,random_state=RS,verbose=False)
Xumap=ump.fit_transform(Xs)
scatter_density(Xumap,y,'UMAP:Scatter','UMAP:Density');

