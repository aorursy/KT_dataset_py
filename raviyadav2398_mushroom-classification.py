# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install pydotplus
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OneHotEncoder
from sklearn.compose import ColumnTransformer
from io import StringIO
from IPython.display import Image,display_html
from sklearn import tree
import pydotplus
import eli5
from eli5.sklearn import PermutationImportance
import shap
import lime
import statsmodels.api as sm
import scipy.stats as ss

RS=405
pd.set_option('max_columns',25)
shap.initjs()
mush=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv',dtype='category')
mush.columns=mush.columns.str.replace('-','_')
mush.rename(columns={'class':'toxic'},inplace=True)
mush.head()
mush.toxic.value_counts()
mush.info()
mush.stalk_root.value_counts()
mush.nunique().sort_values(ascending=False)
mush_enc=mush.drop(columns='veil_type').apply(lambda x:x.cat.codes)
X,y=mush.drop(columns=['toxic','veil_type']),mush.toxic
X_enc,y_enc=X.apply(lambda x:x.cat.codes),y.cat.codes
# categorical encoded dataset
X_train,X_test,y_train,y_test=train_test_split(X_enc,y_enc,test_size=.20,random_state=RS)
#One-hot encoded dataset
Xoh=pd.get_dummies(X,drop_first=False)
Xoh_train,Xoh_test,yoh_train,yoh_test=train_test_split(Xoh,y_enc,test_size=.20,random_state=RS)
X.shape,Xoh.shape
Xoh.head()
ftnames=X.columns.values#feature names
ftnames_oh=Xoh.columns.values#One-hot encoded feature names
def conditional_entropy(x,y):
    y=y.astype(np.int64)
    y_counter=np.bincount(y)
    xy_counter=Counter(list(zip(x,y)))
    total_occurrences=y_counter.sum()
    entropy=0
    for k,v in xy_counter.items():
        p_xy=v/total_occurrences
        p_y=y_counter[k[1]]/total_occurrences
        entropy +=p_xy*np.log(p_y/p_xy)
    return entropy
def cramers_v(x,y):
    "Calculates Cramer's V statistic for categorical-categorical association.this is a symmetric coefficient:V(x,y)=v(y,x)"
    confusion_matrix=pd.crosstab(x,y)
    chi2=ss.chi2_contingency(confusion_matrix)[0]
    n=confusion_matrix.sum().sum()
    phi2=chi2/n
    r,k=confusion_matrix.shape
    phi2corr=max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr=r-((r-1)**2)/(n-1)
    kcorr=k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
def theils_u(x,y):
    """Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
          This is the uncertainity of x given y:value is on the range of [0,1]
             -where 0 means y priovides no information about x,and 1 means y priovides full information about x. This is an asymmetric coefficient :U(x,y)!=U(y,x)"""
    x=x.astype(np.int64)
    s_xy=conditional_entropy(x,y)
    x_counter=np.bincount(x)
    total_occurrences=x_counter.sum()
    p_x=x_counter/total_occurrences
    s_x=ss.entropy(p_x)
    if s_x==0:
        return 1
    return (s_x-s_xy)/s_x
def catcorr(data,method='theils'):
    """Compute categorical correlations using uncertainty coefficients (Theil's U) or Cramer's V"""
    if method=='cramers':
        return data.corr(method=cramers_v)
    elif method !='theils':
        raise NotImplementedError(f"method:'{method}'not implemented,choose either 'cramers'or 'theils'")
        cols=data.columns
        clen=cols.size
        pairings=list(itertools.product(data.columns,repeat=2))
        theils_mat=np.reshape([theils_u(data[p[1]],data[p[0]]) for p in pairings],(clen,clen))
        return pd.DataFrame(theils_mat,index=cols,columns=cols)
        
        
    

    
    
def multi_table(*dfs):
    html_str=''
    for df in dfs:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
def pplot_cm(y_true,y_pred,labels=None,filename=None,ymap=None,cf_report=False,figsize=(7,5),**kwargs):
    if ymap is not None:
        y_pred=[ymap[yi] for yi in y_pred]
        y_true=[ymap[yi]for yi in y_true]
        labels=[ymap[yi]for yi in labels]
    if cf_report:
        print(classification_report(y_true,y_pred))
    labels=labels if labels is not None else y_true.unique()
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    cm_sum=np.sum(cm,axis=1,keepdims=True)
    cm_perc=cm/cm_sum.astype(float)*100
    annot=np.empty_like(cm).astype(str)
    nrows,ncols=cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c=cm[i,j]
            p=cm_perc[i,j]
            if i==j:
                s=cm_sum[i]
                annot[i,j]='%.1f%%\n%d/%d'%(p,c,s)
            elif c==0:
                annot[i,j]=''
            else:
                annot[i,j]='%.1f%%\n%d'%(p,c)
    cm=pd.DataFrame(cm,index=labels,columns=labels)
    cm.index.name='Actual'
    cm.columns.name='Predicted'
    fig,ax=plt.subplots(figsize=figsize)
    sns.heatmap(cm,annot=annot,fmt='',ax=ax,**kwargs)
    plt.savefig(filename) if filename is not None else plt.show()
def plot_tree(dtree,featnames,cnames=None,width=600,height=800):
    dot_data=StringIO()
    tree.export_graphviz(dtree,out_file=dot_data,feature_names=featnames,class_names=cnames,filled=True,rounded=True,special_characters=True)
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png(),width=width,height=height)
rfc=RandomForestClassifier(100,n_jobs=-1,random_state=RS)
rfc.fit(X_train,y_train)
preds=rfc.predict(X_test)

pplot_cm(y_test,preds,rfc.classes_,cf_report=True,figsize=(7,5),cmap='Blues')
skf=StratifiedKFold(5,shuffle=True,random_state=RS)
for train_idx,test_idx in skf.split(X_enc,y_enc):
    X_train,X_test,y_train,y_test=X_enc.loc[train_idx],X_enc.loc[test_idx],y_enc[train_idx],y_enc[test_idx]
    rfc.fit(X_train,y_train)
    y_pred=rfc.predict(X_test)
    print(classification_report(y_test,y_pred))
metrics=['precision','recall','f1','roc_auc']
scores=cross_validate(rfc,X_enc,y_enc,scoring=metrics,cv=10,return_train_score=True,n_jobs=-1)
for m in metrics:
    test_score,train_score=[scores[x] for x in scores.keys() if m in x]
    print(m+':\n','{:>4} train scores:{}'.format('',list(train_score)))
    print('{:>5}test scores :{}'.format('',list(test_score)))
    print('{:>5}test mean:{}'.format('',test_score.mean()))
rfc_fi=pd.DataFrame({'feature':X.columns,'importance':rfc.feature_importances_}).sort_values(by='importance',ascending=False)
sns.catplot(x='feature',y='importance',data=rfc_fi,kind='bar',aspect=1.5).set_xticklabels(rotation=90);
#filter out non perfect scoring decision trees,then take tree with fewest leaves
smallest_dt=min(filter(lambda dt:dt.score(X_test,y_test)==1,rfc.estimators_),key=lambda dt:dt.get_n_leaves())
plot_tree(smallest_dt,ftnames,['edible','poisonous'],500,600)
rfc_oh=RandomForestClassifier(100,n_jobs=-1,random_state=RS)
rfc_oh.fit(Xoh_train,yoh_train)
preds_oh=rfc_oh.predict(Xoh_test)
pplot_cm(yoh_test,preds_oh,rfc_oh.classes_,cf_report=True,figsize=(7,5),cmap='Blues')
rfc_oh_fi=pd.DataFrame({'feature':Xoh.columns,'importance':rfc_oh.feature_importances_}).sort_values(by='importance',ascending=False)
sns.catplot(x='feature',y='importance',data=rfc_oh_fi[:21],kind='bar',aspect=1.5).set_xticklabels(rotation=90)
odorXtox=pd.crosstab(mush.odor,mush.toxic)
gsizXtox=pd.crosstab(mush.gill_size,mush.toxic)
gcolXtox=pd.crosstab(mush.gill_color,mush.toxic)
multi_table(odorXtox,gsizXtox,gcolXtox)
smallest_dt_oh=min(filter(lambda dt:dt.score(Xoh_test,yoh_test)==1.0,rfc_oh.estimators_),key=lambda dt:dt.get_n_leaves())
plot_tree(smallest_dt_oh,ftnames_oh,['edilble','posionous'])
xgbc=xgb.XGBClassifier(n_jobs=-1,random_state=RS)
xgbc.fit(X_train,y_train)
preds=xgbc.predict(X_test)
pplot_cm(y_test,preds,xgbc.classes_,cf_report=True,figsize=(7,5),cmap='Blues')
xgbc_fi=pd.DataFrame({'feature':X.columns,'importance':xgbc.feature_importances_}).sort_values(by='importance',ascending=False)
sns.catplot(x='feature',y='importance',data=xgbc_fi,kind='bar',aspect=1.5).set_xticklabels(rotation=90)
xgbc_oh=xgb.XGBClassifier(n_jobs=-1,random_state=RS)
xgbc_oh.fit(Xoh_train,yoh_train)
preds=xgbc_oh.predict(Xoh_test)
pplot_cm(yoh_test,preds_oh,rfc_oh.classes_,cf_report=True,figsize=(7,5),cmap='Blues')
xgbc_oh_fi=pd.DataFrame({'feature':Xoh.columns,'importance':xgbc_oh.feature_importances_}).sort_values(by='importance',ascending=False)
sns.catplot(x='feature',y='importance',data=xgbc_oh_fi[:21],kind='bar',aspect=1.5).set_xticklabels(rotation=90);
np.random.seed(RS)
RNIDX=np.random.choice(X_test.index)#Random index from test dataset
posidx=X_test.index.get_loc(RNIDX)#positional index within the test dataset of the index label
print(f"Index label(full=split):{RNIDX}\nPostional index (X_test):{posidx}")
(X_enc.loc[RNIDX]==X_test.iloc[posidx]).all()
fi_merge=rfc_fi.merge(xgbc_fi,on='feature',suffixes=('_rf','_xgb')).set_index('feature')
#One-hot encoded feature importances
fi_oh_merge=rfc_oh_fi.merge(xgbc_oh_fi,on='feature',suffixes=('_rf','_xgb')).set_index('feature')
unc_coef=X_enc.corrwith(y_enc,method=theils_u).sort_values(ascending=False)
unc_coef_oh=Xoh.corrwith(y_enc,method=theils_u).sort_values(ascending=False)
fig,axs=plt.subplots(1,2,figsize=(12,6))
fi_merge.plot.bar(ax=axs[0])
unc_coef.plot.bar(ax=axs[1])
axs[0].set_xlabel(None)
axs[0].set_title('Feature Importance [Random Forest,XGBoost]')
axs[1].set_title('Uncertainty Coefficients [toxic]')
plt.tight_layout()
plt.show()
fig,axs=plt.subplots(1,2,figsize=(14,6),gridspec_kw=dict(width_ratios=[3,2]))
fi_oh_merge.query('importance_rf>0.01 | importance_xgb>0.01').plot.bar(ax=axs[0])
#filter out low coefficient values
unc_coef_oh[unc_coef_oh>0.05].plot.bar(ax=axs[1])
axs[0].set_xlabel(None)
axs[0].set_title('Feature Importance [Random Forest,XGBoost]')
axs[1].set_title('Uncertainty Coefficients[toxic]')
plt.tight_layout()

plt.show()
def multi_eli5(*explainers):
    html_str=''
    for expl in explainers:
        html_str +=expl._repr_html_().replace('style="border-collapse:collapse;','style="display:inline;border-collapse:collapse;')
    display_html(html_str,raw=True)
                                             
rfc_pi=PermutationImportance(rfc,random_state=RS,cv='prefit').fit(X_test,y_test)
rfc_oh_pi=PermutationImportance(rfc_oh,random_state=RS,cv='prefit').fit(Xoh_test,yoh_test)
rfc_weights=eli5.show_weights(rfc,feature_names=ftnames)
rfc_pi_weights=eli5.show_weights(rfc_pi,feature_names=ftnames)
multi_eli5(rfc_weights,rfc_pi_weights)
eli5.show_prediction(rfc,X_test.loc[RNIDX],feature_names=ftnames,show_feature_values=True)
eli5.show_prediction(rfc_oh,Xoh.loc[RNIDX],feature_names=ftnames_oh,show_feature_values=True,top=20)
xgbc_pi=PermutationImportance(xgbc,random_state=RS,cv='prefit').fit(X_test,y_test)
xgbc_oh_pi=PermutationImportance(xgbc_oh,random_state=RS,cv='prefit').fit(Xoh_test,yoh_test)
multi_eli5(eli5.show_weights(xgbc_pi,feature_names=ftnames),eli5.show_weights(xgbc_oh_pi,feature_names=ftnames_oh))
catname_map={i:X[c].cat.categories.values for i,c in enumerate(X)}

def strip_html(htmldoc,strip_tags=['html','meta','head','body'],outfile=None,verbose=False):
    """Strip out HTML boilerplate tags but perserve inner content Only will strip out the first occurrence of each tag ,if multiple occurences are desired,function must be modified."""
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(htmldoc)
    for tag in strip_tags:
        rmtag=soup.find(tag)
        if rmtag is not None:
            rmtag.unwrap()
            if verbose:print(tag,'tags removed')
    stripped=soup.prettify()
    if outfile is not None:
        with open(outfile,'w',encoding='utf-8') as f:
            f.write(stripped)
            if verbose:
                print(f'file saved to:{outfile}')
    else:
        return stripped
limeparams=dict(training_data=X_enc.values,
                training_labels=y_enc.values,
               feature_names=ftnames,
               categorical_features=range(X.shape[1]),
               categorical_names=catname_map,
               class_names=['edible','poisonous'])
lte=lime.lime_tabular.LimeTabularExplainer(**limeparams)
limeparams_oh=dict(training_data=Xoh.values,
                  training_labels=y_enc.values,
                  feature_names=ftnames_oh,categorical_features=range(Xoh.shape[1]),class_names=['edible','poisonous'])
lte_oh=lime.lime_tabular.LimeTabularExplainer(**limeparams_oh)
lte_expl=lte.explain_instance(X_test.loc[RNIDX],rfc.predict_proba)
display_html(strip_html(lte_expl.as_html()),raw=True)
lte_expl_oh=lte_oh.explain_instance(Xoh.loc[RNIDX],rfc_oh.predict_proba)
display_html(strip_html(lte_expl_oh.as_html()),raw=True)
yv=y_enc[RNIDX];yv #True label of y @ RNIDX for indexing shap valeus
shap_xgbc=shap.TreeExplainer(xgbc)
shapvals_xgbc=shap_xgbc.shap_values(X_test,y_test)
shap.force_plot(shap_xgbc.expected_value,shapvals_xgbc[posidx],features=X.loc[RNIDX],link='logit')
fp_glb=shap.force_plot(shap_xgbc.expected_value,shapvals_xgbc[:25],features=X.iloc[:25],out_names='toxic',link='logit')
display_html(fp_glb.data,raw=True)
shap.summary_plot(shapvals_xgbc,X_test,feature_names=ftnames,class_names=['edible','poisonous'])
shap.summary_plot(shapvals_xgbc,X_test,feature_names=ftnames,class_names=['edible','poisonous'],plot_type='bar')
siv_xgbc=shap_xgbc.shap_interaction_values(X_test)
shap.summary_plot(siv_xgbc,X_test)
from shap.plots.dependence import *
def dependence_plot(ind, shap_values, features, feature_names=None, display_features=None,
                    interaction_index="auto",
                    color="#1E88E5", axis_color="#333333", cmap=None,
                    dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True):
    """ Create a SHAP dependence plot, colored by an interaction feature.
    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extenstion of the classical parital dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.
    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a string it is
        either the name of the feature to plot, or it can have the form "rank(int)" to specify
        the feature with that rank (ordered by mean absolute SHAP value over all the samples).
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).
    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).
    feature_names : list
        Names of the features (length # features).
    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values).
    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature can also be passed
        as a string. If "auto" then shap.common.approximate_interactions is used to pick what
        seems to be the strongest interaction (note that to find to true stongest interaction you
        need to compute the SHAP interaction values).
        
    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when feature
        is discrete.
    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.
    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.
    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.
    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
         In this case we do not create a Figure, otherwise we do.
    """

    if cmap is None:
        cmap = colors.red_blue
        
    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind else (6, 5)
        fig = pl.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    ind = convert_name(ind, shap_values, feature_names)
    
    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values, feature_names)
        ind2 = convert_name(ind[1], shap_values, feature_names)
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_plot(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=ind2, display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(shap_values.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)
    xv = features[oinds, ind].astype(np.float64)
    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
    name = feature_names[ind]

    # guess what other feature as the stongest interaction with the plotted feature
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, shap_values, features)[0]
    interaction_index = convert_name(interaction_index, shap_values, feature_names)
    categorical_interaction = False

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        cv = features[:, interaction_index]
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(np.float), 5)
        chigh = np.nanpercentile(cv.astype(np.float), 95)
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(np.float))
            chigh = np.nanmax(cv.astype(np.float))
            bounds = np.linspace(clow, chigh, int(chigh - clow + 2))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(np.sort(xvals)))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.ranf(size = len(xv))*jitter_amount) - (jitter_amount/2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = features[oinds, interaction_index].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        p = ax.scatter(
            xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
            cmap=cmap, alpha=alpha, vmin=clow, vmax=chigh,
            norm=color_norm, rasterized=len(xv) > 500
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(p, ticks=tick_positions)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20
        
        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, dict(rotation='vertical', fontsize=11))
    if show:
        with warnings.catch_warnings(): # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()
for i in range(5):
    dependence_plot(f'rank({i})',shapvals_xgbc,X_test,display_features=X.loc[X_test.index])
pd.crosstab(mush.odor,[mush.toxic])

def toxcor(X):
    return X.drop('toxic',1).apply(lambda x:x.cat.codes).corrwith(X.toxic.cat.codes,method=theils_u).sort_values(ascending=False)
odorN=mush[mush.odor=="n"].drop('veil_type',1)
toxcor(odorN)
pd.crosstab([odorN.toxic],[odorN.spore_print_color])
odorN_spcW=odorN[odorN.spore_print_color=='w']

toxcor(odorN_spcW)
odorN_spcW_habLD=odorN_spcW[odorN_spcW.habitat.isin(['l','d'])]
toxcor(odorN_spcW_habLD)
pd.crosstab([odorN_spcW_habLD.toxic],[odorN_spcW_habLD.stalk_root],)
odorN_spcW_habLD_stkB=odorN_spcW_habLD[odorN_spcW_habLD.stalk_root=='b']
toxcor(odorN_spcW_habLD_stkB)
pd.crosstab([odorN_spcW_habLD_stkB.toxic],[odorN_spcW_habLD_stkB.cap_color])
def logic_tree(X):
    preds=[]
    for i,r in X.iterrows():
        if r.odor in ['a','l']:
            preds.append(0)
        elif r.odor=='n':
            if r.spore_print_color=='r':
                preds.append(1)
            elif r.spore_print_color=='w':
                if r.habitat not in ['l','d']:
                    preds.append(0)
                else:
                    if r.stalk_root!='b':
                            preds.append(1)
                    else:
                        preds.append(1 if r.cap_color=='w' else 0)
            else:
                preds.append(0)
        else:
            preds.append(1)
    return preds
                    
ltpreds=logic_tree(X)
pplot_cm(y_enc,ltpreds,cmap='Blues')
