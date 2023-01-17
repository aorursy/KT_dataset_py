import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import IPython
from IPython import display 
import ast

import scipy.stats as stats

def anova(frame,qualitative,target_column):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls][target_column].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

def spearman(frame, features,targe_column):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame[targe_column], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
def ave_top_5_per_feature(frame, features):
    top = pd.DataFrame(columns=['feature','level','mean_test_score','std_test_score','MeanErrorDifference'])
    for feat in features:
        for level in frame[feat].unique():
            values={'feature':feat,
                    'level':level,
                    'mean_test_score':frame[frame[feat]==level].sort_values('mean_test_score',ascending=False).head().mean()['mean_test_score'],
                    'std_test_score':frame[frame[feat]==level].sort_values('mean_test_score',ascending=False).head().mean()['std_test_score'],
                    'MeanErrorDifference':frame[frame[feat]==level].sort_values('mean_test_score',ascending=False).head().mean()['MeanErrorDifference']
                   }
            temp_df = pd.DataFrame(values,columns=['feature','level','mean_test_score','std_test_score','MeanErrorDifference'],index=[0])
            top=pd.concat([top,temp_df])
    return top.sort_values('mean_test_score',ascending=False)
data = pd.read_csv('../input/titanicmodelcvresults/CV_LR_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score',
       'split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical.remove('mean_test_score')
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.sort_values('mean_test_score',ascending=False)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.75)]
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_penalty", y="mean_test_score", hue="param_solver", data=data,
                   size=6, kind="bar", palette="muted")
data.groupby('param_penalty').mean()
spearman(data,numerical,'mean_test_score')
data[data['param_penalty']=='l2'].sort_values('mean_test_score',ascending=False).head(5).describe()
data = pd.read_csv('../input/titanicmodelcvresults/CV_LR_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical.remove('mean_test_score')
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.sort_values('mean_test_score',ascending=False)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.75)]
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_penalty", y="mean_test_score", hue="param_solver", data=data,
                   size=6, kind="bar", palette="muted")
data.groupby('param_penalty').mean()
spearman(data,numerical,'mean_test_score')
data[data['param_penalty']=='l2'].sort_values('mean_test_score',ascending=False).head(5).describe()
data = pd.read_csv('../input/titanicmodelcvresults/CV_LR_X_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.75)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_LR__penalty", y="mean_test_score", hue="param_LR__solver", data=data,
                   size=6, kind="bar", palette="muted")
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_LR__solver",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_LR__solver", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_LR_X_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.75)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_LR__penalty", y="mean_test_score", hue="param_LR__solver", data=data,
                   size=6, kind="bar", palette="muted")
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_LR__solver",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_LR__solver", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_ABC_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_algorithm", y="mean_test_score", data=data,
                   size=6, kind="bar", palette="muted")
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_algorithm",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_algorithm", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_ABC_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_algorithm", y="mean_test_score", data=data,
                   size=6, kind="bar", palette="muted")
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_algorithm",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_algorithm", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_GBC_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
data.fillna('All',inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_max_features", y="mean_test_score", hue='param_loss',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_max_features", y="mean_test_score", hue="param_loss", data=data, split=True,
               inner="quart")
sns.despine(left=True)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_max_features",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_max_features", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_GBC_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
data.fillna('All',inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_max_features", y="mean_test_score", hue='param_loss',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_max_features", y="mean_test_score", hue="param_loss", data=data, split=True,
               inner="quart")
sns.despine(left=True)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_max_features",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_max_features", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_RFC_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.fillna('All',inplace=True)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_criterion", y="mean_test_score",data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_criterion", y="mean_test_score", data=data,
               inner="quart")
sns.despine(left=True)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_max_features",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_max_features", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_RFC_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.fillna('All',inplace=True)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_criterion", y="mean_test_score",hue='param_max_features',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_criterion", y="mean_test_score",hue='param_max_features', split='true',data=data,
               inner="quart")
sns.despine(left=True)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_max_features",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_max_features", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_KNN_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.fillna('All',inplace=True)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_weights", y="mean_test_score",hue='param_algorithm',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_weights", y="mean_test_score",hue='param_algorithm',data=data,
               inner="quart")
sns.despine(left=True)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_weights",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_weights", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_KNN_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.fillna('All',inplace=True)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_weights", y="mean_test_score",hue='param_algorithm',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_weights", y="mean_test_score",hue='param_algorithm',data=data,
               inner="quart")
sns.despine(left=True)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_weights",y_vars='mean_test_score',x_vars=numerical_param)
sns.pairplot(data, x_vars=numerical_param, y_vars=["mean_test_score","mean_train_score"],hue="param_weights", size=5, aspect=.8, kind="reg");
data = pd.read_csv('../input/titanicmodelcvresults/CV_SVM_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.fillna('All',inplace=True)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_kernel", y="mean_test_score",hue='param_decision_function_shape',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_kernel", y="mean_test_score",hue='param_decision_function_shape',data=data,
               inner="quart")
sns.despine(left=True)
spearman(data[data['param_kernel']=='poly'],numerical_param,'mean_test_score')
spearman(data[data['param_kernel']=='poly'],['param_C'],'mean_test_score')
sns.pairplot(data, hue="param_kernel",y_vars='mean_test_score',x_vars=numerical_param)
data = pd.read_csv('../input/titanicmodelcvresults/CV_SVM_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params','split0_test_score', 'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'split5_test_score', 'split5_train_score',
       'split6_test_score', 'split6_train_score', 'split7_test_score',
       'split7_train_score', 'split8_test_score', 'split8_train_score',
       'split9_test_score', 'split9_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data.fillna('All',inplace=True)
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_kernel", y="mean_test_score",hue='param_decision_function_shape',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_kernel", y="mean_test_score",hue='param_decision_function_shape',data=data,
               inner="quart")
sns.despine(left=True)
spearman(data[data['param_kernel']=='poly'],numerical_param,'mean_test_score')
spearman(data[data['param_kernel']=='poly'],['param_C'],'mean_test_score')
sns.pairplot(data, hue="param_kernel",y_vars='mean_test_score',x_vars=numerical_param)
data = pd.read_csv('../input/titanicmodelcvresults/CV_DNN_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params', 'split0_test_score',
       'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
shape_dict={'none':0,'one':1,'two':2}
data.replace(shape_dict,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_activation", y="mean_test_score",hue='param_shape',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_activation", y="mean_test_score",hue='param_shape',data=data,
               inner="quart")
sns.despine(left=True)
sns.factorplot(x="param_activation", y="mean_test_score", hue="param_shape",col="param_optimizer", data=data, kind="box", size=4, aspect=1,row='param_neurons');
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_activation",y_vars=["mean_test_score","mean_train_score"],x_vars=numerical_param,kind='reg')
data = pd.read_csv('../input/titanicmodelcvresults/CV_DNN_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params', 'split0_test_score',
       'split0_train_score','split1_test_score',
       'split1_train_score', 'split2_test_score', 'split2_train_score',
       'split3_test_score', 'split3_train_score', 'split4_test_score',
       'split4_train_score', 'std_fit_time',
       'std_score_time'],axis=1,inplace=True)
shape_dict={'none':0,'one':1,'two':2}
data.replace(shape_dict,inplace=True)
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
g = sns.factorplot(x="param_activation", y="mean_test_score",hue='param_optimizer',data=data,
                   size=6, kind="bar", palette="muted")
sns.violinplot(x="param_activation", y="mean_test_score",hue='param_optimizer',data=data,
               inner="quart")
sns.despine(left=True)
sns.factorplot(x="param_activation", y="mean_test_score", hue="param_shape",col="param_optimizer", data=data, kind="box", size=4, aspect=1,row='param_neurons');
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_activation",y_vars=["mean_test_score","mean_train_score"],x_vars=numerical_param,kind='reg')
data = pd.read_csv('../input/titanicmodelcvresults/CV_MLP_1.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params',
       'split0_test_score', 'split0_train_score', 'split1_test_score', 'split1_train_score',
       'split2_test_score', 'split2_train_score', 'split3_test_score',
       'split3_train_score', 'split4_test_score', 'split4_train_score','std_fit_time',
       'std_score_time'],axis=1,inplace=True)
data['param_n_hidden_layers']=data['param_hidden_layer_sizes'].apply(ast.literal_eval).apply(len)
data['param_n_hidden_cells']=data['param_hidden_layer_sizes'].apply(ast.literal_eval).apply(lambda x: x[0])
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
sns.factorplot(x="param_activation", y="mean_test_score",col="param_n_hidden_cells",row='param_solver',hue="param_n_hidden_layers", data=data, kind="box", size=4, aspect=1)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_activation",y_vars=["mean_test_score","mean_train_score"],x_vars=numerical_param,kind="reg",aspect=1)
data = pd.read_csv('../input/titanicmodelcvresults/CV_MLP_2.csv',index_col='Unnamed: 0')
data['MeanErrorDifference']=2*(data['mean_train_score']*data['mean_test_score'])/(data['mean_train_score']+data['mean_test_score'])
data.head().columns
data.drop(['mean_score_time','params',
       'split0_test_score', 'split0_train_score', 'split1_test_score', 'split1_train_score',
       'split2_test_score', 'split2_train_score', 'split3_test_score',
       'split3_train_score', 'split4_test_score', 'split4_train_score','std_fit_time',
       'std_score_time'],axis=1,inplace=True)
data['param_n_hidden_layers']=data['param_hidden_layer_sizes'].apply(ast.literal_eval).apply(len)
data['param_n_hidden_cells']=data['param_hidden_layer_sizes'].apply(ast.literal_eval).apply(lambda x: x[0])
numerical = [f for f in data.columns if data.dtypes[f] != 'object']
numerical_measures=[x for x in numerical if 'param' not in x]
numerical_param =[x for x in numerical if 'param' in x]
categorical = [f for f in data.columns if data.dtypes[f] == 'object']
data=data[data['mean_test_score']>=data['mean_test_score'].quantile(q=0.9)]
data.sort_values('mean_test_score',ascending=False)
a = anova(data,categorical,'mean_test_score')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
sns.factorplot(x="param_activation", y="mean_test_score",col="param_n_hidden_cells",row='param_n_hidden_layers', data=data, kind="box", size=4, aspect=1)
spearman(data,numerical_param,'mean_test_score')
sns.pairplot(data, hue="param_activation",y_vars=["mean_test_score","mean_train_score"],x_vars=numerical_param,kind="reg",aspect=1)
