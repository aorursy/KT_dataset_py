# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.8f}'.format
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import time
import sklearn
import lightgbm as lgb
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print('Importing 2012')
df_2012=pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2012.csv')
print('Importing 2013')
df_2013=pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2013.csv')
print('Importing 2014')
df_2014=pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2014.csv')
print('Importing 2015')
df_2015=pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2015.csv')
print('Importing 2016')
df_2016=pd.read_csv('../input/reclamacoes-fundamentadas-sindec-2016.csv')
dfs=[df_2012,df_2013,df_2014,df_2015,df_2016]
for i in range(1,5):
    assert sum(dfs[i-1].columns!=dfs[i].columns)==0, "Mismatch...Columns for 1st={}, Columns for 2nd={}".format(dfs[i-1].columns,dfs[i].columns)
df_combined=pd.DataFrame(df_2012)
for df in dfs[1:]:
    df_combined=df_combined.append(df)
df_combined.shape
def view_df_stats(df):
    print("Shape of df={}".format(df.shape))
    print("Number of index levels:{}".format(df.index.nlevels))
    for i in range(df.index.nlevels):
        print("For index level {},unique values count={}".format(i,df.index.get_level_values(i).unique().shape[0]))
    print("Columns of df={}".format(df.columns))
    print("Null count= \n {}".format(df.isnull().sum()))
    print(df.describe())
    
df_combined.head()
df_combined=df_combined.rename(columns={'AnoCalendario':'year','DataArquivamento':'closing_date','DataAbertura':'opening_date',
                        'CodigoRegiao':'region_code','Regiao':'region_name','UF':'state',
                        'strRazaoSocial':'business_legal_name','strNomeFantasia':'business_trade_name',
                       'Tipo':'type','NumeroCNPJ':'registration_number','RadicalCNPJ':'first_8d_registration_number',
                       'RazaoSocialRFB':'business_legal_name_federal','NomeFantasiaRFB':'business_trade_name_federal',
                       'CNAEPrincipal':'business_activity_code','DescCNAEPrincipal':'business_activity_description','Atendida':'Resolved','CodigoAssunto':'complaint_subject_code','DescricaoAssunto':'complaint_subject_desc','CodigoProblema':'issue_code','DescricaoProblema':'issue_description','SexoConsumidor':'gender_consumer','FaixaEtariaConsumidor':'age_group_consumer','CEPConsumidor':'zip_code_consumer'})
df_combined=df_combined.reset_index()
view_df_stats(df_combined)
del df_2012
del df_2013
del df_2014
del df_2015
del df_2016
non_useful_cols=['year','index','business_trade_name','first_8d_registration_number','business_trade_name_federal',
                'business_trade_name_federal','registration_number','business_legal_name_federal',
                'business_activity_description','complaint_subject_desc','issue_description','business_legal_name','region_name']
df_combined=df_combined.drop(non_useful_cols,axis=1)
df_combined.columns
view_df_stats(df_combined)
df_combined.head()
df_combined.type=df_combined.type.apply(lambda x:'Business' if x==1 else 'Person')
df_combined.type.value_counts()
print(sum((df_combined.type=='Person') & (df_combined.business_activity_code.isnull())))
df_combined.loc[df_combined.type=='Person','business_activity_code']=-1
sum(df_combined.business_activity_code==-1)
count_pre_drop_nulls_df_combined=df_combined.shape[0]
df_combined=df_combined.dropna()
df_combined.shape[0]/count_pre_drop_nulls_df_combined
df_combined.dtypes
df_combined.opening_date=pd.to_datetime(df_combined.opening_date)
df_combined.closing_date=pd.to_datetime(df_combined.closing_date)
cat_columns={'region_code':'category','business_activity_code':'category','complaint_subject_code':'category','issue_code':'category','state':'category','type':'category','Resolved':'category','gender_consumer':'category','age_group_consumer':'category','zip_code_consumer':'category'}
df_combined=df_combined.astype(cat_columns)
df_combined.dtypes
df_combined.type.value_counts()
start_time=time.time()
df_combined['time_elapsed_in_days']=(df_combined.closing_date-df_combined.opening_date).apply(lambda x:x.days)
end_time=time.time()
print("Elapsed time:{} sec for processing {} rows".format(end_time-start_time,df_combined.shape[0]))
sum_neg=sum(df_combined.time_elapsed_in_days<0)
print(sum_neg)
sum_neg/df_combined.shape[0]
df_combined=df_combined.drop(df_combined[df_combined.time_elapsed_in_days<0].index)
df_combined.dtypes
df_combined.head()
df_combined.Resolved.value_counts()/df_combined.shape[0]
ser_closing_date_timeseries=pd.Series(df_combined.Resolved.values,df_combined.closing_date)
ser_opening_date_timeseries=pd.Series(df_combined.Resolved.values,df_combined.opening_date)
ser_closing_date_timeseries_solved=ser_closing_date_timeseries[ser_closing_date_timeseries=='S']
ser_closing_date_timeseries_unsolved=ser_closing_date_timeseries[ser_closing_date_timeseries=='N']
ser_opening_date_timeseries_solved=ser_opening_date_timeseries[ser_opening_date_timeseries=='S']
ser_opening_date_timeseries_unsolved=ser_opening_date_timeseries[ser_opening_date_timeseries=='N']
ser_closing_date_timeseries.head()
def plotly_plot_dates(ser0,title,ser1=None,ser0_name='',ser1_name=''):
    trace0=go.Scatter(x=ser0.index,
                      y=ser0,
                      mode='lines',
                      hoverinfo='x+y',
                     name=ser0_name)
    trace1=go.Scatter(x=ser1.index,
                     y=ser1,
                     mode='lines',
                     hoverinfo='x+y',
                     name=ser1_name)
    layout=dict(title=title)
    py.iplot(dict(data=[trace0,trace1],layout=layout))
plotly_plot_dates(ser_opening_date_timeseries_solved.resample('AS').count(),title='Solved and Unsolved Trends For Opening Date Year Wise ',ser1=ser_opening_date_timeseries_unsolved.resample('AS').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_opening_date_timeseries_solved.resample('Q').count(),title='Solved and Unsolved Trends For Opening Date Quaterly',ser1=ser_opening_date_timeseries_unsolved.resample('Q').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_opening_date_timeseries_solved.resample('M').count(),title='Solved and Unsolved Trends For Opening Date Monthly',ser1=ser_opening_date_timeseries_unsolved.resample('M').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_opening_date_timeseries_solved.resample('D').count(),title='Solved and Unsolved Trends For Opening Date Day Wise',ser1=ser_opening_date_timeseries_unsolved.resample('D').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_opening_date_timeseries_solved.resample('M').count(),title='Solved and Unsolved Trends For Opening Date Day Wise ',ser1=ser_opening_date_timeseries_unsolved.resample('M').count(),ser0_name='solved',ser1_name='UnSolved')
plotly_plot_dates(ser_opening_date_timeseries_solved.resample('AS').count(),title='Solved and Unsolved Trends For Opening Date Day Wise ',ser1=ser_opening_date_timeseries_unsolved.resample('AS').count(),ser0_name='solved',ser1_name='UnSolved')
plotly_plot_dates(ser_closing_date_timeseries_solved.resample('AS').count(),title='Solved and Unsolved Trends For Closing Date Year Wise ',ser1=ser_closing_date_timeseries_unsolved.resample('AS').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_closing_date_timeseries_solved.resample('Q').count(),title='Solved and Unsolved Trends For Closing Date Quarter Wise ',ser1=ser_closing_date_timeseries_unsolved.resample('Q').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_closing_date_timeseries_solved.resample('M').count(),title='Solved and Unsolved Trends For Closing Date Month Wise ',ser1=ser_closing_date_timeseries_unsolved.resample('M').count(),ser0_name='Solved',ser1_name='UnSolved')
plotly_plot_dates(ser_closing_date_timeseries_solved.resample('D').count(),title='Solved and Unsolved Trends For Closing Date Daily ',ser1=ser_closing_date_timeseries_unsolved.resample('D').count(),ser0_name='Solved',ser1_name='UnSolved')
df_combined['opening_date_month']=df_combined.opening_date.dt.month
df_combined['opening_date_quarter']=df_combined.opening_date.dt.quarter
df_combined['opening_date_year']=df_combined.opening_date.dt.year
df_combined['closing_date_month']=df_combined.closing_date.dt.month
df_combined['closing_date_quarter']=df_combined.closing_date.dt.quarter
df_combined['closing_date_year']=df_combined.closing_date.dt.year
df_combined=df_combined.drop(['closing_date','opening_date'],axis=1)
df_combined.dtypes
cols={'opening_date_month':'category','opening_date_quarter':'category','opening_date_year':'category','closing_date_month':'category','closing_date_quarter':'category','closing_date_year':'category'}
df_combined=df_combined.astype(cols)
df_combined.dtypes
cat_cols=list(df_combined.columns)
cat_cols.remove('time_elapsed_in_days')
#Modifying dataset so that no two values from two different categorical columns are the same which will create problems during Label Binarization
def cat_cols_mod_df(df,cat_cols):
    for col in cat_cols:
        df[col]=df[col].apply(lambda x:str(col)+'_'+str(x))
cat_cols_mod_df(df_combined,cat_cols)
df_combined.head()
cols_dataset_X_predict_final_status=[x for x in df_combined.columns if x not in {'closing_date_month','closing_date_quarter','closing_date_year','time_elapsed_in_days','Resolved'}]
cols_dataset_X_predict_final_status
cols_dataset_Y_predict_final_status='Resolved'
dataset_X_predict_final_status_np_arr=df_combined.loc[:,cols_dataset_X_predict_final_status].values
dataset_Y_predict_final_status_np_arr=np.array(df_combined[cols_dataset_Y_predict_final_status].values)
print(dataset_X_predict_final_status_np_arr.shape)
print(dataset_Y_predict_final_status_np_arr.shape)
import sklearn.preprocessing
#Note that all of the 
sk_labelBinarizer_dataset_X_final_status=sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)
dataset_X_predict_final_status_cat_encoded_full=sk_labelBinarizer_dataset_X_final_status.fit_transform(dataset_X_predict_final_status_np_arr)
sk_labelEncoder_dataset_Y_final_status=sklearn.preprocessing.LabelEncoder()
dataset_Y_predict_final_status_full=sk_labelEncoder_dataset_Y_final_status.fit_transform(dataset_Y_predict_final_status_np_arr)
print('No. of dimensions after categorical encoding:{}'.format(dataset_X_predict_final_status_cat_encoded_full.shape[1]))
sk_labelEncoder_dataset_Y_final_status.classes_
sk_stratified_shuffle_split_holdout=sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,test_size=0.05)
for train_index,test_index in sk_stratified_shuffle_split_holdout.split(dataset_X_predict_final_status_cat_encoded_full,dataset_Y_predict_final_status_full):
    dataset_X_predict_final_status_cat_encoded_holdout=dataset_X_predict_final_status_cat_encoded_full[test_index]
    dataset_Y_predict_final_status_holdout=dataset_Y_predict_final_status_full[test_index]
    dataset_X_predict_final_status_cat_encoded=dataset_X_predict_final_status_cat_encoded_full[train_index]
    dataset_Y_predict_final_status=dataset_Y_predict_final_status_full[train_index]
    training_indexes=np.array(train_index,copy=True)
    holdout_indexes=np.array(test_index,copy=True)
print('Holdout Set Size ( For evaluation):{}'.format(dataset_X_predict_final_status_cat_encoded_holdout.shape[0]))
print('Training set size:{}'.format(dataset_X_predict_final_status_cat_encoded.shape[0]))
import sklearn.decomposition
sk_svd_x_dataset_viz=sklearn.decomposition.TruncatedSVD(n_components=2)
dataset_X_predict_final_status_SVD=sk_svd_x_dataset_viz.fit_transform(dataset_X_predict_final_status_cat_encoded)
print(dataset_X_predict_final_status_SVD.shape)
print(sk_svd_x_dataset_viz.explained_variance_ratio_)
print('Variance explained in 2D PCA:{}'.format(sk_svd_x_dataset_viz.explained_variance_ratio_.sum()))
dataset_X_predict_final_status_solved=dataset_X_predict_final_status_SVD[np.where(dataset_Y_predict_final_status==0)]
dataset_X_predict_final_status_not_solved=dataset_X_predict_final_status_SVD[np.where(dataset_Y_predict_final_status==1)]
print((dataset_X_predict_final_status_solved.shape,dataset_X_predict_final_status_not_solved.shape))
plt.scatter(dataset_X_predict_final_status_solved[:,0],dataset_X_predict_final_status_solved[:,1],c='g',label='Solved')
plt.scatter(dataset_X_predict_final_status_not_solved[:,0],dataset_X_predict_final_status_not_solved[:,1],c='y',label='Not Solved')
plt.show()
def plot_precision_recall_vs_threshold_plotly(precisions,recalls,thresholds,set_no):
    trace0=go.Scatter(
    x=thresholds,
    y=precisions,
    mode='lines',
    hoverinfo='x+y',
    name='Precision'
    )
    trace1=go.Scatter(
    x=thresholds,
    y=recalls,
    mode='lines',
    hoverinfo='x+y',
    name='Recall'
    )
    layout=go.Layout(
    title='Plot of precision vs recall for various thresholds for set no. {}'.format(set_no)
    )
    py.iplot(dict(data=[trace0,trace1],layout=layout),filename='precision vs recall')
def plot_precision_recall_vs_threshold_mpl(precisions,recalls,thresholds,set_no):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.title('Plot of precision vs recall for various thresholds for set no. {}'.format(set_no))
    plt.show()
def print_evaluation_metric_results(Y_pred,Y_truth):
    print("Confusion matrix: {}".format(sklearn.metrics.confusion_matrix(Y_truth,Y_pred)))
    print("Precision:{}".format(sklearn.metrics.precision_score(Y_truth,Y_pred)))
    print("Recall:{}".format(sklearn.metrics.recall_score(Y_truth,Y_pred)))
    print("F1:{}".format(sklearn.metrics.f1_score(Y_truth,Y_pred)))
def calc_classification_error_ratio(confusion_matrix):
    tn,fp,fn,tp=confusion_matrix.ravel()
    return (fp+fn)/(tn+fp+fn+tp)
%%time
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
from sklearn.metrics import make_scorer,log_loss
from sklearn.model_selection import learning_curve
sk_stratified_shuffle_split=sklearn.model_selection.StratifiedShuffleSplit(n_splits=3,test_size=0.1)
i=1
sk_sgd_final_status=sklearn.linear_model.SGDClassifier(loss="hinge",penalty="l2",verbose=0,n_jobs=-1)
start_time=time.time()
sgd_lc_train_sizes,sgd_lc_train_scores,sgd_lc_test_scores=learning_curve(sk_sgd_final_status,dataset_X_predict_final_status_cat_encoded,
                                                                        dataset_Y_predict_final_status,train_sizes=np.linspace(0.1,1.0,25),
                                                                        cv=sk_stratified_shuffle_split,scoring=make_scorer(log_loss,greater_is_better=False),
                                                                        exploit_incremental_learning=True)
end_time=time.time()
print("Time taken={} secs".format(end_time-start_time))
print(sgd_lc_train_sizes.shape)
sgd_lc_train_scores=np.average(sgd_lc_train_scores,axis=1)
sgd_lc_test_scores=np.average(sgd_lc_test_scores,axis=1)
print(sgd_lc_train_scores.shape)
print(sgd_lc_test_scores.shape)
trace0=go.Scatter(x=sgd_lc_train_sizes,
                      y=sgd_lc_train_scores,
                      mode='lines',
                     name='Train scores (neg logloss)')
trace1=go.Scatter(x=sgd_lc_train_sizes,
                     y=sgd_lc_test_scores,
                     mode='lines',
                     name='Test scores (neg logloss) ')
layout=dict(title='Neg Logloss score as a function of DataSet Size')
py.iplot(dict(data=[trace0,trace1],layout=layout))
def calc_optimal_threshold_score(precisions,recalls,thresholds,tol=0.0001):
    try:
        return thresholds[np.where(precisions==recalls)[0][0]]
    except IndexError:
        return thresholds[np.where(np.abs(precisions-recalls)<tol)[0][0]]
%%time
y_pred_sgd_optimal_threshold=0
sk_stratified_shuffle_split=sklearn.model_selection.StratifiedShuffleSplit(n_splits=3,test_size=0.1)
i=1
sk_sgd_final_status=sklearn.base.clone(sk_sgd_final_status)
for train_index,test_index in sk_stratified_shuffle_split.split(dataset_X_predict_final_status_cat_encoded,dataset_Y_predict_final_status):
    X_train, X_test = dataset_X_predict_final_status_cat_encoded[train_index], dataset_X_predict_final_status_cat_encoded[test_index]
    Y_train, Y_test = dataset_Y_predict_final_status[train_index], dataset_Y_predict_final_status[test_index]
    start=time.time()
    sk_sgd_final_status.fit(X_train,Y_train)
    end=time.time()
    print("----Split no. {}----".format(i))
    print("Training took {} sec".format(end-start))
    Y_pred=sk_sgd_final_status.predict(X_test)
    Y_train_pred=sk_sgd_final_status.predict(X_train)
    Y_decision_score=sk_sgd_final_status.decision_function(X_test)
    precisions,recalls,thresholds=sklearn.metrics.precision_recall_curve(Y_test,Y_decision_score)
    y_pred_sgd_optimal_threshold+=calc_optimal_threshold_score(precisions,recalls,thresholds)
    plot_precision_recall_vs_threshold_mpl(precisions,recalls,thresholds,i)
    #Add 2D plot
    print("Confusion matrix test set: {}".format(sklearn.metrics.confusion_matrix(Y_test,Y_pred)))
    print("Precision test set:{}".format(sklearn.metrics.precision_score(Y_test,Y_pred)))
    print("Recall test set:{}".format(sklearn.metrics.recall_score(Y_test,Y_pred)))
    print("F1 test set:{}".format(sklearn.metrics.f1_score(Y_test,Y_pred)))
    print("Classification error test set:{}".format(calc_classification_error_ratio(sklearn.metrics.confusion_matrix(Y_test,Y_pred))))
    print("Confusion matrix train set: {}".format(sklearn.metrics.confusion_matrix(Y_train,Y_train_pred)))
    print("Precision train set:{}".format(sklearn.metrics.precision_score(Y_train,Y_train_pred)))
    print("Recall train set:{}".format(sklearn.metrics.recall_score(Y_train,Y_train_pred)))
    print("F1 train set:{}".format(sklearn.metrics.f1_score(Y_train,Y_train_pred)))
    print("Classification error train set:{}".format(calc_classification_error_ratio(sklearn.metrics.confusion_matrix(Y_train,Y_train_pred))))
    i+=1
y_pred_sgd_optimal_threshold/=3
y_sgd_pred_holdout=sk_sgd_final_status.predict(dataset_X_predict_final_status_cat_encoded_holdout)
print('Holdout set metrics for SGD(with original predictions):')
print_evaluation_metric_results(y_sgd_pred_holdout,dataset_Y_predict_final_status_holdout)
y_sgd_decision_score_holdout_2=sk_sgd_final_status.decision_function(dataset_X_predict_final_status_cat_encoded_holdout)
y_sgd_pred_holdout_2=(y_sgd_decision_score_holdout_2>y_pred_sgd_optimal_threshold).astype(int)
print('Holdout set metrics for SGD(for balanced precision and recall):')
print_evaluation_metric_results(y_sgd_pred_holdout_2,dataset_Y_predict_final_status_holdout)
%%time
import sklearn.cluster
sk_kmeans_final_status_predict=sklearn.cluster.MiniBatchKMeans(n_clusters=4,verbose=0,max_no_improvement=10)
dataset_predict_final_status_kmeans_clusters=sk_kmeans_final_status_predict.fit_predict(dataset_X_predict_final_status_cat_encoded)
np.unique(dataset_predict_final_status_kmeans_clusters,return_counts=True)
col_scheme=('r','b','g','y')
for i in (0,1,2,3):
    plt.scatter(dataset_X_predict_final_status_SVD[np.where(dataset_predict_final_status_kmeans_clusters==i)][:,0],dataset_X_predict_final_status_SVD[np.where(dataset_predict_final_status_kmeans_clusters==i)][:,1],c=col_scheme[i],label='Cluster {}'.format(i))
plt.show()
pd_dataset_X_predict_final_status_kmeans_clusters_ser=pd.Series(dataset_predict_final_status_kmeans_clusters)
pd_dataset_X_predict_final_status_kmeans_clusters_ser=pd_dataset_X_predict_final_status_kmeans_clusters_ser.transform(lambda x:'clus_'+str(x))
print(pd_dataset_X_predict_final_status_kmeans_clusters_ser.value_counts())
dataset_X_predict_final_status_kmeans_clusters_mod=pd_dataset_X_predict_final_status_kmeans_clusters_ser.values
dataset_X_predict_final_status_kmeans_clusters_mod.shape
from scipy.sparse import hstack
sk_labelBinarizer_kmeans_res=sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)
dataset_X_predict_final_status_kmeans_clusters_mod_cat_encoded=sk_labelBinarizer_kmeans_res.fit_transform(
                                            dataset_X_predict_final_status_kmeans_clusters_mod.reshape(-1,1))
print(sk_labelBinarizer_kmeans_res.classes_)
dataset_X_predict_final_status_kmeans_clusters_mod_cat_encoded
dataset_X_predict_final_status_cat_encoded_2=hstack([dataset_X_predict_final_status_cat_encoded,
                                                    dataset_X_predict_final_status_kmeans_clusters_mod_cat_encoded]).tocsr()
dataset_X_predict_final_status_cat_encoded_2
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
from sklearn.metrics import make_scorer,f1_score,log_loss
from sklearn.model_selection import learning_curve
i=1
sk_sgd_final_status_2=sklearn.linear_model.SGDClassifier(loss="hinge",penalty="l2",verbose=0)
start_time=time.time()
sgd_lc_train_sizes_2,sgd_lc_train_scores_2,sgd_lc_test_scores_2=learning_curve(sk_sgd_final_status_2,dataset_X_predict_final_status_cat_encoded_2,
                                                                        dataset_Y_predict_final_status,train_sizes=np.linspace(0.1,1.0,25),
                                                                        cv=sk_stratified_shuffle_split,scoring=make_scorer(log_loss,greater_is_better=False),
                                                                        exploit_incremental_learning=True)
end_time=time.time()
print("Time taken={} secs".format(end_time-start_time))
print(sgd_lc_train_sizes_2.shape)
sgd_lc_train_scores_2=np.average(sgd_lc_train_scores_2,axis=1)
sgd_lc_test_scores_2=np.average(sgd_lc_test_scores_2,axis=1)
print(sgd_lc_train_scores_2.shape)
print(sgd_lc_test_scores_2.shape)
trace0=go.Scatter(x=sgd_lc_train_sizes_2,
                      y=sgd_lc_train_scores_2,
                      mode='lines',
                     name='Train scores (f1)')
trace1=go.Scatter(x=sgd_lc_train_sizes_2,
                     y=sgd_lc_test_scores_2,
                     mode='lines',
                     name='Test scores (f1)')
layout=dict(title='Negative Logloss as a function of DataSet Size')
py.iplot(dict(data=[trace0,trace1],layout=layout))

%%time
y_pred_sgd_optimal_threshold_2=0
sk_stratified_shuffle_split=sklearn.model_selection.StratifiedShuffleSplit(n_splits=3,test_size=0.1)
i=1
sk_sgd_final_status=sklearn.base.clone(sk_sgd_final_status_2)
for train_index,test_index in sk_stratified_shuffle_split.split(dataset_X_predict_final_status_cat_encoded_2,dataset_Y_predict_final_status):
    X_train, X_test = dataset_X_predict_final_status_cat_encoded_2[train_index], dataset_X_predict_final_status_cat_encoded_2[test_index]
    Y_train, Y_test = dataset_Y_predict_final_status[train_index], dataset_Y_predict_final_status[test_index]
    start=time.time()
    sk_sgd_final_status.fit(X_train,Y_train)
    end=time.time()
    print("----Split no. {}----".format(i))
    print("Training took {} sec".format(end-start))
    Y_pred=sk_sgd_final_status.predict(X_test)
    Y_train_pred=sk_sgd_final_status.predict(X_train)
    Y_decision_score=sk_sgd_final_status.decision_function(X_test)
    precisions,recalls,thresholds=sklearn.metrics.precision_recall_curve(Y_test,Y_decision_score)
    y_pred_sgd_optimal_threshold_2+=calc_optimal_threshold_score(precisions,recalls,thresholds)
    plot_precision_recall_vs_threshold_mpl(precisions,recalls,thresholds,i)
    #Add 2D plot
    print("Confusion matrix test set: {}".format(sklearn.metrics.confusion_matrix(Y_test,Y_pred)))
    print("Precision test set:{}".format(sklearn.metrics.precision_score(Y_test,Y_pred)))
    print("Recall test set:{}".format(sklearn.metrics.recall_score(Y_test,Y_pred)))
    print("F1 test set:{}".format(sklearn.metrics.f1_score(Y_test,Y_pred)))
    print("Classification error test set:{}".format(calc_classification_error_ratio(sklearn.metrics.confusion_matrix(Y_test,Y_pred))))
    print("Confusion matrix train set: {}".format(sklearn.metrics.confusion_matrix(Y_train,Y_train_pred)))
    print("Precision train set:{}".format(sklearn.metrics.precision_score(Y_train,Y_train_pred)))
    print("Recall train set:{}".format(sklearn.metrics.recall_score(Y_train,Y_train_pred)))
    print("F1 train set:{}".format(sklearn.metrics.f1_score(Y_train,Y_train_pred)))
    print("Classification error train set:{}".format(calc_classification_error_ratio(sklearn.metrics.confusion_matrix(Y_train,Y_train_pred))))
    i+=1
y_pred_sgd_optimal_threshold_2/=3
print(y_pred_sgd_optimal_threshold_2)
def kmeans_output_as_a_feature_pipeline(sk_kmeans,existing_X):
    kmeans_out=sk_kmeans.predict(existing_X)
    print(np.unique(kmeans_out,return_counts=True))
    pd_kmeans_out=pd.Series(kmeans_out)
    pd_kmeans_out=pd_kmeans_out.transform(lambda x:'clus_'+str(x))
    print(pd_kmeans_out.value_counts())
    np_kmeans_out=pd_kmeans_out.values
    from scipy.sparse import hstack
    sk_mlb=sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)
    out_bin=sk_mlb.fit_transform(np_kmeans_out.reshape(-1,1))
    output=hstack([existing_X,out_bin]).tocsr()
    return output
y_sgd_pred_holdout_2=sk_sgd_final_status.predict(kmeans_output_as_a_feature_pipeline(sk_kmeans_final_status_predict,dataset_X_predict_final_status_cat_encoded_holdout))
print('Holdout set metrics for SGD(with original predictions):')
print_evaluation_metric_results(y_sgd_pred_holdout_2,dataset_Y_predict_final_status_holdout)
y_sgd_decision_score_holdout_2=sk_sgd_final_status.decision_function(kmeans_output_as_a_feature_pipeline(sk_kmeans_final_status_predict,dataset_X_predict_final_status_cat_encoded_holdout))
y_sgd_pred_holdout_2=(y_sgd_decision_score_holdout_2>y_pred_sgd_optimal_threshold).astype(int)
print('Holdout set metrics for SGD(for balanced precision and recall):')
print_evaluation_metric_results(y_sgd_pred_holdout_2,dataset_Y_predict_final_status_holdout)
%%time
import sklearn.cluster
sk_kmeans_final_status_predict_2=sklearn.cluster.MiniBatchKMeans(n_clusters=2,verbose=0,max_no_improvement=10)
#print("Started at {} sec".format(start))
dataset_predict_final_status_kmeans_clusters=sk_kmeans_final_status_predict_2.fit_predict(dataset_X_predict_final_status_cat_encoded)
#print("Ended at {} sec".format(end))
#print("Time taken: {} sec".format(end-start))

print("Training set results")
print(np.unique(dataset_predict_final_status_kmeans_clusters,return_counts=True))
#Since there were more solved than unsolved, we will assume that 0th class is Solved here and 1 is Unsolved
print_evaluation_metric_results((dataset_predict_final_status_kmeans_clusters==0).astype(int),dataset_Y_predict_final_status)
print("Holdout set results")
kmeans_y_pred_holdout=sk_kmeans_final_status_predict_2.predict(dataset_X_predict_final_status_cat_encoded_holdout)
print(np.unique(kmeans_y_pred_holdout,return_counts=True))
#Since there were more solved than unsolved, we will assume that 0th class is Solved here and 1 is Unsolved
print_evaluation_metric_results((dataset_Y_predict_final_status_holdout==0).astype(int),kmeans_y_pred_holdout)
%%time
i=0
lgbm_optimal_threshold=0
#start=time.time()
sk_stratified_shuffle_split_2=sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,test_size=0.1)
df_X_data_gb=df_combined.loc[training_indexes,cols_dataset_X_predict_final_status]
for train_index,test_index in sk_stratified_shuffle_split_2.split(df_X_data_gb,dataset_Y_predict_final_status):
    data_lgb_train=lgb.Dataset(df_X_data_gb.loc[train_index,:],dataset_Y_predict_final_status[train_index])
    data_lgb_test=lgb.Dataset(df_X_data_gb.loc[test_index,:],dataset_Y_predict_final_status[test_index])
    params = {
        'num_leaves' : 4096,
        'learning_rate':0.003,
        'metric':'binary',
        'objective':'binary',
        'early_stopping_round': 100,
        'max_depth':10,
        'bagging_fraction':0.5,
        'feature_fraction':1,
        'verbose' : 1,
    }
    start=time.time()
    clf = lgb.train(params, data_lgb_train,num_boost_round=500,valid_sets=data_lgb_test,verbose_eval=100)
    end=time.time()
    print("Training took {} sec for {}th round".format(end-start,i))
    lgbm_predict_train_score=clf.predict(df_X_data_gb.loc[train_index,:])
    lgbm_predict_train_class=lgbm_predict_train_score.round().astype(int)
    lgbm_predict_test_score=clf.predict(df_X_data_gb.loc[test_index,:])
    lgbm_predict_test_class=lgbm_predict_test_score.round().astype(int)
    precisions,recalls,thresholds=sklearn.metrics.precision_recall_curve(dataset_Y_predict_final_status[test_index],lgbm_predict_test_score)
    #lgbm_optimal_threshold+=calc_optimal_threshold_score(precisions,recalls,thresholds,0.001)
    #y_pred_sgd_optimal_threshold+=calc_optimal_threshold_score(precisions,recalls,thresholds)
    plot_precision_recall_vs_threshold_mpl(precisions,recalls,thresholds,i)
    print("Train evaluation metrics:")
    print_evaluation_metric_results(lgbm_predict_train_class,dataset_Y_predict_final_status[train_index])
    print("Test evaluation metrics:")
    print_evaluation_metric_results(lgbm_predict_test_class,dataset_Y_predict_final_status[test_index])
#end=time.time()
#print("It took {} sec".format(end-start))
%%time
data_lgb_predict_holdout_score = clf.predict(df_combined.loc[holdout_indexes,cols_dataset_X_predict_final_status])
data_lgb_predict_holdout_class = data_lgb_predict_holdout_score.round().astype(int)
print_evaluation_metric_results(data_lgb_predict_holdout_class,dataset_Y_predict_final_status_holdout)
