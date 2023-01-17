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
view_df_stats(df_combined)
df_combined=df_combined.rename(columns={'AnoCalendario':'year','DataArquivamento':'closing_date','DataAbertura':'opening_date',
                        'CodigoRegiao':'region_code','Regiao':'region_name','UF':'state',
                        'strRazaoSocial':'business_legal_name','strNomeFantasia':'business_trade_name',
                       'Tipo':'type','NumeroCNPJ':'registration_number','RadicalCNPJ':'first_8d_registration_number',
                       'RazaoSocialRFB':'business_legal_name_federal','NomeFantasiaRFB':'business_trade_name_federal',
                       'CNAEPrincipal':'business_activity_code','DescCNAEPrincipal':'business_activity_description','Atendida':'Resolved','CodigoAssunto':'complaint_subject_code','DescricaoAssunto':'complaint_subject_desc','CodigoProblema':'issue_code','DescricaoProblema':'issue_description','SexoConsumidor':'gender_consumer','FaixaEtariaConsumidor':'age_group_consumer','CEPConsumidor':'zip_code_consumer'})
df_combined=df_combined.reset_index()
df_combined.columns
non_useful_cols=['business_trade_name','first_8d_registration_number','business_trade_name_federal',
                'business_trade_name_federal','registration_number','business_legal_name_federal']
del df_2012
del df_2013
del df_2014
del df_2015
del df_2016
df_combined=df_combined.drop(non_useful_cols,axis=1)
view_df_stats(df_combined)
df_combined.dtypes
cat_columns=['region_code','region_name','state','type','business_activity_code','Resolved','complaint_subject_code','issue_code','gender_consumer','age_group_consumer','zip_code_consumer']
for col in cat_columns:
    df_combined[col].astype('category')
df_combined.shape
df_combined.dtypes
df_combined.opening_date=pd.to_datetime(df_combined.opening_date)
df_combined.closing_date=pd.to_datetime(df_combined.closing_date)
df_combined.dtypes
df_combined.head()
#TEMP
print(sum(df_combined.business_activity_code==0))
print(sum((df_combined.business_activity_code.isnull()) & (df_combined.business_activity_description.notnull())))
print(sum((df_combined.business_activity_code.notnull()) & (df_combined.business_activity_description.isnull())))
print(df_combined.business_activity_code[(df_combined.business_activity_code.notnull()) & (df_combined.business_activity_description.isnull())].unique().shape[0])
df_combined.business_activity_code=df_combined.business_activity_code.fillna(-1)
df_combined.issue_code=df_combined.issue_code.fillna(-1)
df_combined.issue_description=df_combined.issue_description.fillna('UNKNOWN')
df_combined.zip_code_consumer=df_combined.zip_code_consumer.fillna('UNKNOWN')
df_combined.gender_consumer=df_combined.gender_consumer.fillna('UNKNOWN')
df_combined.business_legal_name=df_combined.business_legal_name.fillna('UNKNOWN')
df_combined.isnull().sum()
business_activity_code_desc_null_mismatch_value_counts=df_combined.business_activity_code[(df_combined.business_activity_code.notnull()) & (df_combined.business_activity_description.isnull()) & (df_combined.business_activity_code>-1)].value_counts()
business_activity_code_desc_null_mismatch_value_counts_df=pd.DataFrame(business_activity_code_desc_null_mismatch_value_counts).rename(columns={'business_activity_code':'null_counts'})
alt_desc_name_ser='UNKNOWN_'+pd.Series(range(1,business_activity_code_desc_null_mismatch_value_counts_df.shape[0]+1)).astype(str)
alt_desc_name_ser.index=business_activity_code_desc_null_mismatch_value_counts_df.index
business_activity_code_desc_null_mismatch_value_counts_df['business_activity_desc_fill']=alt_desc_name_ser
business_activity_code_desc_null_mismatch_value_counts_df
#start=time.time()
for index,ser in business_activity_code_desc_null_mismatch_value_counts_df.iterrows():
    df_combined.loc[df_combined.business_activity_code==index,'business_activity_description']=ser['business_activity_desc_fill']
#end=time.time()
#print("Time taken for the op:{} secs".format(end-start))
df_combined.business_activity_description=df_combined.business_activity_description.fillna('UNKNOWN_-1')
df_combined.isnull().sum()
#Resolved Count
resolved_count_in_df_combined=(df_combined.Resolved=='S').sum()
resolved_perc_in_df_combined=resolved_count_in_df_combined/df_combined.shape[0]
print("Resolved Count={}".format(resolved_count_in_df_combined))
print("Resolved Perc={}".format(resolved_perc_in_df_combined))
non_resolved_count_in_df_combined=(df_combined.Resolved=='N').sum()
non_resolved_perc_in_df_combined=non_resolved_count_in_df_combined/df_combined.shape[0]
print("Not Resolved Count={}".format(non_resolved_count_in_df_combined))
print("Not Resolved Perc={}".format(non_resolved_perc_in_df_combined))
start_time=time.time()
df_combined['time_elapsed_indays']=(df_combined.closing_date-df_combined.opening_date).apply(lambda x:x.days)
end_time=time.time()
print("Elapsed time:{} sec for processing {} rows".format(end_time-start_time,df_combined.shape[0]))
df_combined.time_elapsed_indays.sort_values(ascending=True).head()
temp_sum_df_combined_time_elapsed_days_negative_count=sum(df_combined.time_elapsed_indays<0)
print(temp_sum_df_combined_time_elapsed_days_negative_count)
temp_sum_df_combined_time_elapsed_days_negative_count/df_combined.shape[0]
temp_time_elapsed_drop_rows_index=df_combined[df_combined.time_elapsed_indays<0].index
#print(temp_time_elapsed_drop_rows_index)
df_combined[df_combined.time_elapsed_indays<0]
df_combined=df_combined.drop(labels=temp_time_elapsed_drop_rows_index,axis=0)
df_combined.shape
time_elapsed_value_counts=df_combined.time_elapsed_indays.value_counts()
time_elapsed_value_counts.head()
#TEMP
time_elapsed_value_counts.shape[0]
#PLOTTING USING plotly takes way too much memory in the front end and makes it laggy
#due to the huge number of datapoints

#data=[go.Histogram(x=df_combined.time_elapsed_indays)]
#layout=go.Layout(title='Histogram of Elapsed Days',xaxis={'title':'Number of elapsed days'},yaxis={'title':'# Count'})
#fig_histogram_elapsed_days=go.Figure(data=data,layout=layout)
#py.iplot(fig_histogram_elapsed_days)
#Histogram of elapsed days distribution
plt.hist(df_combined.time_elapsed_indays,bins=1000)
plt.axis([0,500,0,50000])
plt.xlabel('Elapsed Days')
plt.ylabel('# Count')
plt.title('Histogram of elapsed days(Bins=1000)')
plt.grid(True)
plt.show()
#Lets now take a closer look (x axis constrained from 0 to 200)
#fig=plt.figure()
plt.axis([0,200,0,15000])
plt.bar(time_elapsed_value_counts.index,time_elapsed_value_counts)
plt.xlabel('Elapsed Days')
plt.ylabel('# Count')
plt.title('Histogram of elapsed days(Bins=1000 and x-axis=[0,200])')
plt.grid(True)
#print(type(fig))
plt.show()
#Percentage of cases closed within the spike period seen in the graph (25 to 50 days mark)
sum(time_elapsed_value_counts[((time_elapsed_value_counts.index>24) & (time_elapsed_value_counts.index<51))])/df_combined.shape[0]
agg_time_elapsed_resolved_count=df_combined.groupby(['time_elapsed_indays','Resolved']).size()
agg_time_elapsed_resolved_count.head()
#Dealing with nulls after stacking and unstacking operations are important
agg_time_elapsed_resolved_count_unstacked_level_1_df=agg_time_elapsed_resolved_count.unstack(level=1)
print(view_df_stats(agg_time_elapsed_resolved_count_unstacked_level_1_df))
agg_time_elapsed_resolved_count_unstacked_level_1_df=agg_time_elapsed_resolved_count_unstacked_level_1_df.fillna(0)
agg_time_elapsed_resolved_count_unstacked_level_1_df.head()
#TEMP
agg_time_elapsed_resolved_count_unstacked_level_1_df.max()
print(view_df_stats(agg_time_elapsed_resolved_count_unstacked_level_1_df))
#Box and whiskers plot
trace0=go.Box(y=agg_time_elapsed_resolved_count_unstacked_level_1_df.N,name='Not Solved')
trace1=go.Box(y=agg_time_elapsed_resolved_count_unstacked_level_1_df.S,name='Solved')
data=[trace0,trace1]
layout=go.Layout(title='Box and whiskers plot for Time Elapsed comparing solved vs unsolved')
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
#Note: The reason we are using bars(matplotlib) instead of stacked histograms is because it was a problem setting the bins count
print('Green=solved, red=not solved')
plt.axis((0,500,0,9000))
plt.bar(agg_time_elapsed_resolved_count_unstacked_level_1_df.index,agg_time_elapsed_resolved_count_unstacked_level_1_df.S,color='g');
plt.bar(agg_time_elapsed_resolved_count_unstacked_level_1_df.index,agg_time_elapsed_resolved_count_unstacked_level_1_df.N,color='r');
plt.xlabel('Elapsed Days')
plt.ylabel('# Count')
plt.grid(True)
plt.show()
#LETS NOW TAKE A CLOSER LOOK AT THE PERIOD WHERE MAXIMUM ACTIVITY IS SEEN
#Note: The reason we are using bars(matplotlib) instead of stacked histograms is because it was a problem setting the bins count
print('Green=solved, red=not solved')
plt.axis((0,100,0,9000))
plt.bar(agg_time_elapsed_resolved_count_unstacked_level_1_df.index,agg_time_elapsed_resolved_count_unstacked_level_1_df.S,color='g');
plt.bar(agg_time_elapsed_resolved_count_unstacked_level_1_df.index,agg_time_elapsed_resolved_count_unstacked_level_1_df.N,color='r');
plt.xlabel('Elapsed Days')
plt.ylabel('# Count')
plt.grid(True)
plt.show()
#Percentage of solved cases between 25 and 50 days
print(agg_time_elapsed_resolved_count_unstacked_level_1_df.S[(agg_time_elapsed_resolved_count_unstacked_level_1_df.S.index>24) & (agg_time_elapsed_resolved_count_unstacked_level_1_df.S.index<51)].sum()/resolved_count_in_df_combined)
print(agg_time_elapsed_resolved_count_unstacked_level_1_df.N[(agg_time_elapsed_resolved_count_unstacked_level_1_df.N.index>24) & (agg_time_elapsed_resolved_count_unstacked_level_1_df.N.index<51)].sum()/non_resolved_count_in_df_combined)
def groupby_with_parameter_and_resolved(df,primary_col):
    grouped_df=df.loc[:,[primary_col,'Resolved']].groupby([primary_col,'Resolved']).size()
    return pd.DataFrame(grouped_df).rename(columns={0:'counts'})
def describe_grouped_df_with_parameter_and_resolved_plotly(df,agg_label,df_agg_col_label_name='counts'):
    not_solved=go.Box(y=df.loc[df.index.get_level_values(1)=='N',df_agg_col_label_name],name='not_solved')
    solved=go.Box(y=df.loc[df.index.get_level_values(1)=='S',df_agg_col_label_name],name='solved')
    data=[not_solved,solved]
    layout=go.Layout(title='Box and whiskers plot grouped by '+agg_label+' ('+df_agg_col_label_name+')')
    fig=go.Figure(data=data,layout=layout)
    py.iplot(fig)
region_name_grouped_df=groupby_with_parameter_and_resolved(df_combined,'region_name')
region_name_grouped_df
view_df_stats(region_name_grouped_df)
describe_grouped_df_with_parameter_and_resolved_plotly(region_name_grouped_df,'region_name')
def calc_perc(df,grouped_df,primary_index_col,agg_col='counts'):
    sum_of_cases_per_region=pd.DataFrame(df.groupby([primary_index_col]).size().rename("counts"))
    print(sum_of_cases_per_region)
    return pd.DataFrame(grouped_df.div(sum_of_cases_per_region,level=primary_index_col)['counts']).rename(columns={'counts':'perc'})
region_name_perc_grouped_df=calc_perc(df_combined,region_name_grouped_df,'region_name')
#print(type(region_name_perc_grouped_df))
region_name_perc_grouped_df
describe_grouped_df_with_parameter_and_resolved_plotly(region_name_perc_grouped_df,'region_name',df_agg_col_label_name='perc')
def plot_column_bar_with_respect_to_resolved(grouped_df,primary_index_col,xlabel,agg_col='year',ylabel='# of cases'):
    grouped_df=pd.DataFrame(grouped_df) #Incase its a series
    N=grouped_df.index.get_level_values(primary_index_col).unique().shape[0]
    ind=np.arange(N)
    barS=grouped_df.loc[grouped_df.index.get_level_values('Resolved')=='S',agg_col]
    barN=grouped_df.loc[grouped_df.index.get_level_values('Resolved')=='N',agg_col]
    plt_bar_1=plt.bar(ind,barS)
    plt_bar_2=plt.bar(ind,barN,bottom=barS)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(ind,grouped_df.index.get_level_values(primary_index_col).unique())
    plt.legend((plt_bar_1[0],plt_bar_2[0]),('Solved','Not solved'))
def plot_column_bar_with_respect_to_resolved_plotly(grouped_df,primary_index_col,xlabel,agg_col='counts',ylabel='# of cases',bartype='stack',
                                                    reference_horizontal_lines_for_Resolved=False,
                                                    reference_lines_coordinates_for_solved_unsolved=(None,None),plot_title=None):
    grouped_df=pd.DataFrame(grouped_df) #Incase its a series
    barS=grouped_df.loc[grouped_df.index.get_level_values('Resolved')=='S',agg_col]
    barN=grouped_df.loc[grouped_df.index.get_level_values('Resolved')=='N',agg_col]
    assert bartype=='group' or not reference_horizontal_lines_for_Resolved,"Can only draw reference lines when bartype='Group'"
    x_axis_count=grouped_df.index.get_level_values(primary_index_col).unique().shape[0]
    plt_bar_1=go.Bar(x=grouped_df.index.get_level_values(primary_index_col).unique(),y=barS,name='Solved')
    plt_bar_2=go.Bar(x=grouped_df.index.get_level_values(primary_index_col).unique(),y=barN,name='Not Solved')
    layout_kwargs=dict()
    if reference_horizontal_lines_for_Resolved:
        layout_kwargs['shapes']=[]
        if reference_lines_coordinates_for_solved_unsolved[0]:
            layout_kwargs['shapes'].append({
                'type':'line',
                'xref':'paper',
                'x0': 0,
                'x1':1,
                'y0':reference_lines_coordinates_for_solved_unsolved[0],
                'y1':reference_lines_coordinates_for_solved_unsolved[0],
                'line':{
                    'color':'rgb(31, 119, 255)'
                }
            })
        if reference_lines_coordinates_for_solved_unsolved[1]:
            layout_kwargs['shapes'].append({
                'type':'line',
                'xref':'paper',
                'x0': 0,
                'x1':1,
                'y0':reference_lines_coordinates_for_solved_unsolved[1],
                'y1':reference_lines_coordinates_for_solved_unsolved[1],
                'line':{
                    'color':'rgb(255, 127, 120)'
                }
            })
    if plot_title:
        #print(plot_title)
        layout_kwargs['title']=plot_title
    layout=go.Layout(xaxis={'title':xlabel},yaxis={'title':ylabel},barmode=bartype,**layout_kwargs)
    fig_col_bar=go.Figure(data=[plt_bar_1,plt_bar_2],layout=layout)
    py.iplot(fig_col_bar,filename='plot_column_bar')
plot_column_bar_with_respect_to_resolved_plotly(region_name_grouped_df,'region_name','region_name',plot_title='Distribution of cases region wise')
#print(type(region_name_grouped_df))
#print(type(region_name_perc_grouped_df))
#region_name_perc_grouped_df
plot_column_bar_with_respect_to_resolved_plotly(region_name_perc_grouped_df,'region_name','region_name','perc',ylabel='% of cases',bartype='group',reference_horizontal_lines_for_Resolved=True,reference_lines_coordinates_for_solved_unsolved=(resolved_perc_in_df_combined,non_resolved_perc_in_df_combined),plot_title='Distribution of cases region wise with reference lines denoting the overall percentage')
states_grouped_df=groupby_with_parameter_and_resolved(df_combined,'state')
states_grouped_df
view_df_stats(states_grouped_df)
states_perc_grouped_df=calc_perc(df_combined,states_grouped_df,'state')
states_perc_grouped_df
view_df_stats(states_perc_grouped_df)
describe_grouped_df_with_parameter_and_resolved_plotly(states_grouped_df,'states')
describe_grouped_df_with_parameter_and_resolved_plotly(states_perc_grouped_df,'states','perc')
plot_column_bar_with_respect_to_resolved_plotly(states_grouped_df,'state','States')
plot_column_bar_with_respect_to_resolved_plotly(states_perc_grouped_df,'state','state','perc',ylabel='% of cases',bartype='group',reference_horizontal_lines_for_Resolved=True,reference_lines_coordinates_for_solved_unsolved=(resolved_perc_in_df_combined,non_resolved_perc_in_df_combined),plot_title='Distribution of cases state wise with reference lines denoting the overall percentage')
df_combined.type=df_combined.type.apply(lambda x:'Business' if x==1 else 'Person')
df_combined.type.value_counts()
type_groupby_df=groupby_with_parameter_and_resolved(df_combined
                                                   ,'type')
type_groupby_df
plot_column_bar_with_respect_to_resolved_plotly(type_groupby_df,'type','Complainant')
type_groupby_perc_df=calc_perc(df_combined,type_groupby_df,'type')
type_groupby_perc_df
plot_column_bar_with_respect_to_resolved_plotly(type_groupby_perc_df,'type','type',agg_col='perc',bartype='group',reference_horizontal_lines_for_Resolved=True,reference_lines_coordinates_for_solved_unsolved=(resolved_perc_in_df_combined,non_resolved_perc_in_df_combined),plot_title='Distribution of cases complaint type wise with reference lines denoting the overall percentage')
df_combined.business_activity_description.describe()
business_activity_description_value_counts=df_combined.business_activity_description.value_counts()
business_activity_description_value_counts.describe()
data=[go.Box(y=business_activity_description_value_counts,name='Business Activity Description Counts')]
layout=go.Layout(title='Box and whisker plot of Business Activity Description Value Counts')
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
business_activity_description_value_counts_df=pd.DataFrame(business_activity_description_value_counts).rename(columns={'business_activity_description':'counts'})
business_activity_description_value_counts_df.head()
def calc_cumulative_frequency(df_or_ser,col):
    if col:
        return df_or_ser[col].cumsum(skipna=False)
    else:
        return df_or_ser.cumsum(skipna=False)
def plot_cumsum_plotly(df,cumsum_of,plot_title,cumsum_col='cumsum',hover_col=None):
    trace0=go.Scatter(
        x=np.array(range(df.shape[0]))/df.shape[0],
        y=df[cumsum_col],
        name=cumsum_of,
        mode='lines',
        hoverinfo='x+y+text',
        hovertext=df[hover_col] if hover_col else ""
    )
    layout=go.Layout(
    title=plot_title
    )
    fig=go.Figure(data=[trace0],layout=layout)
    py.iplot(fig)
business_activity_description_value_counts_df['cumsum']=calc_cumulative_frequency(business_activity_description_value_counts_df,'counts')/df_combined.shape[0]
business_activity_description_value_counts_df.head()
print(sum(business_activity_description_value_counts>business_activity_description_value_counts.describe()['75%']))
print(sum(business_activity_description_value_counts>business_activity_description_value_counts.describe()['75%'])/business_activity_description_value_counts.describe()['count'])
sum(business_activity_description_value_counts[business_activity_description_value_counts>business_activity_description_value_counts.describe()['75%']])/df_combined.business_activity_description.describe()['count']
business_activity_description_groupby_df=groupby_with_parameter_and_resolved(df_combined,'business_activity_description')
#print(business_activity_description_groupby_df.columns)
business_activity_description_groupby_df.tail()
business_activity_description_groupby_df=business_activity_description_groupby_df.unstack(level=1).fillna(0)
print('Original:')
print(business_activity_description_groupby_df.columns)
#If u see the output for the above print statement, you can see that column is a multiindex whose level 0 we don't need...so we are removing level 0 here
business_activity_description_groupby_df.columns=business_activity_description_groupby_df.columns.droplevel(0)
#print(type(business_activity_description_groupby_df))
print('After droplevel:')
print(business_activity_description_groupby_df.columns)
print('Nulls={}'.format(business_activity_description_groupby_df.isnull().sum()))
business_activity_description_groupby_df.head()
business_activity_description_groupby_df['sum']=business_activity_description_groupby_df.N+business_activity_description_groupby_df.S
business_activity_description_groupby_df=business_activity_description_groupby_df.sort_values('sum',axis=0,ascending=False)
business_activity_description_groupby_df_restacked=pd.DataFrame(business_activity_description_groupby_df.drop('sum',1).stack()).rename(columns={0:'counts'})
#print(business_activity_description_groupby_df.columns)
business_activity_description_groupby_df_restacked.head()
#plot_column_bar_with_respect_to_resolved_plotly(business_activity_description_groupby_df,'business_activity_description','Business activity Description')
#Note: The reason we are using bars(matplotlib) instead of stacked histograms is because it was a problem setting the bins count
print('Green=solved, red=not solved')
plt.axis((0,500,0,25000))
plt.bar(range(business_activity_description_groupby_df.shape[0]),business_activity_description_groupby_df.S,color='g');
plt.bar(range(business_activity_description_groupby_df.shape[0]),business_activity_description_groupby_df.N,color='r');
plt.xlabel('Elapsed Days')
plt.ylabel('# Count')
plt.grid(True)
plt.show()
business_activity_description_groupby_df['perc_S']=business_activity_description_groupby_df['S']/business_activity_description_groupby_df['sum']
business_activity_description_groupby_df['perc_N']=business_activity_description_groupby_df['N']/business_activity_description_groupby_df['sum']
business_activity_description_groupby_df['diff_perc_S_and_N']=business_activity_description_groupby_df.perc_S-business_activity_description_groupby_df.perc_N
business_activity_description_groupby_df.head()
def plot_line_graph_wrt_resolved_plotly(df,S_col,N_col,diff_col,scatter_mode='lines',title='Line graph'):
    data=[]
    if S_col:
        trace0=go.Scatter(x=list(range(df.shape[0])),y=df[S_col],mode=scatter_mode,name='Solved',hoverinfo='text+y',hovertext=df.index.get_level_values(0))
        data.append(trace0)
    if N_col:
        trace1=go.Scatter(x=list(range(df.shape[0])),y=df[N_col],mode=scatter_mode,name='Not Solved',hoverinfo='text+y',hovertext=df.index.get_level_values(0))
        data.append(trace1)
    if diff_col:
        trace2=go.Scatter(x=list(range(df.shape[0])),y=df[diff_col],mode=scatter_mode,name='Diff',hoverinfo='text+y',hovertext=df.index.get_level_values(0))
        data.append(trace2)
    layout=go.Layout(title=title)
    py.iplot(data,layout)
plot_line_graph_wrt_resolved_plotly(business_activity_description_groupby_df,None,None,'diff_perc_S_and_N',scatter_mode='markers',title='Line graph showing complaints by business description')
print(df_combined.complaint_subject_desc.isnull().sum())
df_combined.complaint_subject_desc.describe()
complaint_subject_desc_value_counts=pd.DataFrame(df_combined.complaint_subject_desc.value_counts()).rename(columns={'complaint_subject_desc':'counts'})
complaint_subject_desc_value_counts.head(10)
complaint_subject_desc_value_counts.describe()
#TODO: Doing the cumulative frequency for just this field...later, need to do it for the other fields too

complaint_subject_desc_value_counts['cumsum']=calc_cumulative_frequency(complaint_subject_desc_value_counts,col='counts')/df_combined.shape[0]
print(complaint_subject_desc_value_counts.head())
complaint_subject_desc_value_counts.tail()
complaint_subject_desc_value_counts.isnull().sum()
plot_cumsum_plotly(complaint_subject_desc_value_counts,'complaint_subject_desc','cumulative sum of complaint subject desc',hover_col='counts')

df_combined.complaint_subject_code.value_counts()/df_combined.shape[0]

