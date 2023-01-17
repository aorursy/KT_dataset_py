!pip install plotly
!pip install hide-code

!jupyter nbextension install --py hide_code

!jupyter nbextension enable --py hide_code

!jupyter serverextension enable --py hide_code
!pip install ipyaggrid

!jupyter nbextension enable --py --sys-prefix ipyaggrid

#!pip install chart_studio
!pip install ipywidgets

!jupyter nbextension enable --py --sys-prefix widgetsnbextension
#General imports

import pandas as pd

from IPython.core.display import display, HTML

# ipyaggrid

from ipyaggrid import Grid
# visualizations

import plotly

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False) #do not miss this line

import plotly.figure_factory as ff

from plotly.offline import iplot

import plotly.graph_objects as go

import plotly.express as px

#import chart_studio.plotly as py

from plotly.subplots import make_subplots
# imports ipywidgets

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from ipywidgets import Layout

from ipywidgets import TwoByTwoLayout
# german.data

data = pd.read_csv('../input/german_data.csv',header=0)
data.head(2)
data.shape
# change name of columns

data.columns = ['check_acct','duration','credit_hist','purpose','credit_amt','saving_acct','present_empl',

                'installment_rate', 'sex','other_debtor','present_resid','property','age','other_install',

                'housing','n_credits','job','n_people','tlf','foreign','target']
# replace values in the target variable

data['target'] = data['target'].replace({1:0, 2:1}) # 0: non_default (good credit), 1:default (bad credit)
data.index.name='index' # need to name ALL cols otherwise ipyaggrid will neglect them

columns_defs = [{'field':data.index.name}] + [{'field':c} for c in data.columns]



grid_options = {'columnDefs': columns_defs,

               'enableSorting': True,

               'enableFilter' :True,

               'enableColResize': True,

               'enableRangeSelection':True,'enableValue': True,

               'statusBar': {

        'statusPanels': [

            { 'statusPanel': 'agTotalAndFilteredRowCountComponent', 'align': 'left' },

            { 'statusPanel': 'agTotalRowCountComponent', 'align': 'center' },

            { 'statusPanel': 'agFilteredRowCountComponent' },

            { 'statusPanel': 'agSelectedRowCountComponent' },

            { 'statusPanel': 'agAggregationComponent' }

        ]

    }

               }



g = Grid(grid_data = data,

        theme = 'ag-theme-fresh',

        quick_filter = False,

        show_toggle_delete = True,

        show_toggle_edit = True,

        grid_options = grid_options,

        index = True,

        width=1500,

        height=500,

        center = False, 

        )



g
stats_table = widgets.Output()

duration_out = widgets.Output()

credit_amt_out = widgets.Output()

installment_rate_out = widgets.Output()

present_resid_out = widgets.Output()

age_out = widgets.Output()

n_credits_out = widgets.Output()

n_people_out = widgets.Output()



tab = widgets.Tab(children = [stats_table,duration_out, credit_amt_out,installment_rate_out, present_resid_out,age_out,n_credits_out,n_people_out])

tab.set_title(0, 'Stats table')

tab.set_title(1, 'duration')

tab.set_title(2, 'credit_amt')

tab.set_title(3, 'installment_rate')

tab.set_title(4, 'present_resid')

tab.set_title(5, 'age')

tab.set_title(6, 'n_credits')

tab.set_title(7, 'n_people')

display(tab)



with stats_table:

    stats = data.describe().round(2)

    display(stats)



with duration_out:

    data_grouped_duration = data.groupby('duration')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped_duration = pd.DataFrame(data_grouped_duration)

    data_grouped_duration.columns = ['duration','target','count']

    data_grouped_duration_0 = data_grouped_duration[data_grouped_duration['target']==0]

    data_grouped_duration_1 = data_grouped_duration[data_grouped_duration['target']==1]

    

    # create data for box plots

    data_duration_0=data[data['target']==0]

    data_duration_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped_duration['percent_bad']=data_grouped_duration.groupby('duration')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped_duration_bad = data_grouped_duration[data_grouped_duration['target']==1]

    data_grouped_duration_good = data_grouped_duration[data_grouped_duration['target']==0]

    

    #fig3 = go.Figure()

    fig1 = make_subplots(rows=1, cols=3,subplot_titles=('Loan duration: non-default vs default credit ', 'Loan duration (months)', 'Loan duration (months)'),horizontal_spacing = 0.1,column_widths=[20, 10,10])

    # probabilities plot

    fig1.add_trace(go.Bar(x=data_grouped_duration_good['duration'].astype(str), y=data_grouped_duration_good['percent_bad'], name = 'non_default',marker_color='blue'))

    fig1.add_trace(go.Bar(x=data_grouped_duration_bad['duration'].astype(str), y=data_grouped_duration_bad['percent_bad'], name = 'default',marker_color='red'))

  

    # box plots

    fig1.add_trace(go.Box(y=data_duration_0['duration'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig1.add_trace(go.Box(y=data_duration_1['duration'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig1.add_trace(go.Histogram(x=data['duration'],histnorm='percent',name='duration'),row=1,col=3)   

    

    fig1.update_layout( barmode='stack',xaxis=dict(title=('Loan duration (months)')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig1.update_xaxes(title_text='Loan duration (months)',row=1,col=3)

    fig1.update_xaxes(title_text='Loan duration (months)',row=1,col=2)

    fig1.update_yaxes(title_text='Count (%)',row=1,col=3)

    #fig1.show()

    iplot(fig1)

    

    



with credit_amt_out:

    data_grouped_credit_amt = data.groupby('credit_amt')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped_credit_amt = pd.DataFrame(data_grouped_credit_amt)

    data_grouped_credit_amt.columns = ['credit_amt','target','count']

    data_grouped_credit_amt_0 = data_grouped_credit_amt[data_grouped_credit_amt['target']==0]

    data_grouped_crdit_amt_1 = data_grouped_credit_amt[data_grouped_credit_amt['target']==1]

    

    # create data for box plots

    data_credit_amt_0=data[data['target']==0]

    data_credit_amt_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped_credit_amt['percent_bad']=data_grouped_credit_amt.groupby('credit_amt')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped_credit_amt_bad = data_grouped_credit_amt[data_grouped_credit_amt['target']==1]

    data_grouped_credit_amt_good = data_grouped_credit_amt[data_grouped_credit_amt['target']==0]

    

    #fig3 = go.Figure()

    fig2 = make_subplots(rows=1, cols=3,subplot_titles=('Credit amount: non-default vs default credit ', 'Credit amount (DM)', 'Credit amount (DM)'),horizontal_spacing = 0.1,column_widths=[2, 1.0,1.0])

    # probabilities plot

    fig2.add_trace(go.Histogram(x=data_grouped_credit_amt_good['credit_amt'],histnorm='percent',name='non_default',marker_color='blue',xbins=dict(start=250, end=18424,size=500)),row=1,col=1)

    fig2.add_trace(go.Histogram(x=data_grouped_credit_amt_bad['credit_amt'], histnorm='percent',name='default',marker_color='red',xbins=dict(start=2500, end=18424,size=5),opacity=0.4),row=1,col=1)

    

    # box plots

    fig2.add_trace(go.Box(y=data_credit_amt_0['credit_amt'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig2.add_trace(go.Box(y=data_credit_amt_1['credit_amt'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig2.add_trace(go.Histogram(x=data['credit_amt'],histnorm='percent',name='credit_amt'),row=1,col=3)   

    

    fig2.update_layout(barmode='stack',xaxis=dict(title=('Credit amount (DM)')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig2.update_xaxes(title_text='Credit amount (DM)',row=1,col=3)

    fig2.update_xaxes(title_text='Credit amount (DM)',row=1,col=2)

    fig2.update_yaxes(title_text='Count (%)',row=1,col=3)

    #fig2.show()

    iplot(fig2)

    

with installment_rate_out:

    # create data for barplots

    data_grouped = data.groupby('installment_rate')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped = pd.DataFrame(data_grouped)

    data_grouped.columns = ['installment_rate','target','count']

    data_grouped_0 = data_grouped[data_grouped['target']==0]

    data_grouped_1 = data_grouped[data_grouped['target']==1]

    

    # create data for box plots

    data_0=data[data['target']==0]

    data_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped['percent_bad']=data_grouped.groupby('installment_rate')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped_bad = data_grouped[data_grouped['target']==1]

    data_grouped_good = data_grouped[data_grouped['target']==0]

    

    #fig3 = go.Figure()

    fig3 = make_subplots(rows=1, cols=3,subplot_titles=('Installment rate non-default vs default credit ', 'Installment rate', 'Installment rate'),horizontal_spacing = 0.1,column_widths=[1.0, 0.9,0.9])

    # probabilities plot

    fig3.add_trace(go.Bar(x=data_grouped_good['installment_rate'].astype('str'), y=data_grouped_good['percent_bad'], name = 'non_default',marker_color='blue',text = data_grouped_good['percent_bad'], textposition='auto'))

    fig3.add_trace(go.Bar(x=data_grouped_bad['installment_rate'].astype('str'), y=data_grouped_bad['percent_bad'], name = 'default',marker_color='red',text = data_grouped_bad['percent_bad'], textposition='auto'))

    

    # box plots

    fig3.add_trace(go.Box(y=data_0['installment_rate'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig3.add_trace(go.Box(y=data_1['installment_rate'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig3.add_trace(go.Histogram(x=data['installment_rate'],histnorm='percent',name='installment_rate'),row=1,col=3)

    

    fig3.update_layout(barmode='group', xaxis=dict(title=('installment_rate (in percentage of disposable income)')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig3.update_xaxes(title_text='installment_rate (in percentage of disposable income)',row=1,col=3)

    fig3.update_xaxes(title_text='installment_rate (in percentage of disposable income)',row=1,col=2)

    fig3.update_yaxes(title_text='Count (%)',row=1,col=3)

    #fig3.show()

    iplot(fig3)

    

with present_resid_out:

   # create data for barplots

    data_grouped2 = data.groupby('present_resid')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped2 = pd.DataFrame(data_grouped2)

    data_grouped2.columns = ['present_resid','target','count']

    data_grouped2_0 = data_grouped2[data_grouped2['target']==0]

    data_grouped2_1 = data_grouped2[data_grouped2['target']==1]

    

    # create data for box plots

    data_0=data[data['target']==0]

    data_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped2['percent_bad']=data_grouped2.groupby('present_resid')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped2_bad = data_grouped2[data_grouped2['target']==1]

    data_grouped2_good = data_grouped2[data_grouped2['target']==0]

    

    #fig3 = go.Figure()

    fig4 = make_subplots(rows=1, cols=3,subplot_titles=('Present_residence: non-default vs default credit ', 'Present residence (years)', 'Present residence (years)'),horizontal_spacing = 0.1,column_widths=[1.0, 0.9,0.9])

    # probabilities plot

    fig4.add_trace(go.Bar(x=data_grouped2_good['present_resid'], y=data_grouped2_good['percent_bad'], name = 'non_default',marker_color='blue',text = data_grouped2_good['percent_bad'], textposition='auto'))

    fig4.add_trace(go.Bar(x=data_grouped2_bad['present_resid'], y=data_grouped2_bad['percent_bad'], name = 'default',marker_color='red',text = data_grouped2_bad['percent_bad'], textposition='auto'))

    

    # box plots

    fig4.add_trace(go.Box(y=data_0['present_resid'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig4.add_trace(go.Box(y=data_1['present_resid'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig4.add_trace(go.Histogram(x=data['present_resid'],histnorm='percent',name='present_resid'),row=1,col=3)

    

    fig4.update_layout(barmode='group', xaxis=dict(title=('Present residence (years)')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig4.update_xaxes(title_text='Present residence (years)',row=1,col=3)

    fig4.update_xaxes(title_text='Present residence (years)',row=1,col=2)

    fig4.update_yaxes(title_text='Count (%)',row=1,col=3)

    !fig4.show()

    iplot(fig4)



with age_out:

    data_grouped_age= data.groupby('age')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped_age = pd.DataFrame(data_grouped_age)

    data_grouped_age.columns = ['age','target','count']

    data_grouped_age_0 = data_grouped_age[data_grouped_age['target']==0]

    data_grouped_age_1 = data_grouped_age[data_grouped_age['target']==1]

    

    # create data for box plots

    data_age_0=data[data['target']==0]

    data_age_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped_age['percent_bad']=data_grouped_age.groupby('age')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped_age_bad = data_grouped_age[data_grouped_age['target']==1]

    data_grouped_age_good = data_grouped_age[data_grouped_age['target']==0]

    

    #fig3 = go.Figure()

    fig5 = make_subplots(rows=1, cols=3,subplot_titles=('Age: non-default vs default credit ', 'Age (years)', 'Age (years)'),horizontal_spacing = 0.1,column_widths=[2, 1.0,0.9])

    # probabilities plot

    fig5.add_trace(go.Bar(x=data_grouped_age_good['age'], y=data_grouped_age_good['percent_bad'], name = 'non_default',marker_color='blue'))

    fig5.add_trace(go.Bar(x=data_grouped_age_bad['age'], y=data_grouped_age_bad['percent_bad'], name = 'default',marker_color='red'))

    

    

    # box plots

    fig5.add_trace(go.Box(y=data_age_0['age'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig5.add_trace(go.Box(y=data_age_1['age'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig5.add_trace(go.Histogram(x=data['age'],histnorm='percent',name='credit_amt'),row=1,col=3)   

    

    fig5.update_layout(barmode='stack', xaxis=dict(title=('Age')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig5.update_xaxes(title_text='Age',row=1,col=3)

    fig5.update_xaxes(title_text='Age',row=1,col=2)

    fig5.update_yaxes(title_text='Count (%)',row=1,col=3)

    #fig5.show()

    iplot(fig5)

    

with n_credits_out:

    data_grouped3 = data.groupby('n_credits')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped3 = pd.DataFrame(data_grouped3)

    data_grouped3.columns = ['n_credits','target','count']

    data_grouped3_0 = data_grouped3[data_grouped3['target']==0]

    data_grouped3_1 = data_grouped3[data_grouped3['target']==1]

    

    # create data for box plots

    data_0=data[data['target']==0]

    data_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped3['percent_bad']=data_grouped3.groupby('n_credits')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped3_bad = data_grouped3[data_grouped3['target']==1]

    data_grouped3_good = data_grouped3[data_grouped3['target']==0]

    

    #fig3 = go.Figure()

    fig6 = make_subplots(rows=1, cols=3,subplot_titles=('Number of credits: non-default vs default credit ', 'Number of credits', 'Number of credits'),horizontal_spacing = 0.1,column_widths=[1.0, 0.9,0.9])

    # probabilities plot

    fig6.add_trace(go.Bar(x=data_grouped3_good['n_credits'], y=data_grouped3_good['percent_bad'], name = 'non_default',marker_color='blue',text = data_grouped3_good['percent_bad'], textposition='auto'))

    fig6.add_trace(go.Bar(x=data_grouped3_bad['n_credits'], y=data_grouped3_bad['percent_bad'], name = 'default',marker_color='red',text = data_grouped3_bad['percent_bad'], textposition='auto'))

    

    # box plots

    fig6.add_trace(go.Box(y=data_0['n_credits'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig6.add_trace(go.Box(y=data_1['n_credits'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig6.add_trace(go.Histogram(x=data['n_credits'],histnorm='percent',name='n_credits'),row=1,col=3)

    

    fig6.update_layout(barmode='group', xaxis=dict(title=('Number of credits')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig6.update_xaxes(title_text='Number of credits',row=1,col=3)

    fig6.update_xaxes(title_text='Number of cedits',row=1,col=2)

    fig6.update_yaxes(title_text='Count (%)',row=1,col=3)

    #fig6.show()

    iplot(fig6)

    

with n_people_out:

    data_grouped4 = data.groupby('n_people')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped4 = pd.DataFrame(data_grouped4)

    data_grouped4.columns = ['n_people','target','count']

    data_grouped4_0 = data_grouped4[data_grouped4['target']==0]

    data_grouped4_1 = data_grouped4[data_grouped4['target']==1]

    

    # create data for box plots

    data_0=data[data['target']==0]

    data_1=data[data['target']==1]

    

    #create data for probabilities default vs non-default

    data_grouped4['percent_bad']=data_grouped4.groupby('n_people')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped4_bad = data_grouped4[data_grouped4['target']==1]

    data_grouped4_good = data_grouped4[data_grouped4['target']==0]

    

    #fig3 = go.Figure()

    fig7 = make_subplots(rows=1, cols=3,subplot_titles=('Number of people being liable to provide maintenance for ', 'Number of people', 'Number of people'),horizontal_spacing = 0.1,column_widths=[1.0, 1.0 ,1.0])

    # probabilities plot

    fig7.add_trace(go.Bar(x=data_grouped4_good['n_people'], y=data_grouped4_good['percent_bad'], name = 'non_default',marker_color='blue',text = data_grouped4_good['percent_bad'], textposition='auto'))

    fig7.add_trace(go.Bar(x=data_grouped4_bad['n_people'], y=data_grouped4_bad['percent_bad'], name = 'default',marker_color='red',text = data_grouped4_bad['percent_bad'], textposition='auto'))

    

    # box plots

    fig7.add_trace(go.Box(y=data_0['n_people'],marker_color='blue',name='non-default',boxmean=True,showlegend=False),row=1,col=2)

    fig7.add_trace(go.Box(y=data_1['n_people'],marker_color='red',name='default',boxmean=True,showlegend=False),row=1,col=2)

    

    # hist

    fig7.add_trace(go.Histogram(x=data['n_people'],histnorm='percent',name='n_people'),row=1,col=3)

    

    fig7.update_layout(barmode='group', xaxis=dict(title=('Number of people')), yaxis=dict(title='Probability'),template='plotly_white',height=650, width=1400)

    fig7.update_xaxes(title_text='Number of people',row=1,col=3)

    fig7.update_xaxes(title_text='Number of people',row=1,col=2)

    fig7.update_yaxes(title_text='Count (%)',row=1,col=3)

    #fig7.show()

    iplot(fig7)

    

    
out1 = widgets.Output()

out2 = widgets.Output()



tab = widgets.Tab(children=[out2,out1])

tab.set_title(0,'Correlation matrix')

tab.set_title(1,'Scattermatrix plot')

display(tab)



with out2:

    corr = data.corr().round(2)

    display(corr)

    



with out1:

    fig=go.Figure(data=go.Splom(dimensions=[dict(label='duration',values=data['duration']),

                                       dict(label='credit_amt',values=data['credit_amt']),

                                        dict(label='installment_rate',values=data['installment_rate']),

                                        dict(label='present_resid',values=data['present_resid']),

                                        dict(label='age',values=data['age']),

                                        dict(label='n_credits',values=data['n_credits']),

                                        dict(label='n_people',values=data['n_people']),

                                        dict(label='target',values=data['target'])], text=data['target'],marker=dict(color=data['target'],colorscale='Bluered')))

    fig.update_layout(width=1200,height=900,template='plotly_white')

    fig.show()
check_acc_out = widgets.Output()

credit_hist_out = widgets.Output()

purpose_out = widgets.Output()

saving_acct_out = widgets.Output()

present_empl_out = widgets.Output()

sex_out = widgets.Output()

other_debtor_out = widgets.Output()

property_out = widgets.Output()

other_install_out = widgets.Output()

housing_out = widgets.Output()

job_out = widgets.Output()

tlf_out = widgets.Output()

foreign_out = widgets.Output()



tab = widgets.Tab(children = [check_acc_out,credit_hist_out,purpose_out,saving_acct_out,present_empl_out,sex_out,

                             other_debtor_out,property_out,other_install_out,housing_out,job_out,tlf_out,foreign_out])

tab.set_title(0, 'check_acc')

tab.set_title(1, 'credit_hist')

tab.set_title(2, 'purpose')

tab.set_title(3, 'saving_acc')

tab.set_title(4, 'present_empl')

tab.set_title(5, 'sex')

tab.set_title(6, 'other_debtor')

tab.set_title(7, 'property')

tab.set_title(8, 'other_install')

tab.set_title(9, 'housing')

tab.set_title(10, 'job')

tab.set_title(11, 'tlf')

tab.set_title(12, 'foreign')

display(tab)



with check_acc_out:

    ct_check_acc = pd.crosstab(data['target'].astype('str'), data['check_acct'],margins=True, margins_name='Total')

    ct_check_acc = 100*ct_check_acc.div(ct_check_acc['Total'],axis=0).round(3)

    display(ct_check_acc)

    

with credit_hist_out:

    ct_credit_hist = pd.crosstab(data['target'].astype('str'), data['credit_hist'],margins=True, margins_name='Total')

    ct_credit_hist = 100*ct_credit_hist.div(ct_credit_hist['Total'],axis=0).round(3)

    display(ct_credit_hist)



with purpose_out:

    ct_purpose = pd.crosstab(data['target'].astype('str'), data['purpose'],margins=True, margins_name='Total')

    ct_purpose = 100*ct_purpose.div(ct_purpose['Total'],axis=0).round(3)

    display(ct_purpose)



with saving_acct_out:

    ct_saving_acc = pd.crosstab(data['target'].astype('str'), data['saving_acct'],margins=True, margins_name='Total')

    ct_saving_acc = 100*ct_saving_acc.div(ct_saving_acc['Total'],axis=0).round(3)

    display(ct_saving_acc)



with present_empl_out:

    ct_present_empl = pd.crosstab(data['target'].astype('str'), data['present_empl'],margins=True, margins_name='Total')

    ct_present_empl = 100*ct_present_empl.div(ct_present_empl['Total'],axis=0).round(3)

    display(ct_present_empl)



with sex_out:

    ct_sex = pd.crosstab(data['target'].astype('str'), data['sex'],margins=True, margins_name='Total')

    ct_sex = 100*ct_sex.div(ct_sex['Total'],axis=0).round(3)

    display(ct_sex)



with other_debtor_out:

    ct_other_debtor = pd.crosstab(data['target'].astype('str'), data['other_debtor'],margins=True, margins_name='Total')

    ct_other_debtor = 100*ct_other_debtor.div(ct_other_debtor['Total'],axis=0).round(3)

    display(ct_other_debtor)

    

with property_out:

    ct_property = pd.crosstab(data['target'].astype('str'), data['property'],margins=True, margins_name='Total')

    ct_property = 100*ct_property.div(ct_property['Total'],axis=0).round(3)

    display(ct_property)



with other_install_out:

    ct_other_install = pd.crosstab(data['target'].astype('str'), data['other_install'],margins=True, margins_name='Total')

    ct_other_install = 100*ct_other_install.div(ct_other_install['Total'],axis=0).round(3)

    display(ct_other_install)



with housing_out:

    ct_housing = pd.crosstab(data['target'].astype('str'), data['housing'],margins=True, margins_name='Total')

    ct_housing = 100*ct_housing.div(ct_housing['Total'],axis=0).round(3)

    display(ct_housing)



with job_out:

    ct_job = pd.crosstab(data['target'].astype('str'), data['job'],margins=True, margins_name='Total')

    ct_job= 100*ct_job.div(ct_job['Total'],axis=0).round(3)

    display(ct_job)



with tlf_out:

    ct_tlf = pd.crosstab(data['target'].astype('str'), data['tlf'],margins=True, margins_name='Total')

    ct_tlf = 100*ct_tlf.div(ct_tlf['Total'],axis=0).round(3)

    display(ct_tlf)



with foreign_out:

    ct_foreign= pd.crosstab(data['target'].astype('str'), data['foreign'],margins=True, margins_name='Total')

    ct_foreign = 100*ct_foreign.div(ct_foreign['Total'],axis=0).round(3)

    display(ct_foreign)

check_acc_out = widgets.Output()

credit_hist_out = widgets.Output()

purpose_out = widgets.Output()

saving_acct_out = widgets.Output()

present_empl_out = widgets.Output()

sex_out = widgets.Output()

other_debtor_out = widgets.Output()

property_out = widgets.Output()

other_install_out = widgets.Output()

housing_out = widgets.Output()

job_out = widgets.Output()

tlf_out = widgets.Output()

foreign_out = widgets.Output()



tab = widgets.Tab(children = [check_acc_out,credit_hist_out,purpose_out,saving_acct_out,present_empl_out,sex_out,

                             other_debtor_out,property_out,other_install_out,housing_out,job_out,tlf_out,foreign_out])

tab.set_title(0, 'check_acc')

tab.set_title(1, 'credit_hist')

tab.set_title(2, 'purpose')

tab.set_title(3, 'saving_acct')

tab.set_title(4, 'present_empl')

tab.set_title(5, 'sex')

tab.set_title(6, 'other_debtor')

tab.set_title(7, 'property')

tab.set_title(8, 'other_install')

tab.set_title(9, 'housing')

tab.set_title(10, 'job')

tab.set_title(11, 'tlf')

tab.set_title(12, 'foreign')

display(tab)



    

with check_acc_out:

    data_grouped = data.groupby('check_acct')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped = pd.DataFrame(data_grouped)

    data_grouped.columns = ['check_acct','target','count']

    data_grouped_0 = data_grouped[data_grouped['target']==0]

    data_grouped_1 = data_grouped[data_grouped['target']==1]

    

    data_grouped['percent_bad']=data_grouped.groupby('check_acct')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped_bad = data_grouped[data_grouped['target']==1]

    data_grouped_good = data_grouped[data_grouped['target']==0]

    

    #fig1 = go.Figure()

    fig1 = make_subplots(rows=1, cols=2,subplot_titles=('Status checking account ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig1.add_trace(go.Bar(x=data_grouped_0['check_acct'], y=data_grouped_0['count'], name = 'non-default',marker_color='blue',text = data_grouped_0['count'], textposition='auto'))

    fig1.add_trace(go.Bar(x=data_grouped_1['check_acct'], y=data_grouped_1['count'], name = 'default',marker_color='red',text = data_grouped_1['count'], textposition='auto'))

    

    fig1.add_trace(go.Bar(x=data_grouped_good['check_acct'], y=data_grouped_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig1.add_trace(go.Bar(x=data_grouped_bad['check_acct'], y=data_grouped_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig1.update_layout(barmode='group', xaxis=dict(title=('check_acct')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig1.update_xaxes(title_text='check_acct', row=1,col=2)

    fig1.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig1.show()

    

with credit_hist_out:

    data_grouped2 = data.groupby('credit_hist')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped2 = pd.DataFrame(data_grouped2)

    data_grouped2.columns = ['credit_hist','target','count']

    data_grouped2_0 = data_grouped2[data_grouped2['target']==0]

    data_grouped2_1 = data_grouped2[data_grouped2['target']==1]

    

    data_grouped2['percent_bad']=data_grouped2.groupby('credit_hist')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped2_bad = data_grouped2[data_grouped2['target']==1]

    data_grouped2_good = data_grouped2[data_grouped2['target']==0]

    

    #fig2 = go.Figure()

    fig2 = make_subplots(rows=1, cols=2,subplot_titles=('Credit history ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    

    

    fig2.add_trace(go.Bar(x=data_grouped2_0['credit_hist'], y=data_grouped2_0['count'], name = 'non_default',marker_color='blue',text = data_grouped2_0['count'], textposition='auto'),

                  row=1,col=1)

    fig2.add_trace(go.Bar(x=data_grouped2_1['credit_hist'], y=data_grouped2_1['count'], name = 'default',marker_color='red',text =data_grouped2_1['count'], textposition ='auto'),

                  row=1,col=1)

    

    fig2.add_trace(go.Bar(x=data_grouped2_good['credit_hist'], y=data_grouped2_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped2_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig2.add_trace(go.Bar(x=data_grouped2_bad['credit_hist'], y=data_grouped2_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped2_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig2.update_layout(barmode='group', xaxis=dict(title=('credit_hist')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig2.update_xaxes(title_text='credit_hist', row=1,col=2)

    fig2.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

   

    fig2.show()

    

with purpose_out:

    data_grouped3 = data.groupby('purpose')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped3 = pd.DataFrame(data_grouped3)

    data_grouped3.columns = ['purpose','target','count']

    data_grouped3_0 = data_grouped3[data_grouped3['target']==0]

    data_grouped3_1 = data_grouped3[data_grouped3['target']==1]

    

    data_grouped3['percent_bad']=data_grouped3.groupby('purpose')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped3_bad = data_grouped3[data_grouped3['target']==1]

    data_grouped3_good = data_grouped3[data_grouped3['target']==0]

    

    #fig3 = go.Figure()

    fig3 = make_subplots(rows=1, cols=2,subplot_titles=('Loan purpose ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig3.add_trace(go.Bar(x=data_grouped3_0['purpose'], y=data_grouped3_0['count'], name = 'non_default',marker_color='blue',text = data_grouped3_0['count'], textposition='auto'))

    fig3.add_trace(go.Bar(x=data_grouped3_1['purpose'], y=data_grouped3_1['count'], name = 'default',marker_color='red',text =data_grouped3_1['count'], textposition ='auto'))

    

    fig3.add_trace(go.Bar(x=data_grouped3_good['purpose'], y=data_grouped3_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped3_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig3.add_trace(go.Bar(x=data_grouped3_bad['purpose'], y=data_grouped3_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped3_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig3.update_layout(barmode='group', xaxis=dict(title=('purpose')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig3.update_xaxes(title_text='purpose', row=1,col=2)

    fig3.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig3.show()

    

with saving_acct_out:

    data_grouped4 = data.groupby('saving_acct')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped4 = pd.DataFrame(data_grouped4)

    data_grouped4.columns = ['saving_acct','target','count']

    data_grouped4_0 = data_grouped4[data_grouped4['target']==0]

    data_grouped4_1 = data_grouped4[data_grouped4['target']==1]

    

    data_grouped4['percent_bad']=data_grouped4.groupby('saving_acct')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped4_bad = data_grouped4[data_grouped4['target']==1]

    data_grouped4_good = data_grouped4[data_grouped4['target']==0]

    

    #fig4 = go.Figure()

    fig4 = make_subplots(rows=1, cols=2,subplot_titles=('Saving accounst / Bonds ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig4.add_trace(go.Bar(x=data_grouped4_0['saving_acct'], y=data_grouped4_0['count'], name = 'non_default',marker_color='blue',text = data_grouped4_0['count'], textposition='auto'))

    fig4.add_trace(go.Bar(x=data_grouped4_1['saving_acct'], y=data_grouped4_1['count'], name = 'default',marker_color='red',text =data_grouped4_1['count'], textposition ='auto'))

    

    fig4.add_trace(go.Bar(x=data_grouped4_good['saving_acct'], y=data_grouped4_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped4_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig4.add_trace(go.Bar(x=data_grouped4_bad['saving_acct'], y=data_grouped4_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped4_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig4.update_layout(barmode='group', xaxis=dict(title=('saving_acc')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig4.update_xaxes(title_text='saving_acct', row=1,col=2)

    fig4.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig4.show()

    

with present_empl_out:

    data_grouped5 = data.groupby('present_empl')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped5 = pd.DataFrame(data_grouped5)

    data_grouped5.columns = ['present_empl','target','count']

    data_grouped5_0 = data_grouped5[data_grouped5['target']==0]

    data_grouped5_1 = data_grouped5[data_grouped5['target']==1]

    

    data_grouped5['percent_bad']=data_grouped5.groupby('present_empl')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped5_bad = data_grouped5[data_grouped5['target']==1]

    data_grouped5_good = data_grouped5[data_grouped5['target']==0]

    

    #fig5 = go.Figure()

    fig5 = make_subplots(rows=1, cols=2,subplot_titles=('Present employment since', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig5.add_trace(go.Bar(x=data_grouped5_0['present_empl'], y=data_grouped5_0['count'], name = 'non_default',marker_color='blue',text = data_grouped5_0['count'], textposition='auto'))

    fig5.add_trace(go.Bar(x=data_grouped5_1['present_empl'], y=data_grouped5_1['count'], name = 'default',marker_color='red',text =data_grouped5_1['count'], textposition ='auto'))

    

    fig5.add_trace(go.Bar(x=data_grouped5_good['present_empl'], y=data_grouped5_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped5_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig5.add_trace(go.Bar(x=data_grouped5_bad['present_empl'], y=data_grouped5_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped5_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig5.update_layout(barmode='group', xaxis=dict(title=('present_empl')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig5.update_xaxes(title_text='present_empl', row=1,col=2)

    fig5.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig5.show()

    

with sex_out:

    data_grouped6 = data.groupby('sex')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped6= pd.DataFrame(data_grouped6)

    data_grouped6.columns = ['sex','target','count']

    data_grouped6_0 = data_grouped6[data_grouped6['target']==0]

    data_grouped6_1 = data_grouped6[data_grouped6['target']==1]

    

    data_grouped6['percent_bad']=data_grouped6.groupby('sex')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped6_bad = data_grouped6[data_grouped6['target']==1]

    data_grouped6_good = data_grouped6[data_grouped6['target']==0]

    

    #fig6 = go.Figure()

    fig6 = make_subplots(rows=1, cols=2,subplot_titles=('Marital status / sex ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig6.add_trace(go.Bar(x=data_grouped6_0['sex'], y=data_grouped6_0['count'], name = 'non_default',marker_color='blue',text = data_grouped6_0['count'], textposition='auto'))

    fig6.add_trace(go.Bar(x=data_grouped6_1['sex'], y=data_grouped6_1['count'], name = 'default',marker_color='red',text =data_grouped6_1['count'], textposition ='auto'))

    

    fig6.add_trace(go.Bar(x=data_grouped6_good['sex'], y=data_grouped6_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped6_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig6.add_trace(go.Bar(x=data_grouped6_bad['sex'], y=data_grouped6_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped6_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig6.update_layout(barmode='group', xaxis=dict(title=('sex')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig6.update_xaxes(title_text='sex', row=1,col=2)

    fig6.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig6.show()



with other_debtor_out:

    data_grouped7 = data.groupby('other_debtor')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped7 = pd.DataFrame(data_grouped7)

    data_grouped7.columns = ['other_debtor','target','count']

    data_grouped7_0 = data_grouped7[data_grouped7['target']==0]

    data_grouped7_1 = data_grouped7[data_grouped7['target']==1]

    

    data_grouped7['percent_bad']=data_grouped7.groupby('other_debtor')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped7_bad = data_grouped7[data_grouped7['target']==1]

    data_grouped7_good = data_grouped7[data_grouped7['target']==0]

    

    #fig7 = go.Figure()

    fig7 = make_subplots(rows=1, cols=2,subplot_titles=('Other debtors / guarantors', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig7.add_trace(go.Bar(x=data_grouped7_0['other_debtor'], y=data_grouped7_0['count'], name = 'non_default',marker_color='blue',text = data_grouped7_0['count'], textposition='auto'))

    fig7.add_trace(go.Bar(x=data_grouped7_1['other_debtor'], y=data_grouped7_1['count'], name = 'default',marker_color='red',text =data_grouped7_1['count'], textposition ='auto'))

    

    fig7.add_trace(go.Bar(x=data_grouped7_good['other_debtor'], y=data_grouped7_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped7_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig7.add_trace(go.Bar(x=data_grouped7_bad['other_debtor'], y=data_grouped7_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped7_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig7.update_layout(barmode='group', xaxis=dict(title=('other_debtor')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig7.update_xaxes(title_text='other_debtor', row=1,col=2)

    fig7.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig7.show()

    

with property_out:

    data_grouped8 = data.groupby('property')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped8 = pd.DataFrame(data_grouped8)

    data_grouped8.columns = ['property','target','count']

    data_grouped8_0 = data_grouped8[data_grouped8['target']==0]

    data_grouped8_1 = data_grouped8[data_grouped8['target']==1]

    

    data_grouped8['percent_bad']=data_grouped8.groupby('property')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped8_bad = data_grouped8[data_grouped8['target']==1]

    data_grouped8_good = data_grouped8[data_grouped8['target']==0]

    

    #fig8 = go.Figure()

    fig8 = make_subplots(rows=1, cols=2,subplot_titles=('Property ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig8.add_trace(go.Bar(x=data_grouped8_0['property'], y=data_grouped8_0['count'], name = 'non_default',marker_color='blue',text = data_grouped8_0['count'], textposition='auto'))

    fig8.add_trace(go.Bar(x=data_grouped8_1['property'], y=data_grouped8_1['count'], name = 'default',marker_color='red',text =data_grouped8_1['count'], textposition ='auto'))

    

    fig8.add_trace(go.Bar(x=data_grouped8_good['property'], y=data_grouped8_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped8_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig8.add_trace(go.Bar(x=data_grouped8_bad['property'], y=data_grouped8_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped8_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

        

    fig8.update_layout(barmode='group', xaxis=dict(title=('property')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig8.update_xaxes(title_text='property', row=1,col=2)

    fig8.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig8.show()



with other_install_out:

    data_grouped9 = data.groupby('other_install')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped9 = pd.DataFrame(data_grouped9)

    data_grouped9.columns = ['other_install','target','count']

    data_grouped9_0 = data_grouped9[data_grouped9['target']==0]

    data_grouped9_1 = data_grouped9[data_grouped9['target']==1]

    

    data_grouped9['percent_bad']=data_grouped9.groupby('other_install')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped9_bad = data_grouped9[data_grouped9['target']==1]

    data_grouped9_good = data_grouped9[data_grouped9['target']==0]

    

    #fig9 = go.Figure()

    fig9 = make_subplots(rows=1, cols=2,subplot_titles=('Other installment plans ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig9.add_trace(go.Bar(x=data_grouped9_0['other_install'], y=data_grouped9_0['count'], name = 'non_default',marker_color='blue',text = data_grouped9_0['count'], textposition='auto'))

    fig9.add_trace(go.Bar(x=data_grouped9_1['other_install'], y=data_grouped9_1['count'], name = 'default',marker_color='red',text =data_grouped9_1['count'], textposition ='auto'))

    

    fig9.add_trace(go.Bar(x=data_grouped9_good['other_install'], y=data_grouped9_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped9_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig9.add_trace(go.Bar(x=data_grouped9_bad['other_install'], y=data_grouped9_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped9_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig9.update_layout(barmode='group', xaxis=dict(title=('other_install')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig9.update_xaxes(title_text='other_install', row=1,col=2)

    fig9.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig9.show()

    

with housing_out:

    data_grouped10 = data.groupby('housing')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped10 = pd.DataFrame(data_grouped10)

    data_grouped10.columns = ['housing','target','count']

    data_grouped10_0 = data_grouped10[data_grouped10['target']==0]

    data_grouped10_1 = data_grouped10[data_grouped10['target']==1]

    

    data_grouped10['percent_bad']=data_grouped10.groupby('housing')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped10_bad = data_grouped10[data_grouped10['target']==1]

    data_grouped10_good = data_grouped10[data_grouped10['target']==0]

    

    

    #fig10 = go.Figure()

    fig10 = make_subplots(rows=1, cols=2,subplot_titles=('Housing', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig10.add_trace(go.Bar(x=data_grouped10_0['housing'], y=data_grouped10_0['count'], name = '0',marker_color='blue',text = data_grouped10_0['count'], textposition='auto'))

    fig10.add_trace(go.Bar(x=data_grouped10_1['housing'], y=data_grouped10_1['count'], name = '1',marker_color='red',text =data_grouped10_1['count'], textposition ='auto'))

    

    fig10.add_trace(go.Bar(x=data_grouped10_good['housing'], y=data_grouped10_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped10_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig10.add_trace(go.Bar(x=data_grouped10_bad['housing'], y=data_grouped10_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped10_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig10.update_layout(barmode='group', xaxis=dict(title=('housing')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig10.update_xaxes(title_text='housing', row=1,col=2)

    fig10.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig10.show()

    

with job_out:

    data_grouped11 = data.groupby('job')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped11 = pd.DataFrame(data_grouped11)

    data_grouped11.columns = ['job','target','count']

    data_grouped11_0 = data_grouped11[data_grouped11['target']==0]

    data_grouped11_1 = data_grouped11[data_grouped11['target']==1]

    

    data_grouped11['percent_bad']=data_grouped11.groupby('job')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped11_bad = data_grouped11[data_grouped11['target']==1]

    data_grouped11_good = data_grouped11[data_grouped11['target']==0]

    

    #fig11 = go.Figure()

    fig11 = make_subplots(rows=1, cols=2,subplot_titles=('Job ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig11.add_trace(go.Bar(x=data_grouped11_0['job'], y=data_grouped11_0['count'], name = 'non_default',marker_color='blue',text = data_grouped11_0['count'], textposition='auto'))

    fig11.add_trace(go.Bar(x=data_grouped11_1['job'], y=data_grouped11_1['count'], name = 'default',marker_color='red',text =data_grouped11_1['count'], textposition ='auto'))

    

    fig11.add_trace(go.Bar(x=data_grouped11_good['job'], y=data_grouped11_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped11_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig11.add_trace(go.Bar(x=data_grouped11_bad['job'], y=data_grouped11_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped11_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig11.update_layout(barmode='group', xaxis=dict(title=('job')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig11.update_xaxes(title_text='job', row=1,col=2)

    fig11.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig11.show()

    

with tlf_out:

    data_grouped12 = data.groupby('tlf')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped12 = pd.DataFrame(data_grouped12)

    data_grouped12.columns = ['tlf','target','count']

    data_grouped12_0 = data_grouped12[data_grouped12['target']==0]

    data_grouped12_1 = data_grouped12[data_grouped12['target']==1]

    

    data_grouped12['percent_bad']=data_grouped12.groupby('tlf')['count'].apply(lambda x: x.astype(float)/x.sum()).round(2)

    data_grouped12_bad = data_grouped12[data_grouped12['target']==1]

    data_grouped12_good = data_grouped12[data_grouped12['target']==0]

    

    #fig12 = go.Figure()

    fig12 = make_subplots(rows=1, cols=2,subplot_titles=('Telephone avaiability ', 'Percentage of bad loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig12.add_trace(go.Bar(x=data_grouped12_0['tlf'], y=data_grouped12_0['count'], name = 'non_default',marker_color='blue',text = data_grouped12_0['count'], textposition='auto'))

    fig12.add_trace(go.Bar(x=data_grouped12_1['tlf'], y=data_grouped12_1['count'], name = 'default',marker_color='red',text =data_grouped12_1['count'], textposition ='auto'))

    

    fig12.add_trace(go.Bar(x=data_grouped12_good['tlf'], y=data_grouped12_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped12_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig12.add_trace(go.Bar(x=data_grouped12_bad['tlf'], y=data_grouped12_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped12_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    fig12.update_layout(barmode='group', xaxis=dict(title=('tlf')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig12.update_xaxes(title_text='tlf', row=1,col=2)

    fig12.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig12.show()

    

with foreign_out:

    data_grouped13 = data.groupby('foreign')['target'].apply(lambda x: x.value_counts()).reset_index()

    data_grouped13 = pd.DataFrame(data_grouped13)

    data_grouped13.columns = ['foreign','target','count']

    data_grouped13_0 = data_grouped13[data_grouped13['target']==0]

    data_grouped13_1 = data_grouped13[data_grouped13['target']==1]

    

    data_grouped13['percent_bad']=((data_grouped13.groupby('foreign')['count'].apply(lambda x: x.astype(float)/x.sum()))*100).round(2)

    data_grouped13_bad = data_grouped13[data_grouped13['target']==1]

    data_grouped13_good = data_grouped13[data_grouped13['target']==0]

    

   # fig13 = go.Figure()

    fig13 = make_subplots(rows=1, cols=2,subplot_titles=('Foreign', 'Percentage of bad loans vs good loans'),horizontal_spacing = 0.1,column_widths=[0.9, 0.9])

    fig13.add_trace(go.Bar(x=data_grouped13_0['foreign'], y=data_grouped13_0['count'], name = 'non_default',marker_color='blue',text = data_grouped13_0['count'], textposition='auto'))

    fig13.add_trace(go.Bar(x=data_grouped13_1['foreign'], y=data_grouped13_1['count'], name = 'default',marker_color='red',text =data_grouped13_1['count'], textposition ='auto'))

    

    fig13.add_trace(go.Bar(x=data_grouped13_good['foreign'], y=data_grouped13_good['percent_bad'],showlegend=False, marker_color='blue',text=data_grouped13_good['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    fig13.add_trace(go.Bar(x=data_grouped13_bad['foreign'], y=data_grouped13_bad['percent_bad'],showlegend=False, marker_color='red',text=data_grouped13_bad['percent_bad'],textposition ='auto',opacity=0.4),row=1,col=2)

    

    

    

    fig13.update_layout(barmode='group', xaxis=dict(title=('foreign')), yaxis=dict(title='Count'),template='plotly_white',height=650, width=1400)

    fig13.update_xaxes(title_text='foreign', row=1,col=2)

    fig13.update_yaxes(title_text='Bad loans (%)',row=1,col=2)

    fig13.show()
cat_features = data.select_dtypes(exclude='number')

cat_features.head()
cat_features = cat_features.apply(lambda x: x.astype('category').cat.codes)
cat_features.corr().round(2)
# check dimensions

print('Dimensions train data (cat_features):', cat_features.shape)
# get dummies for cat variables

x_cat = data.select_dtypes(exclude='number')

x_cat = pd.get_dummies(x_cat)
# join to num variables

x_num = data.select_dtypes(exclude='object')

x = x_num.merge(x_cat,left_index=True,right_index=True)
# drop the target

x = x.drop(['target'],axis=1)
# split data

X = x.copy()

y = data['target'] # get the target for the test data
# check dimensions

print('Dimensions train data (X):', X.shape)

print('----------')

print('Dimensions test data (y):',y.shape)
# import libraries for ML

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
# dataset divided in two parts in a ratio ofn 75:25, meaning that 75% of the data will be used for training, and 25% will be used for testing

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
# save the column names for later

X_train_columns = X_train.columns
# feature scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

# get the col names back

X_train= pd.DataFrame(X_train, columns=X_train_columns)



X_test = sc.transform(X_test)

# get the col names back

X_test = pd.DataFrame(X_test, columns=X_train_columns)
logreg = LogisticRegression(C=0.015,solver='lbfgs').fit(X_train, y_train)

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))

print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
# predict the Test set results

y_pred = logreg.predict(X_test)
confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))