from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
sns.set_context("poster")
from subprocess import check_output
print(check_output(["ls", "../input/lendingclub-issued-loans"]).decode("utf8"))
print(check_output(["ls", "../input/supplemental-files-to-loan-analysis"]).decode("utf8"))
# Reading loan dataset
df_loan=pd.read_csv("../input/lendingclub-issued-loans/lc_loan.csv")
print("Number of rows: " + str(df_loan.shape[0])+", Number of Columns: "+ str(df_loan.shape[1]))
# Reading US states dataset
df_us_states=pd.read_csv('../input/lendingclub-issued-loans/us-state-codes.csv', low_memory=False)
df_us_states['state_code']=df_us_states.state_code.apply(lambda x: x[-2:])
# Reading Region information from dataset
df_region=pd.read_csv('../input/supplemental-files-to-loan-analysis/Region.csv',low_memory=False)
df_region['State']=df_region['State'].apply(lambda x: x.replace(' ',''))
# Reading US state wise population information
df_population = pd.read_csv("../input/supplemental-files-to-loan-analysis/population.csv")
df_population['state']=df_population['state'].apply(lambda x: x.replace(' ',''))
# Merge datasets to create final dataframe 'main_df'
main_df=df_loan.merge(df_us_states, left_on='addr_state', right_on='state_code')
main_df=main_df.merge(df_region, left_on='state', right_on='State')
main_df=pd.merge(main_df, df_population, on='state')
main_df=main_df.drop(['state','addr_state'], axis=1)
main_df['issue_year']=main_df['issue_d'].apply(lambda x: x[-4:]) # create 'issue year' column from 'loan issue date'
main_df['emp_title']=main_df['emp_title'].str.lower() # Convert all employee titles to lower for wordcloud usage
main_df['Region']=main_df['Region'].apply(lambda x: x.replace(' ',''))
main_df['Default']=main_df['loan_status'].apply(lambda x: 1 if (x=='Default') |( x=='Charged Off') else 0)# create default variable
main_df.head(3)
df_grade_int=main_df.groupby(['grade','issue_year'], as_index=False)['int_rate'].mean()
trace0=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='A'],
    mode='lines',
    name='grade_A'
)
trace1=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='B'],
    mode='lines',
    name='grade_B'
)
trace2=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='C'],
    mode='lines',
    name='grade_C'
)
trace3=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='D'],
    mode='lines',
    name='grade_D'
)
trace4=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='E'],
    mode='lines',
    name='grade_E'
)
trace5=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='F'],
    mode='lines',
    name='grade_F'
)
trace6=go.Scatter(
    x=df_grade_int.issue_year,
    y=df_grade_int.int_rate[df_grade_int.grade=='G'],
    mode='lines',
    name='grade_G'
)
layout=go.Layout(title="Average Interest Rate By Loan Grade (2007-2015)",
                 font=dict(size=18),
                 xaxis={'title':'Loan Issue Year',
                    'tickfont':dict(size=16)}, 
                 yaxis={'title':'Average interest rate (%)',
                       'tickfont':dict(size=16)},
                 showlegend=False)
annotations=[]
for i in df_grade_int.index:
    if df_grade_int.iloc[i,1]=='2014':
        annotations.append(dict(x=2014, y=df_grade_int.iloc[i,2]+0.5, text="Grade "+df_grade_int.iloc[i,0],
                                font=dict(family='Arial', size=14,
                                color='rgba(0, 0, 102, 1)'),
                                showarrow=False,))    
    layout['annotations']=annotations
data=[trace0, trace1, trace2, trace3, trace4, trace5, trace6]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)  ## Riskier the loan higher the interest rate
# Creating dataframe of average interest rate by grade of loan
df_int_rate=main_df.groupby(['issue_year', 'grade'],as_index=False).agg({'int_rate':'mean'})
df_2008=df_int_rate[df_int_rate.issue_year=='2008']   
df_2015=df_int_rate[df_int_rate.issue_year=='2015']
df_2008.index=['A', 'B', 'C', 'D', 'E', 'F', 'G']
df_2015.index=['A', 'B', 'C', 'D', 'E', 'F', 'G']
df_int_rate_diff=round(df_2015.int_rate-df_2008.int_rate, 3)
data=[go.Bar({
    'x':df_int_rate_diff.index,
    'y':df_int_rate_diff.values,  
    'marker':dict(
        color=['rgba(115, 244, 65,1)', 'rgba(115, 244, 65,0.8)',
               'rgba(244, 65, 98,1)', 'rgba(244, 65, 98,1)',
               'rgba(244, 65, 98,1)', 'rgba(244, 65, 98,1)', 
               'rgba(244, 65, 98,1)']),
    'opacity':1
})]
layout=go.Layout(title="Average change in interest rate by grade (2008-2015)",
                 font=dict(size=18),
                 xaxis={'title':'Grade of loan',
                    'tickfont':dict(size=16)}, 
                 yaxis={'title':'Change in interest rate (%)',
                       'tickfont':dict(size=16)})
annotations=[]
for i in df_int_rate_diff.index:
    temp=df_int_rate_diff[i]
    if i not in ['A', 'B']:
        temp=df_int_rate_diff[i] + 1
    annotations.append(dict(x=i, y=temp-0.5, text=df_int_rate_diff[i],
                                font=dict(family='Arial', size=14,
                                color='rgba(0, 0, 102, 1)'),
                                showarrow=False,))
    
    layout['annotations']=annotations
fig=dict(data=data, layout=layout)
py.iplot(fig)
df_state=main_df.groupby(['state_code'], as_index=False).agg({'loan_amnt':'count', 'Default':'mean',\
                                                        'population':'mean','State':'unique'})
df_state['pct_loan_issued']=df_state['loan_amnt']/df_state['loan_amnt'].sum()*100.00
df_state.loan_amnt=df_state.loan_amnt.apply(lambda x: str(x))
df_state.population=df_state.population.apply(lambda x: str(x))
df_state['text']=df_state['State'] +'\n'+\
            'Average Loan Amount: '+ df_state.loan_amnt+ " "+ \
            'Population: '+ df_state.population 
data=[dict(
        type='choropleth',
        autocolorscale=True,
        colorbar=dict(
            title="% Loan Count"),
        locations=df_state['state_code'],
        z=df_state['pct_loan_issued'],
        locationmode='USA-states' ,
        text = df_state['text']
        ),
     go.Scattergeo(
            lat=[34.27,31.17,46.71,40.74,25.97],
            lon=[-124.27,-100.07,-73.97,-89.50,-83.83], 
            mode="text",
            text=["California<br>129,517","Texas<br>71,138","New York<br>74,086",
                  "Illinois<br>35,476","Florida<br>60,935"],            
            textfont=dict(size=16,color='#000000')
        )]
layout=dict(
        title='% Loans Issued per State (2007-2015)<br>(Hover for breakdown)',
        font=dict(size=18),
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            lataxis=dict(range=[40, 70]),
            lonaxis=dict(range=[-130,-55])
            )) 
fig=dict(data=data, layout=layout)
py.iplot(fig)
df_state.loan_amnt=df_state.loan_amnt.apply(lambda x: float(x))
df_state.sort_values(by='loan_amnt',ascending=False).tail(5)
df_state.sort_values(by='loan_amnt',ascending=False).head(5)
# finding top 5 defaulter states
df_state.sort_values(by='Default',ascending=False).state_code.head(5).to_frame() 
# Creating dataframe to compare top 5 states to states with worst defaulter rates
df_state_default=main_df.groupby(['state_code'], as_index=False).agg({'loan_amnt':'count','Default':'mean'})
ten_states=['CA','NY','TX','FL','IL','ID','IA','NV','HI','AL'] 
df_ten_state=df_state_default[df_state_default.state_code.isin(ten_states)]
df_ten_state.Default=df_ten_state.Default*100
df_ten_state=df_ten_state.sort_values(by='Default',ascending=False)
trace1=go.Scatter(
    x=df_ten_state.state_code,
    y=df_ten_state.Default,
    yaxis='y2',
    name='Default Rate'
)
trace2=go.Bar(
    x=df_ten_state.state_code,
    y=df_ten_state.loan_amnt,
    name='Total Loans'
)

data=[trace1, trace2]
layout=go.Layout(
    title='Default Rate & Total Loans vs States',
    font=dict(size = 18),
    yaxis=dict(
        title='Total Loans'
    ),
    yaxis2=dict(
        title='Default rate',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    ),
    xaxis=dict(
        title='State'),
    legend=dict(x=1, y=1.2)
)
fig=go.Figure(data=data, layout=layout)
py.iplot(fig)
df_emp_cloud=main_df.emp_title.value_counts()[:40]
df_emp_cloud=df_emp_cloud.to_frame().reset_index()
d={}
for a, x in df_emp_cloud.values:
    d[a]=x
wordcloud=WordCloud(background_color='white',min_font_size=9, width=800,height=400)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show();
# Creating dataframe with count of loans per month
df_loan['issue_year']=df_loan.issue_d.str.split("-").apply(lambda x: x[1])
df_loan['issue_month']=df_loan.issue_d.str.split("-").apply(lambda x: x[0])
df_ts_count_month=df_loan.groupby(["issue_year", "issue_month"])['id'].count().to_frame().reset_index()
df_ts_count_month=df_ts_count_month.pivot(index=df_ts_count_month.issue_year, columns='issue_month')['id']
# Creating ordered categories for months
months=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df_ts_count_month=df_ts_count_month.reindex(pd.Categorical(
                    df_ts_count_month.columns, categories=months, ordered=True).sort_values(), axis=1)
data=[{
    'x':df_ts_count_month.index,
    'y':df_ts_count_month[col],
    'name':col,
    'opacity':1,
  }for col in df_ts_count_month.columns]
layout=go.Layout(title="Time series of loan count per year per month",
                 font=dict(size=18),
                 xaxis={'title':'Year',
                    'tickfont':dict(size=16)}, 
                 yaxis={'title':'Total Loans',
                       'tickfont':dict(size=16)})
fig=dict(data=data, layout=layout)
py.iplot(fig)
df_default_region=main_df.groupby(['Region','issue_year'],as_index=False)['Default'].mean()
trace0=go.Scatter(
    x=df_default_region.issue_year,
    y=df_default_region.Default[df_default_region.Region=='SOUTH'],
    mode='lines',
    name='South'
)

trace1=go.Scatter(
    x=df_default_region.issue_year,
    y=df_default_region.Default[df_default_region.Region=='WEST'],
    mode='lines',
    name='West'
)

trace2=go.Scatter(
    x=df_default_region.issue_year,
    y=df_default_region.Default[df_default_region.Region=='NORTHEAST'],
    mode='lines',
    name='NorthEast'
)

trace3=go.Scatter(
    x=df_default_region.issue_year,
    y=df_default_region.Default[df_default_region.Region=='MIDWEST'],
    mode='lines',
    name='MidWest'
)
layout=go.Layout(title="% Default Rate per Region (2007-2015)",
                 font=dict(size=18),
                 xaxis={'title':'Loan Issue Year',
                    'tickfont':dict(size=16)}, 
                 yaxis={'title':'% Default Rate',
                       'tickfont':dict(size=16)})
data=[trace0, trace1, trace2, trace3]
fig=dict(data=data, layout=layout)
py.iplot(fig) 
df_default_grade=main_df.groupby(['grade','issue_year'],as_index=False)['Default'].mean()
trace0=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='A'],
    mode='lines',
    name='grade_A'
)
trace1=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='B'],
    mode='lines',
    name='grade_B'
)
trace2=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='C'],
    mode='lines',
    name='grade_C'
)
trace3=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='D'],
    mode='lines',
    name='grade_D'
)
trace4=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='E'],
    mode='lines',
    name='grade_E'
)
trace5=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='F'],
    mode='lines',
    name='grade_F'
)
trace6=go.Scatter(
    x=df_default_grade.issue_year,
    y=df_default_grade.Default[df_default_grade.grade=='G'],
    mode='lines',
    name='grade_G'
)
layout=go.Layout(title="% Default Rate per Grade (2007-2015)",
                 font=dict(size=18),
                 xaxis={'title':'Loan Issue Year',
                    'tickfont':dict(size=16)}, 
                 yaxis={'title':'% Default Rate',
                       'tickfont':dict(size=16)})
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]
fig = dict(data=data, layout=layout)
py.iplot(fig) 
df_public_rec=main_df.groupby('pub_rec', as_index=False)['loan_amnt'].count()
trace0=go.Bar(
    x=df_public_rec.pub_rec,
    y=df_public_rec.loan_amnt
)
layout=go.Layout(title="Total Public Records Vs. Total Loans Issued<br>(Hover for breakdown)",
                 font=dict(size=18),
                 xaxis=dict(title='Total Public Records',
                    tickfont=dict(size=16), range=[0,10]), 
                 yaxis={'title':'Total loans',
                       'tickfont':dict(size=16)
                       })
data=[trace0]
fig=dict(data=data, layout=layout)
py.iplot(fig) 