import pandas as pd
import numpy as np
# Reading kiva loans file
kiva_loan = pd.read_csv("../input/kiva_loans.csv")
kiva_loan.head(n=3)
kiva_loan.shape[0]
# Reading Kiva Region Location file.
kiva_region_location = pd.read_csv("../input/kiva_mpi_region_locations.csv")
kiva_region_location.head(n=5)
# Reading Kiva Loan theme File
kiva_loan_theme = pd.read_csv("../input/loan_theme_ids.csv")
kiva_loan_theme.head(n=3)
# Reading Kiva Loan Themes by Region File
kiva_loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
kiva_loan_themes_by_region.head(n=3)
#Word Cloud for Loan Use Statements throws some intersting findings.For example.
#1. A good number of loans have been taken for buying  and maintaining the rickshaw.
#2. A good number of loans have been taken to start a turducken farm.
#3. Purchase Biscuits for Resale.
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
%matplotlib inline
fig = plt.figure(figsize=(18,12))
stopwords = set(STOPWORDS)
data = kiva_loan['use']
wordcloud = WordCloud(background_color='white',max_words=200,random_state=1,stopwords=stopwords,).generate(str(data))
plt.imshow(wordcloud)
plt.show()
# Word Cloud for female gender throws interesting insights.
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
%matplotlib inline
fig = plt.figure(figsize=(18,12))
stopwords = set(STOPWORDS)
kiva_loan_mod = kiva_loan[kiva_loan.borrower_genders.notnull()]
data = kiva_loan_mod[kiva_loan_mod['borrower_genders'].str.contains("female")]['use']
wordcloud = WordCloud(background_color='white',max_words=200,random_state=1,stopwords=stopwords).generate(str(data))
plt.imshow(wordcloud)
plt.show()
# Post Execution Notes:#The loan is being utilized for buying seeds,cements for house maintenance and construction etc.
import plotly
import plotly.plotly as py
#import plotly.figure_factory as ff
from plotly.tools import FigureFactory as ff
#plotly.tools.set_credentials_file(username='abhay.rnj', api_key='8MzosR0uht5BE3tfUlhd')
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
#from plotly.graph_objs import Scatter,Figure,Layout
from plotly.graph_objs import *
init_notebook_mode(connected=True)
# Loan Repayment Schedules
repayment_type_cnt = pd.DataFrame(kiva_loan.repayment_interval.value_counts()).head(n=50).reset_index()
repayment_type_cnt.columns =['RepaymentInterval','Number Of Loans']
repayment_type_cnt.head(n=50)
#print(repayment_type_cnt.index.values)
data = Bar(
    x = repayment_type_cnt['RepaymentInterval'],
    y = repayment_type_cnt['Number Of Loans'],
    marker = dict(
        color = repayment_type_cnt['Number Of Loans'],
        colorscale = 'Jet',
        reversescale = True
        ),
)
        
layout = Layout(
    title= 'Loan Repayment Interval'
)    

fig = dict(data = [data], layout = layout)
iplot(fig,filename="LoanRepaymentINtervals")
#Post Execution Notes: It appears that the repayment schedules of most of thr loans is either monthly or irregular.
repayment_terms_df = pd.DataFrame(kiva_loan.term_in_months.value_counts().reset_index())
repayment_terms_df.columns = ['Repayment_Terms','Number_Of_Loans']
repayment_terms_df.head(n=5)
data = Bar(x=repayment_terms_df['Repayment_Terms'],y=repayment_terms_df['Number_Of_Loans'],orientation="v",
          marker = dict(
          colorscale='Picnic',
          reversescale=True
          ),
)

layout = Layout(title='Terms Wise Distribution Of Loan',
               )
fig = Figure(data=[data],layout=layout)
iplot(fig,filename='LoanDistribution')
# Histogram for Loan Amount
#kiva_loan['loan_amount']
import seaborn as sns
import numpy as np

%matplotlib inline
upperLimit = np.percentile(kiva_loan['loan_amount'].values,98)
lowerLimit = np.percentile(kiva_loan['loan_amount'].values,2)
#print(kiva_loan.index.values)
kiva_loan['mod_loan_amount'] = kiva_loan['loan_amount'].copy()
kiva_loan['mod_loan_amount'].loc[kiva_loan['loan_amount'] > upperLimit] = upperLimit
kiva_loan['mod_loan_amount'].loc[kiva_loan['loan_amount'] < lowerLimit ] = lowerLimit
plt.figure(figsize=(12,10))
sns.distplot(kiva_loan.mod_loan_amount.values,bins=16)
plt.xlabel('Loan Amount Truncated',fontsize=12)
plt.title('Loan Amount Histogram')
plt.show()
# Post Execution Notes: Appears that most of the loans are below 1000.
loan_count_by_country_df = pd.DataFrame(kiva_loan['country'].value_counts()).head(n=25).reset_index()
loan_count_by_country_df.columns = ['Country','number_of_loans']
loan_count_by_country_df.head(n=2)
import plotly.offline as offline
#scl = [[0.0,'rgb(242,240,248)'],[0.20,'rgb(215,220,225)'],[0.40,'rgb(190,195,198)'],[0.60,'rgb(160,170,180)'],[0.80,'rgb(140,144,150)'],[1.0,'rgb(110,114,120)']]
colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],\
            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
loan_data = [ dict(
        type='choropleth',
        colorscale = colorscale,
        autocolorscale = False,
        reversescale = True,
        locations = loan_count_by_country_df['Country'],
        z = loan_count_by_country_df['number_of_loans'].astype(float),
        locationmode = 'USA-states',
        text = loan_count_by_country_df['Country'],
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 2
            )
        ),
        colorbar = dict(
            tickprefix = '',
            title = "Number Of Loans"
        )
    ) ]

loan_layout = dict(
    title = 'Loans Instances by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=loan_data,layout=loan_layout)
#py.iplot(fig,validate=False,filename='choropleth-map')
iplot(fig,validate=False,filename='choropleth-map')
#fig = ff.dict(data=loan_data,layout=loan_layout)
#fig = offline.plot({'data':loan_data,'layout':loan_layout},image='png')
#py.iplot(fig)
#Bar Graph Of number of loans by country
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams


%matplotlib inline
rcParams['figure.figsize'] = 15,10
x = loan_count_by_country_df['Country']
y = loan_count_by_country_df['number_of_loans']
figure_size = (18,15)
figure, ax = plt.subplots(figsize=figure_size)
ax = sns.barplot(x='Country', y ='number_of_loans',data=loan_count_by_country_df)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
for item in ax.get_xticklabels():
    item.set_rotation(45)
import plotly.graph_objs as go
init_notebook_mode(connected=True)
#loan_count_by_country_df
trace = go.Pie(labels=loan_count_by_country_df.Country,values=loan_count_by_country_df.number_of_loans)
layout = dict(
    title='Loans Distribution by Country',
)
fig = go.Figure(data=[trace],layout=layout)
#py.iplot(fig,filename='Loan Count By Country')
iplot(fig,filename='Loan_Distribution_By_Country')
# Post Execution Notes: The pie chart below shows that from countries like Philipines,Kenya,El Salvador,Cambodia highest number of loans are being taken.
# Number of Loans by Sector

# number_of_loans_by_sector = kiva_loan.groupby(['sector'])['id'].count()
# print(number_of_loans_by_sector)
# print("*************")
import plotly.graph_objs as go
import plotly.plotly as py

kiva_loans_count_by_sector = kiva_loan['sector'].value_counts().reset_index()
kiva_loans_count_by_sector.columns = ['Sector','NumberOfLoans']
kiva_loans_count_by_sector.head(n=5)
trace = go.Pie(labels=kiva_loans_count_by_sector.Sector,values=kiva_loans_count_by_sector.NumberOfLoans)
layout = dict(
    title='Loan Distribution By Sector',
)
fig = go.Figure(data=[trace],layout=layout)
iplot(fig,filename='Loan Distribution By Sector')
# Average Dollar Value of Loan per Country
average_loan_value_by_country_df = pd.DataFrame(kiva_loan.groupby(['country'])['loan_amount'].mean()).head(n=25).reset_index()
average_loan_value_by_country_df.columns = ['Country','AverageLoanAmount']
average_loan_value_by_country_df = average_loan_value_by_country_df.sort_values(by=['AverageLoanAmount'],ascending=False)
average_loan_value_by_country_df.head(n=5)
rcParams['figure.figsize'] = 15,10
x = average_loan_value_by_country_df['Country']
y = average_loan_value_by_country_df['AverageLoanAmount']
figure_size = (18,15)
figure, ax = plt.subplots(figsize=figure_size)
ax = sns.barplot(x='Country', y ='AverageLoanAmount',data=average_loan_value_by_country_df)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
for item in ax.get_xticklabels():
    item.set_rotation(45)
# Total Loan Dollar Value Per Country
total_loan_value_by_country_df = pd.DataFrame(kiva_loan.groupby(['country'])['loan_amount'].sum()).head(n=25).reset_index()
total_loan_value_by_country_df.columns = ['Country','TotalLoanAmount']
total_loan_value_by_country_df = total_loan_value_by_country_df.sort_values(by=['TotalLoanAmount'],ascending=False)
total_loan_value_by_country_df.head(n=5)
rcParams['figure.figsize'] = 15,10
x = total_loan_value_by_country_df['Country']
y = total_loan_value_by_country_df['TotalLoanAmount']
figure_size = (18,15)
figure, ax = plt.subplots(figsize=figure_size)
ax = sns.barplot(x='Country', y ='TotalLoanAmount',data=total_loan_value_by_country_df)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
for item in ax.get_xticklabels():
    item.set_rotation(45)