import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


import re
import warnings
warnings.filterwarnings('ignore')

child = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
child.disbursed_time = pd.to_datetime(child.disbursed_time)
child.posted_time = pd.to_datetime(child.posted_time)
child.funded_time = pd.to_datetime(child.funded_time)
child.head(3)
"""Thank you @SRK"""
loan_sec = child['sector'].value_counts()
trace = go.Bar(
    y=loan_sec.index[::-1],
    x=loan_sec.values[::-1],
    orientation = 'h',
    marker=dict(
        color=loan_sec.values[::-1],
        colorscale = 'Picnic',
        reversescale = True
              ),
            )

layout = dict(
    title='Distribution of loans by sector',
    xaxis=dict(
        title='Number of loans'),
            )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Loans_by_sector')
print("It would be interesting to dig deeper and see what Personal Use refers to")
loan_activity = child['activity'].value_counts().head(40)
trace = go.Bar(
    x=loan_activity.index[::-1],
    y=loan_activity.values[::-1],
    orientation = 'v',
    marker=dict(
        color=loan_activity.values[::-1],
        colorscale = 'Picnic',
        reversescale = True
              ),
            )

layout = dict(
    title='Distribution of loans by activity (top 40)',
    yaxis=dict(
        title='Number of loans'),
    xaxis=dict(tickfont=dict(size=8))
            )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Loans_by_activity')
edu_all = child[child['sector']=='Education']

# find rows with the token child, son, daughter
c_child = edu_all['use'].apply(lambda x: len(re.findall(r"(child)", str(x)))) 
c_son = edu_all['use'].apply(lambda x: len(re.findall(r"([\s.,:;]son[\s.,:;s])|([\s.,:;]son\s*[.'’]+)", str(x)))) 
c_daughter = edu_all['use'].apply(lambda x: len(re.findall(r"([\s.,:;]daughter[\s.,:;s])|([\s.,:;]daughter\s*[.'’]+)", str(x))))

c_child = c_child[c_child.values>0].index
c_son = c_son[c_son.values>0].index
c_daughter = c_daughter[c_daughter.values>0].index

# combine indices of all loans with son, daughter, or child in their description
children = c_child.union(c_son)
children = children.union(c_daughter)

edu_child = child.iloc[children,:]
schooling = pd.read_csv('../input/school/schooling.csv')
schooling = schooling.rename(index=str, columns={"Human Development Index (HDI) ": "HDI", \
                                     "Gross national income (GNI) per capita":"GNI", \
                                     "Country": "country", \
                                     "GNI per capita rank minus HDI rank": "GNI_rank"})

edu_country = edu_child.groupby('country')['loan_amount'].sum()
edu_country = pd.DataFrame(edu_country).reset_index()

edu_country = edu_country.merge(schooling, on='country', how='left')
edu_country.dropna(inplace=True)
edu_country = edu_country[edu_country.loan_amount < 30000]

data = [go.Scatter(
    y = edu_country['Mean years of schooling'],
    x = edu_country['loan_amount'],
    mode='markers+text',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= (7 * (edu_country.HDI))**2,
        color=edu_country['Mean years of schooling'],
        colorscale='Portland',
        reversescale=True,
        showscale=True)
    ,text=edu_country['country']
    ,textposition=["top center"]
)]
layout = go.Layout(
    autosize=True,
    title='Years of Schooling vs. Sum of Education Loans for Children (under 30000 USD)',
    hovermode='closest',
    xaxis= dict(title='Sum of Loans', ticklen= 5, showgrid=True, zeroline=False, showline=False),
    yaxis=dict(title='Mean Years of Schooling', showgrid=True, zeroline=False, ticklen=5, gridwidth=2)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter_schooling_loans')
edu_for_everyone = child[child['sector']=='Education']['country'].value_counts()
edu_for_everyone = pd.DataFrame(edu_for_everyone).reset_index()
edu_for_everyone.columns = ['country', 'num_all_loans']

edu_for_children = edu_child['country'].value_counts()
edu_for_children = pd.DataFrame(edu_for_children).reset_index()
edu_for_children.columns = ['country', 'num_child_loans']

edu = edu_for_children.merge(edu_for_everyone, on='country', how='right')
edu.fillna(0, inplace=True)
edu['percent'] = edu['num_child_loans']/edu['num_all_loans']
edu['other_loans'] = edu['num_all_loans']-edu['num_child_loans']

edu = edu.sort_values(by='num_child_loans', ascending=False)
edu = edu[:15]
trace1 = go.Bar(
    x=edu['country'],
    y=edu['num_child_loans'],
    name='Child education_loans'
)
trace2 = go.Bar(
    x=edu['country'],
    y=edu['other_loans'],
    name='Non-child education_loans'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title="Education loans for child and non-child related reasons",
    yaxis=dict(title="Number of loans")
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
nigeria = edu_child.copy()
nigeria = nigeria[nigeria.country=='Nigeria']
print(nigeria.activity.value_counts())
print('All education loans for children are managed through 1 field partner ID 288: "Babban Gona Farmers Organization"')
print(nigeria.partner_id.value_counts())
print("The borrowers are overwhelmingly men")
print(nigeria.borrower_genders.value_counts())
for x in nigeria.use.head(10):
    print (x)
nigeria = edu_child.copy()
nigeria = nigeria[nigeria.country=='Nigeria']
nigeria['num']= 1 #add 1 to each row so we can count number of loans 

disbursed = nigeria.set_index(nigeria['disbursed_time'])
disbursed = disbursed.loc['2016-09':'2017-02'].resample('5D').sum()

posted = nigeria.set_index(nigeria['posted_time'])
posted = posted.loc['2016-09':'2017-02'].resample('5D').sum()

funded = nigeria.set_index(nigeria['funded_time'])
funded = funded.loc['2016-09':'2017-02'].resample('5D').sum()

plt.figure(figsize=(15,5))
plt.plot(disbursed['num'], color='green', label='Disbursed to borrower', marker='o')
plt.plot(posted['num'], color='red', label='Posted on kiva.org', marker='o')
plt.plot(funded['num'], color='blue', label='Funded on kiva.org', marker='o')
plt.legend(loc='upper left')
plt.title("Number of Education loans for Children, in 5-day intervals (Nigeria)")
plt.ylabel("Number of loans")
plt.show()
ed = child[child['sector']=='Education']
son = ed['use'].apply(lambda x: len(re.findall(r"([\s.,:;]son[\s.,:;s])|([\s.,:;]son\s*[.'’]+)", str(x)))) 
daughter = ed['use'].apply(lambda x: len(re.findall(r"([\s.,:;]daughter[\s.,:;s])|([\s.,:;]daughter\s*[.'’]+)", str(x))))

son = son[son.values>0]
daughter = daughter[daughter.values>0]

ed_son = child.iloc[son.index,:]
ed_daughter = child.iloc[daughter.index,:]
girl = round(ed_daughter.loan_amount.mean(),2)
boy = round(ed_son.loan_amount.mean(),2)

labels = ['Daughters','Sons']
values = [girl,boy]
colors = ['#38e1a3', '#ffeb38']

trace = go.Pie(labels=labels, values=values,
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=1)))
layout = go.Layout(
        title="Average Education Loans (USD) for Sons and Daughters")

fig = go.Figure(data=[trace], layout=layout)

py.iplot(fig, filename='basic_pie_chart')
edu_all = child[child['sector']=='Education']

c_son = edu_all['use'].apply(lambda x: len(re.findall(r"([\s.,:;]son[\s.,:;s])|([\s.,:;]son\s*[.'’]+)", str(x)))) 
c_daughter = edu_all['use'].apply(lambda x: len(re.findall(r"([\s.,:;]daughter[\s.,:;s])|([\s.,:;]daughter\s*[.'’]+)", str(x))))

c_son = c_son[c_son.values>0].index
c_daughter = c_daughter[c_daughter.values>0].index

#unite indices of all loans with son, daughter, or child in their description
son_dautr = c_daughter.union(c_son)

#slice original dataset to extract only son_daughter rows
edu_son_dautr = child.iloc[son_dautr,:]


dollars = edu_son_dautr.groupby('country')['loan_amount'].sum()
all_children = pd.DataFrame(dollars).reset_index()
data = [ dict(
        type = 'choropleth',
        locations = all_children['country'],
        locationmode = 'country names',
        z = all_children['loan_amount'],
        text = all_children['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.20,"rgb(40, 60, 190)"],[0.40,"rgb(70, 100, 245)"],\
            [0.55,"rgb(90, 120, 245)"],[0.75,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(280,10,30)',  # color of country borders
                width = 0.5                # width of country borders
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Loans (USD)'),
      ) ]

layout = dict(
    title = 'Total Education Loans for sons and daughters, by Country',
    geo = dict(
        showframe = True,                 # frame of full map
        showcoastlines = True,            # coastline of full map
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='education-loans-for-children-world-map')
print("Southern hemisphere is predominantely the main recipient of education loans")
b = child['use'].apply(lambda x: len(re.findall(r"([\s.,:;]son[\s.,:;s])|([\s.,:;]son\s*[.'’]+)", str(x)))) 
g = child['use'].apply(lambda x: len(re.findall(r"([\s.,:;]daughter[\s.,:;s])|([\s.,:;]daughter\s*[.'’]+)", str(x)))) 

b = b[b.values>0]
g = g[g.values>0]

b = child.iloc[b.index,:]
g = child.iloc[g.index,:]
plt.scatter(x=range(g.shape[0]), y=np.sort(g.loan_amount.values))
plt.title('Loan Amount for Daughters')
plt.xlabel('Index')
plt.ylabel('Loan Amount (\$$)')
plt.show()
# For Girls

# ulimit (4925.0) is the value below which 99% of observations in this group of loan observations fall
# i.e. 99% of observations in this group of loans fall under $4925.0
ulimit = np.percentile(g.loan_amount.values, 99) 
llimit = np.percentile(g.loan_amount.values, 1) # 1% of observations in this group fall below llimit ($125.0)

g['loan_trunc'] = g['loan_amount'].copy()
g['loan_trunc'].loc[g['loan_amount']>ulimit] = ulimit # if loan > $4925, assign 4925 to loan_amount 
g['loan_trunc'].loc[g['loan_amount']<llimit] = llimit # if loan < $125, assign 125 to loan_amount

# For Boys
ulimit = np.percentile(b.loan_amount.values, 99) 
llimit = np.percentile(b.loan_amount.values, 1) # 1% of observations in this group fall below llimit ($125.0)

b['loan_trunc'] = b['loan_amount'].copy()
b['loan_trunc'].loc[b['loan_amount']>ulimit] = ulimit
b['loan_trunc'].loc[b['loan_amount']<llimit] = llimit
"""Compute ECDF for a one-dimensional array of measurements."""
def ecdf(data, datab):
    
    # Number of data points: n
    n = len(data)
    nb= len(datab)

    # x-data for the ECDF: x
    x = np.sort(data)
    xb= np.sort(datab)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    yb= np.arange(1, nb+1) / nb

    return x, y, xb, yb

"""Plot the ECDF"""
x, y, xb, yb = ecdf(g.loan_trunc.values, b.loan_trunc.values)

plt.figure(figsize=(10,6))
plt.plot(x,y, linestyle='none', marker='.', color='yellow', alpha=0.9)
plt.plot(xb,yb, linestyle='none', marker='.', color='gray', alpha=0.4)

plt.xlabel("Loans")
plt.ylabel("Probability")
plt.legend(('daughters','sons'))
plt.xlim((-100,3000))
plt.show()
boys = b.sector.value_counts()
girls = g.sector.value_counts()

trace1 = go.Bar(
    x=girls.index,
    y=girls.values,
    name='Loans for daughters'
)
trace2 = go.Bar(
    x=boys.index,
    y=boys.values,
    name='Loans for sons'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title="Loans by sector (son or daughter)",
    yaxis=dict(title="Number of loans")
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar2')
plt.figure(figsize=[15,5])
sns.stripplot(x='sector', y='loan_amount', data=b, jitter=True)
plt.title('Loans in USD for --Sons--')
plt.xticks(rotation=25)
plt.ylabel('Loan(\$$)')
plt.show()

plt.figure(figsize=[15,5])
sns.stripplot(x='sector', y='loan_amount', data=g, jitter=True, order=b.sector.unique())
plt.title('Loans in USD for --Daughters--')
plt.xticks(rotation=25)
plt.ylabel('Loan(\$$)')
plt.show()
for x in g[g.sector=='Transportation']['use'].head(20):
    print (x)
"""Combine loans for all sons and daughters"""
sons = child['use'].apply(lambda x: len(re.findall(r"([\s.,:;]son[\s.,:;s])|([\s.,:;]son\s*[.'’]+)", str(x)))) 
dtrs = child['use'].apply(lambda x: len(re.findall(r"([\s.,:;]daughter[\s.,:;s])|([\s.,:;]daughter\s*[.'’]+)", str(x)))) 

sons = sons[sons.values>0].index
dtrs= dtrs[dtrs.values>0].index

# unite indices of all loans with son and daughter in their description
son_dtrs = sons.union(dtrs)

#slice original dataset to extract only son_dtrs rows
df = child.iloc[son_dtrs,:]
df['left_to_fund'] = df.loan_amount - df.funded_amount
child['left_to_fund'] = child.loan_amount - child.funded_amount

dm = df.left_to_fund.mean()
cm = child.left_to_fund.mean()
data = [go.Bar(
            x=['Sons and Daughters', 'All Loans'],
            y=[dm,cm]
    )]

layout = dict(
    title='Average $ Missing for Unfunded Loans',
    yaxis=dict(
        title="(USD)"),
            )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar2')
print("On average, loans for sons and daughters are $39 short when not fully funded, compared to $59 for all loans")
all_loans = child['country'].value_counts()
all_loans = pd.DataFrame(all_loans).reset_index()
all_loans.columns = ['country', 'num_all_loans']

not_funded = child[child.left_to_fund>0]['country'].value_counts()
not_funded = pd.DataFrame(not_funded).reset_index()
not_funded.columns = ['country', 'unfunded_loans']

percent = all_loans.merge(not_funded, on='country', how='left')
percent['prcnt_unfunded'] = percent.unfunded_loans / percent.num_all_loans
percent['prcnt_unfunded'] = percent['prcnt_unfunded'].apply(lambda x: 0.0 if pd.isnull(x) else x)
percent['prcnt_unfunded'] = round((percent['prcnt_unfunded']*100),2)
data = [ dict(
        type = 'choropleth',
        locations = percent['country'],
        locationmode = 'country names',
        z = percent['prcnt_unfunded'],
        text = percent.country,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.20,"rgb(40, 60, 190)"],[0.40,"rgb(70, 100, 245)"],\
            [0.55,"rgb(90, 120, 245)"],[0.75,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(280,10,30)',  # color of country borders
                width = 0.5                # width of country borders
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = '% of Loans not funded'),
      ) ]

layout = dict(
    title = 'Percentage of Loans Not fully funded',
    geo = dict(
        showframe = True,                 # frame of full map
        showcoastlines = True,            # coastline of full map
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='unfunded loans')
print("36% of loans in the United States end up not fully funded. Perhaps that is due to rarely using a field partner")
not_fully_funded = child[child.left_to_fund>0]
plt.figure(figsize=[15,7])
sns.violinplot(x='sector', y='left_to_fund', data=not_fully_funded)
plt.ylim((0,4000))
plt.ylabel("USD")
plt.title("Loans not Fully Funded (USD), by sector")
plt.xticks(rotation=30)
plt.show()
plt.figure(figsize=(10,5))
plt.scatter(not_fully_funded.left_to_fund, not_fully_funded.loan_amount)
plt.xlim((0,10000))
plt.xlabel("Missing $ to Fully Fund Loan")
plt.ylabel("Loan Size (USD)")
plt.show()