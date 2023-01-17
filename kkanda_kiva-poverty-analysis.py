import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
import os
print(os.listdir("../input"))
kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
print (kiva_loans_df.head(5))
kiva_mpi_region_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
print (kiva_mpi_region_locations_df.head(5))
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
print (loan_theme_ids_df.head(5))
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
print (loan_themes_by_region_df.head(5))
#  Average funded amount by country
mean_funded_amt_reg = kiva_loans_df.groupby('country').agg({'funded_amount':'mean'}).reset_index()
mean_funded_amt_reg.columns = ['country','funded_amount_mean']
mean_funded_amt_reg = mean_funded_amt_reg.sort_values('funded_amount_mean',ascending=False)
print (mean_funded_amt_reg.head())

data = [ dict(
        type = 'choropleth',
        locations = mean_funded_amt_reg['country'],
        locationmode = 'country names',
        z = mean_funded_amt_reg['funded_amount_mean'],
        text = mean_funded_amt_reg['country'],
        colorscale='Earth',
        reversescale=False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Average Funded Loan Amount'),
      ) ]

layout = dict(
    title = 'Average Funded Loan Amount by Country (US Dollars)',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        showlakes = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot(fig, validate=False)
## Top 25 countries with most number of loans
loan_by_country = kiva_loans_df.groupby('country')['country'].count().sort_values(ascending=False).head(25)
#print (loan_by_country)
plt.figure(figsize=(16,8))
sns.barplot(x=loan_by_country.index,y=loan_by_country.values,palette="BuGn_d")
plt.xticks(rotation="vertical")
plt.xlabel("Countries with most number of borrowers",fontsize=20)
plt.ylabel("Number of active loans",fontsize=18)
plt.title("Top 25 countries with most number of active loans", fontsize=22)
plt.show()
## Top 25 regions with most number of loans sanctioned
loan_by_con_reg = kiva_loans_df.groupby('country')['region'].value_counts().nlargest(25)     
plt.figure(figsize=(16,8))
sns.barplot(x=loan_by_con_reg.index,y=loan_by_con_reg.values,palette="RdBu_r")
plt.xticks(rotation="vertical")
plt.xlabel("Regions with most number of borrowers",fontsize = 16)
plt.ylabel("Number of loans",fontsize=14)
plt.title("Top 25 Regions with most number of active loans",fontsize = 20)
plt.show()
## Let's find out the activities that got more loans approved and plot them by sector
activity_type_by_sector = kiva_loans_df.groupby(['sector','activity']).size().sort_values(ascending=False).reset_index(name='total_count')
activity_type_by_sector = activity_type_by_sector.groupby('sector').nth((0,1)).reset_index()
plt.figure(figsize=(16,10))
sns.barplot(x="activity",y="total_count",data=activity_type_by_sector)
plt.xticks(rotation="vertical")
plt.xlabel("Activites with most number of Borrowers",fontsize = 16)
plt.ylabel("Number of loans",fontsize=14)
plt.title("Top Two Actvities By Sector vs Number Of Loans",fontsize = 20)
plt.show()
kiva_loans = kiva_loans_df.filter(['country','region','funded_amount','loan_amount','activity','sector','borrower_genders', 'repayment_interval'])
kiva_loans = kiva_loans.dropna(subset=['country','region'])          
#print (kiva_loans.shape)

kiva_mpi_region_locations = kiva_mpi_region_locations_df[['country','region','world_region','MPI','lat','lon']]
kiva_mpi_region_locations = kiva_mpi_region_locations.dropna()                         
#print (kiva_mpi_region_locations.shape)
mpi_values = pd.merge(kiva_mpi_region_locations,kiva_loans,how="left")
#print (mpi_values.shape)
# Distribution of top 50 poorest regions by MPI
mpi_values_df = mpi_values.sort_values(by=['MPI'],ascending=False).head(50)
data = [ dict(
        type = 'scattergeo',
        lon = mpi_values_df['lon'],
        lat = mpi_values_df['lat'],
        text = mpi_values_df['MPI'].astype('str') + ' ' + mpi_values_df.region + ' ' + mpi_values_df.country,
        mode = 'markers',
        marker = dict(
            size = mpi_values_df['MPI']/0.04,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            cmin = 0,
            color = mpi_values_df['MPI'],
            cmax = mpi_values_df['MPI'].max(),
            colorbar=dict(
                title="MPI"
            )
        ))]

layout = dict(
        title = 'Top 50 Poverty Regions by MPI',
        colorbar = True,
        geo = dict(
            showland = True,
            landcolor = "rgb(250, 250, 250)",
        ),
    )

fig1= dict( data=data, layout=layout)
iplot(fig1, validate=False)
# Let's see the distribution of average loan/funded_amount by MPI
mpi_values_amount = mpi_values.groupby('MPI',as_index=False)['loan_amount','funded_amount'].mean()
mpi_values_amount = mpi_values_amount.dropna(subset=['loan_amount','funded_amount'],how="all")
#print (mpi_values_amount.shape)
fig,ax = plt.subplots(figsize=(15,8))
lns1 = ax.plot('MPI','loan_amount',data=mpi_values_amount,label="Loan amount",color="Blue")
ax2 = ax.twinx()
lns2 = ax2.plot('MPI','funded_amount',data=mpi_values_amount,label = "Funded amount",color="Green")

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.set_title("Loan/Funded Amount by MPI")
ax.set_xlabel("MPI")
ax.set_ylabel("Loan Amount (in US dollars)")
ax.tick_params(axis='y',labelcolor="Blue")
ax.set_ylim(100,10000)
ax2.set_ylabel("Funded Loan Amount (in  US dollars)")
ax2.tick_params(axis='y',labelcolor="Green")
ax2.set_ylim(100,10500)
plt.show()
## Let's plot the total amount funded for each world region
fund_by_world_region = mpi_values.groupby('world_region')['funded_amount'].sum()
plt.figure(figsize=(15,8))
sns.barplot(x = fund_by_world_region.index, y=fund_by_world_region.values,log=True)
#plt.xticks(rotation="vertical")
plt.xlabel("World Region",fontsize=20)
plt.ylabel("Funded loan amount(US dollars)",fontsize=20)
plt.title("Funded loan amount by world-region",fontsize=22)
plt.show()
# Join/Merge Loan,loan_theme,loan_theme_region datasets for further Analysis
kiva_loans_df.rename(columns={'partner_id':'Partner ID'},inplace=True)
loan_themes  = kiva_loans_df.merge(loan_theme_ids_df,how='left').merge(loan_themes_by_region_df,on=['Loan Theme ID','Partner ID','country','region'])
print (loan_themes.columns)
theme_country = loan_themes.groupby(['country'])['Loan Theme Type_x'].value_counts().sort_values(ascending=False)
theme_country = theme_country.unstack(level=0)
theme_country = theme_country.unstack().dropna(how="any")
theme_country = theme_country.sort_values(ascending=False).reset_index(name="loan_theme_count")
theme_country = theme_country.rename(columns={'Loan Theme Type_x': 'Loan_Theme_Type'})
theme_country = theme_country.head(20)
print (theme_country)
theme_country.iplot(kind="bar",barmode="stack",title="Loan Theme Count by Theme Type and Country")
# Lets analyse the distribution of average funded_amount and funded_amount count by by loan theme type
amt_theme = loan_themes.groupby('Loan Theme Type_x').agg({'funded_amount':['mean','count']}).reset_index()
amt_theme.columns = ['Loan Theme Type_x','funded_amount_mean','funded_amount_count']
#print (amt_theme)
amt_theme = amt_theme.sort_values(['funded_amount_mean','funded_amount_count'],ascending=[False,False])
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111) 
ax2 = ax.twinx() 
width = 0.4
amt_theme = amt_theme.set_index('Loan Theme Type_x')
amt_theme.head(20).funded_amount_mean.plot(kind='bar',color='dodgerblue',width=width,ax=ax,position=0)
amt_theme.head(20).funded_amount_count.plot(kind='bar',color='goldenrod',width=width,ax=ax2,position=1)
ax.grid(None, axis=1)
ax2.grid(None)
ax.set_xlabel("Loan Theme Type")
ax.set_ylabel('Funded Amount(mean)')
ax2.set_ylabel('Funded Amount(count)')
ax.set_title("Mean funded amount and count by Loan Theme Type")
plt.show()
# Analyze loan count by field partner name
loancount_by_fpartner = loan_themes.groupby(['Field Partner Name'])['Field Partner Name'].count().sort_values(ascending=False).head(20)
plt.figure(figsize=(15,8))
pal = sns.color_palette("Oranges", len(loancount_by_fpartner))
sns.barplot(x=loancount_by_fpartner.index,y=loancount_by_fpartner.values,palette=np.array(pal[::-1]))
plt.xticks(rotation="vertical")
plt.xlabel("Field Partner Name",fontsize=20)
plt.ylabel("Loan Count",fontsize=20)
plt.title("Loan count by Field Partners",fontsize=22)
plt.show()
# Lowest 10 average loan amount by field partner name
mloan_fpartner = loan_themes.groupby('Field Partner Name')['loan_amount'].mean().sort_values(ascending=True)
print (mloan_fpartner.head(10))
mloan_fpartner.head(15).iplot(kind='bar',yTitle="Average Loan Amount(US Dollars)",title="Average Loan amount by Field Partner Name")
# Clean the borrower_genders column to replace list of male/female values to 'group'
mask = ((loan_themes.borrower_genders!= 'female') &
                                  (loan_themes.borrower_genders != 'male') & (loan_themes.borrower_genders != 'NaN'))
loan_themes.loc[mask,'borrower_genders'] = 'group'
print (loan_themes.borrower_genders.unique())
bgenders = loan_themes.borrower_genders.value_counts()
labels = bgenders.index
values = bgenders.values
trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)
layout = go.Layout(
    title="Borrowers' genders"
)
data = trace
fig = go.Figure(data=[data], layout=layout)
iplot(fig,validate=False)
# Funded loan amount by gender/type
plt.figure(figsize=(9,6))
g = sns.violinplot(x="borrower_genders",y="funded_amount",data=loan_themes,order=['male','female','group'])
plt.xlabel("")
plt.ylabel("Funded Loan Amount",fontsize=12)
plt.title("Funded loan amount by borrower type/gender",fontsize=15)
plt.show()
# plot distribution of lender count by Gender/Type
loan_themes['lender_count_lval'] = np.log(loan_themes['lender_count'] + 1)
(sns
  .FacetGrid(loan_themes, 
             hue='borrower_genders', 
             size=6, aspect=2)
  .map(sns.kdeplot, 'lender_count_lval', shade=True,cut=0)
 .add_legend()
)
plt.xlabel("Lender count")
plt.show()