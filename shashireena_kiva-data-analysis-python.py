import pandas as pd
import numpy as np
import pandas_profiling as pro
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import warnings
warnings.filterwarnings("ignore")
loans_df=pd.read_csv("../input/kiva_loans.csv")
mpiregions_df=pd.read_csv("../input/kiva_mpi_region_locations.csv")
theme_df=pd.read_csv("../input/loan_theme_ids.csv")
themereg_df=pd.read_csv("../input/loan_themes_by_region.csv")
filename={"kiva_loans.csv":loans_df,
          "kiva_mpi_region_locations.csv":mpiregions_df,
          "loan_theme_ids.csv":theme_df,
          "loan_themes_by_region.csv":themereg_df}

def datastatfun(filenamedic):
    datastat=pd.DataFrame()
    for key in filenamedic.keys():
        row=pd.DataFrame({key:list(filename[key].shape)}).transpose()
        datastat=datastat.append(row)
    datastat.columns=["NoOfRows","NoOfColumns"]        
    return datastat    
        
datastatfun(filename) 
# Change data type to category

loans_df.activity=loans_df.activity.astype("category")
loans_df.sector=loans_df.sector.astype("category")
loans_df.country_code=loans_df.country_code.astype("category")
loans_df.country=loans_df.country.astype("category")
loans_df.region=loans_df.region.astype("category")

# function to print datatypes 

#Convert datatypes
loans_df["posted_time"]=pd.to_datetime(loans_df["posted_time"])
loans_df["disbursed_time"]=pd.to_datetime(loans_df["disbursed_time"])
loans_df["funded_time"]=pd.to_datetime(loans_df["funded_time"])


def get_var_category(colname):
    unique_cnt=colname.nunique(dropna=False)
    total_cnt=len(colname)
    if pd.api.types.is_numeric_dtype(colname):
        return 'Numeric'
    elif pd.api.types.is_datetime64_dtype(colname):
        return 'Date'
    elif unique_cnt==total_cnt:
        return 'Text(Unique)'
    else:
        return 'Categorical'


def check_dtypes(dataset):
    datastatlst=[]
    for col_name in dataset.columns:
        row={'Colname':[col_name],'DataTypes':[get_var_category(dataset[col_name])]}
        datastatlst.append(row)
    datastat = pd.DataFrame(datastatlst)
    return(datastat)



check_dtypes(loans_df)
#Missing value analysis for loan_df file

total=loans_df.isnull().sum().sort_values(ascending = False)
Percentnull=round((loans_df.isnull().sum()/list(loans_df.shape)[0])*100,2).sort_values(ascending=False)
missing_values=pd.concat([total,Percentnull],axis=1,keys=['Total', 'Percent'])
missing_values
# Data preparation
countryloans=loans_df.country.value_counts().reset_index()
countryloans.rename(columns={'index':'country','country':'NoOfLoans'},inplace=True)

#Top 15 Countries who have availed Kiva loans

trace = go.Table(
    header=dict(values=list(countryloans.columns),
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[countryloans.country, countryloans.NoOfLoans],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

data = [trace] 
iplot(data, filename = 'pandas_table')

# Plot using Ploty

data=[dict(
          type="choropleth",
          colorscale="Rainbow",
          autocolorscale=False,
          locations=countryloans["country"],
          z=countryloans.NoOfLoans,
          locationmode='country names',
          marker=dict(
                      line=dict(
                                 color="black",
                                 width=0.5
                      )),
         colorbar=dict(title="Number of loans ")               
    
    ) ]
    
layout=dict(
           title="  Kiva Loan analysis" ,
           geo=dict(showframe=False,showcoastlines=True,projection=dict(type='natural earth'))
            )
    
fig=dict(data=data,layout=layout)
iplot(fig)


#Pie plot for Gender based loans

#Data prep

loans_df['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in loans_df.borrower_genders]
Gendercnt=round((loans_df['borrower_genders'].value_counts()/(list(loans_df.shape)[0]))*100,2)
Gendercnt.values.tolist()

fig = {
  "data": [
    {
      "values": Gendercnt.values.tolist(),
      "labels": Gendercnt.index.tolist(),
      "textposition":"inside",
      "name": "Loan demographics",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Global Kiva loan distribution demographics",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Gender",
                "x": 0.52,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
countryloan_temp=loans_df[['loan_amount','country']].groupby(["country"])["loan_amount"].sum().reset_index().sort_values(["loan_amount"],ascending=False).iloc[0:20,]
countryloan_temp.loan_amount=round(countryloan_temp.loan_amount/1000000,3)
countryloan=countryloan_temp.reset_index(drop=True)

#matplotlib

fig,ax=plt.subplots(figsize=(10,5))
ax.bar(x=countryloan.index,height=countryloan.loan_amount,color=sns.color_palette('viridis_r',10),zorder=3)

#adds a title and axes labels
ax.set_title("Country vs Loan_amount")
ax.set_xlabel("Kiva Countries")
ax.set_ylabel("Loan amount in millions $")

#adds ticks
ax.set_xticks(countryloan.index)
ax.set_xticklabels(countryloan.country,rotation='vertical')

# remove Axes border
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

#Axes background color
rect = ax.patch
rect.set_facecolor('#E7F1FE')

#grid lines
ax.grid(color="white",linestyle="-",linewidth=0.90,alpha=0.8,zorder=0)

# Add bar heights in text
for rect in ax.patches:
    width = rect.get_width()
    xloc=rect.get_x()+width/2
       # Center the text vertically in the bar
    yloc = rect.get_height()+2
    ax.text(xloc, yloc, round(rect.get_height(),1),color='m', weight='bold',
                         clip_on=True,ha = 'center',va = 'center')

plt.show()
countryloan_temp=loans_df[['loan_amount','country']].groupby(["country"])["loan_amount"].sum().reset_index().sort_values(["loan_amount"],ascending=False).iloc[0:25,]
countryloan_temp.loan_amount=round(countryloan_temp.loan_amount/1000000,3)
countryloan=countryloan_temp.reset_index(drop=True)


fig,ax=plt.subplots(figsize=(10,5))
sns.set_style("darkgrid")
rec=sns.barplot(data=countryloan,x=countryloan.index,y="loan_amount",ax=ax)
# remove Axes border
rec.spines["top"].set_visible(False)
rec.spines["right"].set_visible(False)
rec.spines["left"].set_visible(False)
rec.spines["bottom"].set_visible(False)

#xaxis label
rec.set_label("Loan amount vs Country")
rec.set_xlabel("Country")
rec.set_ylabel("Kiva loan amount in Millions $")

rec.set_xticks(countryloan.index)
rec.set_xticklabels(countryloan.country,rotation='vertical')

for axc in rec.patches:
    width=axc.get_width()
    xloc=axc.get_x()+width/2
    yloc=axc.get_height()+2
    rec.text(xloc,yloc,round(axc.get_height(),1),weight='bold',
                         clip_on=True,ha = 'center',va = 'center')    

plt.show()

#Using Ploty*************************************************************************************************

#trace creation
trace1=go.Bar(x=countryloan.country,y=countryloan.loan_amount\
              ,marker = dict(color = 'rgba(255, 174, 255, 0.5)'\
              ,line=dict(color='rgb(0,0,0)',width=1.5))\
              ,text=countryloans.country\
              ,name="Kiva loan")

# data
data = [trace1]

#Layout aesthetic
layout=go.Layout(title="Loan amount vs Country",autosize=False,width=800,height=500\
                ,xaxis=dict(title="Kiva Country",ticklen =8,tickangle=-45)\
                ,yaxis=dict(title='USD (millions)',titlefont=dict(size=16,color='rgb(107, 107, 107)')\
                            ,tickfont=dict(size=14,color='rgb(107, 107, 107)'))\
                ,legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'))

#Figure combines layout and data
fig=go.Figure(data=data,layout=layout)

iplot(fig)


sect=loans_df.sector.unique()
top10cntry=pd.DataFrame()
for cntry in countryloan.iloc[0:10,].country:
    row=loans_df[loans_df.country==cntry].groupby(["sector","activity"])["loan_amount"].median().reset_index()
    row["country"]=cntry
    row=row[["country","sector","activity","loan_amount"]]
    top10cntry=top10cntry.append(row)

top10cntry.rename(columns={"loan_amount":"MedianLoanAmount"},inplace=True)

#Code for adding missing activity for a country-Sector combination 
for i in top10cntry.country.unique():
        for j in sect:
            fullact=set(loans_df[loans_df.sector==j].activity.unique())
            subact= top10cntry[(top10cntry.sector==j)&(top10cntry.country==i)].activity
            diff= fullact.difference(subact)
            top10cntry=top10cntry.append(pd.DataFrame({"country":i,"sector":j,"activity":list(diff),"MedianLoanAmount":None}))


# Plot
fig,ax=plt.subplots(15,1,figsize=(17,100))
fig.subplots_adjust(hspace=.65)
for i,sectorval in enumerate(sect):
    sec_data=top10cntry[top10cntry.sector==sectorval]
    ax[i]=sns.swarmplot(x="activity",y="MedianLoanAmount",data=sec_data,hue="country",ax=ax[i])
    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[i].set_title(sectorval,color="red",fontsize=16)
    ax[i].set_xlabel([])
    axis = ax[i].xaxis
    ax[i].set_xticklabels(axis.get_ticklabels(),rotation=90,fontsize=13)

plt.tight_layout()
plt.show()
    

