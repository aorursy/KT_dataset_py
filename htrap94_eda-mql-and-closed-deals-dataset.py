import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (15.0, 15.0)

plt.style.use('ggplot')

pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)

%matplotlib inline



leads_df = pd.read_csv("../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv")

closed_leads_df = pd.read_csv("../input/marketing-funnel-olist/olist_closed_deals_dataset.csv")
#helper function

def pichart_with_table(main_df,column_name,title,top_n,filename):

    fig = plt.figure(figsize=(10,6))



    summary = main_df.groupby(column_name)["mql_id"].nunique().sort_values(ascending=False)

    df = pd.DataFrame({'source':summary.index, 'counts':summary.values})

    labels = df['source']

    counts = df['counts']



    ax1 = fig.add_subplot(121)

    if top_n > 0:

        ax1.pie(counts[0:top_n], labels=labels[0:top_n], autopct='%1.1f%%', startangle=180)

    else:

        ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=180)

    ax1.set_title(title)

    ax1.axis('equal')



    ax2 = fig.add_subplot(122)

    font_size=10

    ax2.axis('off')

    if top_n > 0:

        df_table = ax2.table(cellText=df.values[0:top_n], colLabels=df.columns, loc='center',colWidths=[0.8,0.2])

    else:

        df_table = ax2.table(cellText=df.values, colLabels=df.columns, loc='center',colWidths=[0.8,0.2])



    df_table.auto_set_font_size(False)

    df_table.set_fontsize(font_size)



    fig.tight_layout()

    plt.savefig(filename)

    plt.show()

leads_df.describe(include="all")
pichart_with_table(leads_df,"origin","Origin Sources",-1,"origin_mql.png")
leads_df["first_contact_date"] = leads_df["first_contact_date"].astype("datetime64")

ldf = leads_df.groupby([leads_df["first_contact_date"].dt.year, leads_df["first_contact_date"].dt.month]).count()

ldf.index.names = ['year','month']

ldf = ldf.drop(['first_contact_date','landing_page_id','origin'], axis = 1) 

print(ldf)

ldf.plot(kind = "bar",legend = False)

plt.title("First Contact Month Counts")

plt.savefig('first_contact_mql.png')

plt.show()

closed_leads_df.describe(include="all")
pichart_with_table(closed_leads_df,"business_segment","Top 10 Segments",10,'business_segment_closed_deals.png')
pichart_with_table(closed_leads_df,"business_type","Business Types",-1,'business_type_closed_deals.png')
pichart_with_table(closed_leads_df,"lead_type","Lead Types",-1,'lead_type_closed_deals.png')
pichart_with_table(closed_leads_df,"lead_behaviour_profile","Lead Behaviour Profiles",5,'lead_behaviour_closed_deals.png')
summary_sdr = closed_leads_df.groupby("sdr_id")["mql_id"].nunique().sort_values(ascending=False)

sdr_df = pd.DataFrame({'sdr_id':summary_sdr.index, 'counts':summary_sdr.values})

sdr_df["total_percentage_impact"] = round(sdr_df["counts"] / len(closed_leads_df),3) * 100

print(sdr_df[0:10])
summary_sr = closed_leads_df.groupby("sr_id")["mql_id"].nunique().sort_values(ascending=False)

sr_df = pd.DataFrame({'sr_id':summary_sr.index, 'counts':summary_sr.values})

sr_df["total_percentage_impact"] = round(sr_df["counts"] / len(closed_leads_df),3) * 100

print(sr_df[0:10])


fig = plt.figure(figsize=(15,40))

combined_sr_sdr = closed_leads_df.groupby(["sdr_id","sr_id"]).size().sort_values(ascending=True)

combined_sr_sdr.plot(kind="barh")

fig.tight_layout()

plt.title("SDR - SR Pair Closed Deals Counts")

plt.savefig('charts/sdr_sr_mapping_count_closed_deals.png')

plt.show()

combined_sr_sdr
fig = plt.figure(figsize=(20,20))

closed_leads_df["won_date"] = closed_leads_df["won_date"].astype("datetime64")

ldf = closed_leads_df.groupby([closed_leads_df["won_date"].dt.year, closed_leads_df["won_date"].dt.month]).count()

ldf.index.names = ['year','month']

ldf.drop(ldf.iloc[:, 1:], inplace = True, axis = 1)

print(ldf)

ldf.plot(kind = "bar",legend = False)

fig.tight_layout()

plt.title("Closed Deals Month Counts")

plt.savefig('won_date_closed_deals.png')

plt.show()

funnel_df = closed_leads_df.merge(leads_df, on="mql_id", how= "left")

funnel_df.head()
#Datetime cleaning and period counting

funnel_df.first_contact_date = pd.to_datetime(funnel_df.first_contact_date)

funnel_df.won_date = pd.to_datetime(funnel_df.won_date)

funnel_df["close_duration"] = funnel_df.won_date-funnel_df.first_contact_date
#Checking if close_duration is proper or not (won date cannot be before first contact)

print("Erroneous values: - ",funnel_df[funnel_df.close_duration.values/np.timedelta64(1, 'D')<0].shape[0])

funnel_df.loc[funnel_df.close_duration.values/np.timedelta64(1, 'D')<0,["mql_id","first_contact_date","won_date","close_duration"]]
funnel_df.iloc[667]
#Removing the wrong value

funnel_df = funnel_df.drop(667)
#count missing values (NAs)

missing_count = pd.DataFrame(funnel_df.isna().sum(),columns=['Number'])

missing_count['Percentage'] = round(missing_count / len(funnel_df),2) * 100

missing_count
#Dropping columns has_company,has_gtin,average_stock,declared_product_catalog_size as they have substantial missing values (>90%)

funnel_df = funnel_df.drop(['has_company','has_gtin','average_stock','declared_product_catalog_size'], axis = 1) 



#filling the rest NAs

rest_cols = ['lead_behaviour_profile','origin','business_segment','lead_type','business_type']

funnel_df[rest_cols] = funnel_df[rest_cols].fillna('unknown')



#checking if any null is remaining or not

funnel_df[funnel_df.isnull().any(axis=1)]
#writing the csv for further use

funnel_df.to_csv("cleaned_marketing_funnel.csv",index=False)
funnel_df.describe(include="all")
#Days taken to close deals

fig = plt.figure(figsize=(15,6))

(funnel_df['close_duration'].astype('timedelta64[h]') / 24).plot.hist()

plt.title("Closed Deals day Counts")

plt.xlabel("No. of days")

fig.tight_layout()

plt.savefig('duration_days_closed_deals.png')

plt.show()

fig = plt.figure(figsize=(15,6))

(funnel_df['close_duration'].astype('timedelta64[h]') / 24).plot.hist()



plt.title("Refined look for 45 to 450 days count")

plt.xlabel("No of days")

plt.xlim(45,450)

plt.ylim(0,90)

fig.tight_layout()

plt.savefig('duration_45to450days_closed_deals.png')

plt.show()



fdf = funnel_df.landing_page_id.value_counts()

fdf[fdf.values > 5].plot(kind="bar")

plt.title("Landing pages count - closed deals")

plt.savefig("landing_page_counts.png")

plt.show()

fig = plt.figure(figsize=(10,10))

funnel_df.sr_id.value_counts().plot.bar()

plt.title("Top SRs")

plt.ylabel("No. of Deals closed")

fig.tight_layout()

plt.savefig('top_sr.png')

plt.show()
fig = plt.figure(figsize=(10,10))

funnel_df.sdr_id.value_counts().plot.bar()

plt.title("Top SDRs")

plt.ylabel("No. of Deals closed")

fig.tight_layout()

plt.savefig('top_sdr.png')

plt.show()
funnel_df.sr_id.value_counts()
fig = plt.figure(figsize=(5,5))



print(len(funnel_df[funnel_df.declared_monthly_revenue>0])) #total count

print(len(funnel_df[funnel_df.declared_monthly_revenue>0])/len(funnel_df)*100) #total percent of data

funnel_df[funnel_df.declared_monthly_revenue>0].declared_monthly_revenue.value_counts().plot.bar()

plt.title("Revenue of closed deals ($)")

plt.xlabel("Amount")

plt.ylabel("Total number of sellers")

fig.tight_layout()

plt.savefig("revenue_disclosed.png")

plt.show()
pichart_with_table(funnel_df,"origin","Origin Sources",-1,"origin_closed.png")