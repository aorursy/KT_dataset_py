import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import bq_helper
from bq_helper import BigQueryHelper
usa = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
bq_assistant = BigQueryHelper("bigquery-public-data", "usa_names")
bq_assistant.list_tables()
bq_assistant.head("usa_1910_2013", num_rows=3)
bq_assistant.table_schema("usa_1910_current")
# create a helper object for this dataset
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

# query and export data 
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name ORDER BY year ASC"""
agg_names = usa_names.query_to_pandas_safe(query)
agg_names.to_csv("usa_names.csv")
agg_names.head(5)
plt.pie(agg_names.groupby("gender")["number"].sum(),labels=["Female","Male"],autopct='%1.0f%%',colors=["lightpink","skyblue"])
plt.figure(figsize=(20,6))
ax=agg_names[agg_names["gender"]=="F"].groupby("year")["number"].sum().plot(label="Female",color="lightpink")
ax=agg_names[agg_names["gender"]=="M"].groupby("year")["number"].sum().plot(label="Male",color="skyblue")
plt.legend()
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.title('Total applicants per year')
from wordcloud import WordCloud, STOPWORDS
wordcloud=WordCloud(max_font_size=50,stopwords=STOPWORDS,background_color="black").generate(" ".join(agg_names["name"].sample(5000).tolist()))
plt.figure(figsize=(14,7))
plt.title("Wordcloud for top names",fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
top_name_byyear=agg_names.sort_values('number', ascending=False).drop_duplicates(['year',"gender"]).sort_values(by="year")
top_name_byyear.head()
ax=top_name_byyear[top_name_byyear["gender"]=="F"].groupby(by="name")["year"].count().sort_values(ascending=False).plot(kind="bar",label="Female name")
plt.title("Number of years the name was the top female name")
ax.set_xlabel("Top Female names")
ax.set_ylabel("Number of Years")
ax=top_name_byyear[top_name_byyear["gender"]=="M"].groupby(by="name")["year"].count().sort_values(ascending=False).plot(kind="bar",label="Male name")
plt.title("Number of years the name was the top male name")
ax.set_xlabel("Top Male names")
ax.set_ylabel("Number of Years")
agg_names["decade"]=agg_names["year"].apply(lambda row: row//10*10)
names_decade=pd.pivot_table(agg_names,aggfunc="sum",index=["decade","gender","name"],values="number").reset_index()
dec=names_decade.groupby(["decade","gender"])["number"].sum()
names_decade["dec_total"]=names_decade.apply(lambda row: dec[row["decade"]][row["gender"]],axis=1)
names_decade["perc"]=np.round(names_decade["number"]/names_decade["dec_total"],3)
three_top=pd.pivot_table(names_decade,values="perc",index=["decade","gender","name"])
three_top=three_top.groupby(level=["decade","gender"])["perc"].nlargest(3).reset_index(level=0,drop=True).reset_index(level=0,drop=True).reset_index()
three_top.head()
fig,axs=plt.subplots(figsize=(20,14),ncols=3,nrows=4,sharey=True)

decade=three_top["decade"].unique().tolist()
counter=0
for col in range(4):
    for row in range(3): 
        if col==3 and row==2:         
            axs[col][row].axis('off')
        elif col==0 and row==0:   
            sns.barplot(x="name", y="perc", hue="gender", data=three_top[three_top["decade"]==decade[counter]],ax=axs[col][row])
            axs[col][row].set_title(decade[counter]) 
            axs[col][row].set_xlabel("") 
            axs[col][row].set_ylabel("")    
        else:
            sns.barplot(x="name", y="perc", hue="gender", data=three_top[three_top["decade"]==decade[counter]],ax=axs[col][row])
            axs[col][row].set_title(decade[counter]) 
            axs[col][row].set_xlabel("") 
            axs[col][row].set_ylabel("") 
            axs[col][row].get_legend().remove()
            counter+=1
            
fig.suptitle("Top three names per decade and percentage of applicants with this name for male and female",fontsize=25,x=0.5,y=1.1)
fig.tight_layout()

order_top_average_names=agg_names.groupby("name")["number"].mean().sort_values(ascending=False).head(20).index.tolist()
data2=agg_names[agg_names["name"].isin(order_top_average_names) ][["year","name","number"]]
heat_map2=pd.pivot_table(data2,columns="year",index="name",values="number")
heat_map2.index = pd.Categorical(heat_map2.index, order_top_average_names)
heat_map2.sort_index(level=0, inplace=True)
plt.figure(figsize=(35,13))
plt.yticks(fontsize=20)
plt.xticks(fontsize=14, rotation=90)
sns.heatmap(heat_map2,cmap="YlGnBu",linewidths=0.5)
