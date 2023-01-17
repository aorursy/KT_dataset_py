import pandas as pd
df=pd.read_csv("../input/fitness-analysis/fitness analysis.csv")
df.info()

df.head()
df.columns
new_cols=['Timestamp','Name','Gender','Age','Exercise_importance','Fitness_level','Regularity','Barriers','Exercises','Do_you','Time','Time_spent','Balanced_diet','prevents_balanced','Health_level','Recommend_fitness','Equipment','Motivation']
column_reference=pd.DataFrame(new_cols,df.columns)

column_reference
df.columns=new_cols
df.drop(columns=['Timestamp','Name'],inplace=True)
df.head()
import seaborn as sns

import matplotlib.pyplot as plt
age_vals=df["Age"].unique()

grid = sns.FacetGrid(df, col='Age',col_order=age_vals[[1,0,4,3,2]])

grid.map(sns.distplot,'Exercise_importance')

grid.add_legend()

grid.set(xlim=(0,5))

grid.set(ylim=(0,1))

grid.set(xlabel="Importance of Exercise")

grid.despine()

plt.show()
exercises_list={}

for selected_options in df['Exercises']:

    for exercise in selected_options.split(";"):

        if exercise in exercises_list:

            exercises_list[exercise]+=1

        else:

            exercises_list[exercise]=1



            

        
sorted_list={}

for i in sorted(exercises_list,key=exercises_list.get,reverse=True):

    sorted_list[i]=exercises_list[i]

     
count=sum(sorted_list.values())

for i in sorted_list:

    sorted_list[i]=(sorted_list[i]/count)*100
sorted_list
plt.bar(sorted_list.keys(),sorted_list.values())

plt.xticks(rotation=90)

plt.title("Exercise preferred by Participants")

plt.ylabel("Percentage")

plt.ylim(0,100)

plt.show()
times=df["Time"].value_counts(normalize=True)*100

plt.pie(times,labels=times.index,explode=(0.05,0.05,0.1),shadow=True,autopct='%.1f%%',startangle=90)

plt.title("Preferred Time to exercise")

plt.show()
df["Time_spent"].unique()
times=df["Time_spent"].str.split(" ",n=1,expand=True)
df["Time_spent_minutes"]=times[0]
def convertor(val):

    if val=="I":

        return 0

    else:

        return int(val)
df["Time_spent_minutes"]=df["Time_spent_minutes"].apply(convertor)
df["Time_spent_minutes"]=df["Time_spent_minutes"].apply(lambda x:x*60 if x!=30 else x)
df["Time_spent_minutes"].value_counts()
df.groupby("Gender").mean()["Time_spent_minutes"].plot.barh()

for i,v in enumerate(df.groupby("Gender").mean()["Time_spent_minutes"]):

    plt.text(v,i,(str(round(v,2))+" mins"))

plt.title("Average time spent daily on Exercise by Gender")

plt.xlabel("Time (Minutes)")

plt.xlim(0,70)

plt.show()
groups=df.groupby("Age").mean()["Time_spent_minutes"]

ax=sns.barplot(groups.index,groups)

ax.text(0,48,"Total Average=45.6 minutes",c="purple")

plt.title("Average time spent on Exercise daily by Age")

plt.ylim(10,60)

plt.axhline(df["Time_spent_minutes"].mean(),color="purple")

plt.show()
motivation_list={}

for selected_options in df['Motivation']:

    for motivation in selected_options.split(";"):

        if motivation in motivation_list:

            motivation_list[motivation]+=1

        else:

            motivation_list[motivation]=1
motivation_list
top_5_motivation=pd.DataFrame.from_dict(motivation_list.items()).sort_values(by=1,ascending=False)[:5]
top_5_motivation[0]=top_5_motivation[0].apply(lambda x:x.replace("I want to ",""))
sns.barplot(x=0,y=1,data=top_5_motivation)

plt.xticks(rotation=45)

plt.ylabel("Number of responses")

plt.title("Top five reasons to exercise")

plt.xlabel("")

plt.show()
barrier_list={}

for selected_options in df['Barriers']:

    for barrier in selected_options.split(";"):

        if barrier in barrier_list:

            barrier_list[barrier]+=1

        else:

            barrier_list[barrier]=1
barrier_list
top_5_barrier=pd.DataFrame.from_dict(barrier_list.items()).sort_values(by=1,ascending=False)[:6].drop(3,axis=0)
top_5_barrier["Percentage"]=(top_5_barrier[1]/top_5_barrier[1].sum())*100
top_5_barrier
plt.pie(top_5_barrier["Percentage"],labels=top_5_barrier[0],autopct="%.1f%%",shadow=True)

centre_circle = plt.Circle((0,0),0.8,color='yellow', fc='white',linewidth=1.5)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title("Top 5 barriers to daily exercise")

plt.show()