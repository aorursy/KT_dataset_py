#Importing Libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly #interactive visualization

import plotly.graph_objs as go

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/kickstarter-projects/ks-projects-201801.csv')
df.head()
df_state = df["state"].value_counts().reset_index().rename(

    columns={"index": "State","state":"Project"})

df_state
fig1=px.bar(df_state, x="State",y='Project',title="Status of Projects")

fig1.show()

fig2=px.pie(df_state, values='Project',names='State',title="Distribuition of States")

fig2.show()
df_category = df["category"].value_counts().reset_index().rename(

    columns={"index": "Category","category":"Project"})
fig3=px.bar(df_category.head(25), x='Category',y='Project',title="Top 25 Categories by the Number of Projects")

fig3.show()

df_category.loc[df_category['Project'] < 4000, 'Category'] = 'Other Categories' # Represent only large categories

fig4=px.pie(df_category, values='Project',names="Category",title="Distribuition of Projects by Category Name")

fig4.show()
main_category = df["main_category"].value_counts().reset_index().rename(

    columns={"index": "Main Category","main_category":"Project"})
fig5=px.bar(main_category, x='Main Category',y='Project',title="Projects by Main Category Name")

fig5.show()

fig6=px.pie(main_category, values='Project',names="Main Category",title="Distribuition of Project by Main Category Name")

fig6.show()
failed= df[df["state"] == "failed"]["main_category"].value_counts().reset_index().rename(

    columns={"main_category":"Failed","index": "Main Category"})

successful = df[df["state"] == "successful"]["main_category"].value_counts().reset_index().rename(

    columns={"main_category":"Succesful","index": "Main Category"})

canceled = df[df["state"] == "canceled"]["main_category"].value_counts().reset_index().rename(

    columns={"main_category":"Canceled","index": "Main Category"})

undefined = df[df["state"] == "undefined"]["main_category"].value_counts().reset_index().rename(

    columns={"main_category":"Undefined","index": "Main Category"})

live = df[df["state"] == "live"]["main_category"].value_counts().reset_index().rename(

    columns={"main_category":"Live","index": "Main Category"})

suspended = df[df["state"] == "suspended"]["main_category"].value_counts().reset_index().rename(

    columns={"main_category":"Suspended","index": "Main Category"})



#Maybe there is an easy way to merge the data but I wanted to try this method.



main_cats_merged=failed.merge(successful,on="Main Category")



state=[canceled,undefined,live,suspended]



for i in state:

    main_cats_merged=main_cats_merged.merge(i,on="Main Category")



main_cats_merged
fig7 = go.Figure(data=[

    go.Bar(name='Failed',    x=main_cats_merged["Main Category"], y=main_cats_merged["Failed"]),

    go.Bar(name='Succesful', x=main_cats_merged["Main Category"], y=main_cats_merged["Succesful"]),

    go.Bar(name='Canceled',  x=main_cats_merged["Main Category"], y=main_cats_merged["Canceled"]),

    go.Bar(name='Undefined', x=main_cats_merged["Main Category"], y=main_cats_merged["Undefined"]),

    go.Bar(name='Live',      x=main_cats_merged["Main Category"], y=main_cats_merged["Live"]),

    go.Bar(name='Suspended', x=main_cats_merged["Main Category"], y=main_cats_merged["Suspended"])

])

# Change the bar mode

fig7.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'},title="Distribuition of Projects by Status")

fig7.show()
df['launch_year']=pd.to_datetime(df['launched'], format="%Y/%m/%d").dt.year

fig8 = sns.countplot(df.launch_year)

plt.xlabel("Year")

plt.ylabel("Project")

plt.title("Number of Projects by Years")

plt.show(fig8)

df_state_successful = df.loc[df.state=='successful'].drop(

    ['ID','goal','launched','pledged','usd pledged','deadline','launch_year'],axis=1)

df_state_failed = df.loc[df.state=='failed'].drop(

    ['ID','goal','launched','pledged','usd pledged','deadline','launch_year'],axis=1)
successful_mean=df_state_successful.groupby(['main_category']).mean().sort_values(by=['backers'],ascending=False)

successful_mean

fig9=px.bar(successful_mean, x=successful_mean.index,y="backers",title="Backers by Main Category-Succesful Projects")

fig9.show()
successful_mean=df_state_successful.groupby(['main_category']).mean().sort_values(by=['usd_pledged_real'],ascending=False)

fig10 = go.Figure()



fig10.add_trace(

    go.Bar(

        x=successful_mean.index,

        y=successful_mean.usd_pledged_real,

        name="usd_pledged_real"

    ))



fig10.add_trace(

    go.Scatter(

        x=successful_mean.index,

        y=successful_mean.usd_goal_real,

        name="usd_goal_real",

        mode="markers"

    ))



fig10.update_layout(title="Pledged vs Goal Values (Real $)-Successful Projects")

fig10.show()
failed_mean=df_state_failed.groupby(['main_category']).mean().sort_values(by=['backers'],ascending=False)

failed_mean


fig11=px.bar(failed_mean, x=failed_mean.index,y="backers",title="Backers by Main Category-Failed")

fig11.show()
failed_mean=df_state_failed.groupby(['main_category']).mean().sort_values(by=['usd_goal_real'],ascending=False)

fig12 = go.Figure()



fig12.add_trace(

    go.Scatter(

        x=failed_mean.index,

        y=failed_mean.usd_pledged_real,

        name="usd_pledged_real",

        mode="markers"

    ))



fig12.add_trace(

    go.Bar(

        x=failed_mean.index,

        y=failed_mean.usd_goal_real,

        name="usd_goal_real"

    ))

fig12.update_layout(title="Pledged vs Goal Values (Real $)-Failed Projects")

fig12.show()