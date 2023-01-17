import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
google = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
review = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")
google
review
print(google.info())
print("-"*50)
print(review.info())
#Missing data
print(google[google.columns[google.isnull().any()]].isnull().sum())
print("-"*50)
print(review[review.columns[review.isnull().any()]].isnull().sum())
google = google.dropna()
review = review.dropna()
print(google[google.columns[google.isnull().any()]].isnull().sum())
print("-"*50)
print(review[review.columns[review.isnull().any()]].isnull().sum())
#Duplicate data
print(google.duplicated().sum())
print(review.duplicated().sum())
google = google.drop_duplicates(['App'])
review = review.drop_duplicates()
print("-"*50)
print(google.duplicated().sum())
print(review.duplicated().sum())
#Top 10 Rating Apps
popular = google[google['Installs'] == '1,000,000,000+']
popular = popular[['App','Category','Rating','Reviews','Type','Content Rating']].sort_values(['Rating'], ascending=False).head(10)
display(popular)
#Percentage of Paid Apps by Installs, Category
import matplotlib.pyplot as plt
import seaborn as sns
def percent_paid(var):
    table = pd.DataFrame(pd.crosstab(google[var], google['Type']))
    table['paid_ratio'] = table['Paid']/(table['Free'] + table['Paid'])*100
    sns.barplot(table.index,table['paid_ratio'],order=table.sort_values('paid_ratio').index)
    plt.title("Percentage of Paid Apps", size=13)
    plt.ylabel("Paid app percent")
    plt.xticks(rotation='vertical')
    plt.show()
percent_paid('Installs')
percent_paid('Category')
#Distributions
list = ['Type','Installs','Content Rating','Category']
fig, axes = plt.subplots(len(list), 1, figsize=(12, 12))

for i, ax in enumerate(fig.axes):
    if i < len(list):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=google[list[i]], alpha=0.7, data=google, ax=ax, order=google[list[i]].value_counts().index)

fig.tight_layout()
#two-way table
from matplotlib import cm
table = pd.crosstab(index=google["Installs"], columns=google["Category"])
table['sum'] = table.sum(axis=1)
for i in table.columns[0:33]:
    table[i] = table[i]/table['sum']*100
table = table.drop(['sum'],axis=1)
table['sum'] = table.sum(axis=1)
table = table.drop(['sum'],axis=1)
display(table[['FAMILY','GAME','TOOLS','FINANCE','LIFESTYLE','PRODUCTIVITY','PERSONALIZATION','MEDICAL','PHOTOGRAPHY','BUSINESS','SPORTS','COMMUNICATION','SOCIAL']])
table.plot(kind="bar", 
                 figsize=(10,6),
                 stacked=True)
plt.legend(loc=7,bbox_to_anchor=(1.4,0.5))
google.columns
fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(pd.crosstab(google.Installs, google['Android Ver']),
            cmap="YlGnBu", annot=True, cbar=False, linewidths=.5,ax=ax)
import plotly.express as px
table = pd.crosstab(google.Installs, google['Content Rating'])
mtable = pd.melt(table.reset_index(), id_vars='Installs')
fig = px.line(mtable, x="Installs", y="value", color='Content Rating',title='Age limit and App installment numbers')
fig.show()
table = pd.crosstab(google.Installs, google['Size'])
mtable = pd.melt(table.reset_index(), id_vars='Installs')
df = mtable[mtable.Size != 'Varies with device']
fig = px.line(df, x="Installs", y="value", color='Size',title='App size and App installment numbers')
fig.show()
table = google[['Installs','Rating']].groupby(['Installs']).mean()
table = table.reset_index()
fig = px.bar(table, x="Installs", y="Rating",text='Rating',title='App size and App installment numbers')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()
