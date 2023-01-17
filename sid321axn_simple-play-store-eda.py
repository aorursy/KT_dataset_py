import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams
df=pd.read_csv("../input/googleplaystore.csv")
df.shape
df.head(3)
df.tail(3)
print(df.dtypes)
df.drop_duplicates(subset='App', inplace=True)
df = df[df['Android Ver'] != np.nan]
df = df[df['Android Ver'] != 'NaN']
df = df[df['Installs'] != 'Free']
df = df[df['Installs'] != 'Paid']
# - Installs : Remove + and ,

df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: int(x))
#print(type(df['Installs'].values))
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)


df['Size'] = df['Size'].apply(lambda x: float(x))
df['Installs'] = df['Installs'].apply(lambda x: float(x))

df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
x = df['Rating'].dropna()
y = df['Size'].dropna()
z = df['Installs'][df.Installs!=0].dropna()
p = df['Reviews'][df.Reviews!=0].dropna()
t = df['Type'].dropna()
price = df['Price']

g = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, price)), 
                        columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type', palette="rainbow")



sns.countplot(x='Type', data=df)
labels =df['Type'].value_counts(sort = True).index
sizes = df['Type'].value_counts(sort = True)


colors = ["lightblue","orangered"]
explode = (0.1,0)  # explode 1st slice
 
rcParams['figure.figsize'] = 8,8
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percent of Free App in store',size = 25)
plt.show()
sns.countplot(x='Category',data=df)
plt.xticks(rotation=90,ha="right",size = 10)
plt.figure(figsize = (10,10))
paid_apps = df[df.Price>0]
p = sns.jointplot( "Price", "Rating", paid_apps, size=8)
correl = df.corr()
#f, ax = plt.subplots()
p =sns.heatmap(correl, annot=True, cmap='coolwarm',linecolor='white',linewidths=1)
from pylab import rcParams

rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(df.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)
g = sns.catplot(x="Category",y="Rating",data=df, kind="swarm", height = 10 ,
palette = "Set1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Swarm plot of Rating VS Category',size = 20)
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(df.Installs, color="Red", shade = True)
g.set_xlabel("Installs")
g.set_ylabel("Frequency")
plt.title('Distribution of Installs',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'green',data=df[df['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)
df1 = df.dropna()  #Delete missing values

df2 = df1.groupby('Category')
#group1.App.count().sort_values(by='App', ascending = False)
df_inst = df2.Installs.sum().sort_values(ascending = False)

df_inst = pd.DataFrame(df_inst)

df_inst[:10]
fig,ax = plt.subplots(figsize=(25,10))
plt.scatter(x=df["Genres"],y=df["Rating"],color="green",marker="o")
plt.xticks(rotation=90)
plt.grid()
plt.show()
import pyecharts as pe

pe.configure(
    jshost='https://cdnjs.cloudflare.com/ajax/libs/echarts/3.7.2/',
    echarts_template_dir=None,
    force_js_embed=None,
    output_image=None,
    global_theme=None
)
import pyecharts
from pyecharts import Pie
pie = Pie("Total Installs for different categories", "",title_pos='center')

pie.add("categories", 
        df_inst.index[:10], 
        df_inst.Installs[:10],
        radius=[60, 65],
        label_pos='right',
        label_text_size = 9,
        label_text_color='black',
        is_label_show=True,
        legend_orient='vertical',
        legend_pos="left",
        legend_text_size = 9
       )

pie
df_free = df[df.Type == 'Free'].dropna()
df_paid = df[df.Type == 'Paid'].dropna()

dfc = df_free.groupby('Category')
#group1.App.count().sort_values(by='App', ascending = False)
df_i = dfc.Installs.sum().sort_values(ascending = False)
df_i = pd.DataFrame(df_i)[:10]

df_p = df_paid.groupby('Category')
df6 = df_p.Installs.sum().sort_values(ascending = False)
df6 = pd.DataFrame(df6)[:10]
df_i
from pyecharts import Funnel

attr = df_i.index
value = df_i.Installs
funnel = Funnel("Funnel chart of various APP downloads under free conditions", title_pos='center')
funnel.add(
    "",
    attr,
    value,
    is_label_show=True,
    label_pos="inside",
    label_text_color="#fff",
    legend_orient='vertical',
    legend_pos="left",
    legend_text_size = 10
    
)
funnel
