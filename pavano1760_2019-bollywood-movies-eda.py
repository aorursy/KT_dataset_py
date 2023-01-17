from IPython.display import Image
Image("../input/images/data-original.png")

import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime
import calendar
from scipy import stats
import pandas as pd
import seaborn as sns
sns.set(style="white", context="talk")
file=pd.read_csv('/kaggle/input/all-bollywood-movies-2019/2019_Movies.csv')
file.head(3)
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Open/Hide code"></form>''')
timestamp=file['Release date & duration'].str.split("|", n=3,expand=True)
timestamp['date']=timestamp[0]
timestamp['date'] = pd.to_datetime(timestamp['date'])
timestamp['Duration']=timestamp[1]
timestamp.drop(columns =[0,1], inplace = True) 
file.drop(columns =['Release date & duration'], inplace = True) 
timestamp.sample(3)
duration=timestamp['Duration'].str.split(" ", n=4,expand=True)
duration['hr']=duration[1].astype(int)
duration['hr']=duration['hr']*60
duration['min']=duration[3].astype(int)
duration['Duration']=duration['min'].add(duration['hr']) 
duration.drop(columns =[0,1,2,3,4,'hr','min'], inplace = True) 
timestamp.drop(columns =['Duration'], inplace = True) 
duration.sample(3)
timestamp['Month'] = timestamp['date'].dt.month 
def mapper(month):
    return month.strftime('%b') 
timestamp['Monthname']= timestamp['date'].apply(mapper)
timestamp['Day'] = timestamp['date'].dt.day
timestamp['Weekday']= timestamp['date'].dt.weekday_name
timestamp.drop(columns =['date'], inplace = True) 
timestamp.sample(3)
movie_type=file['Movie_type'].str.split("|", n=1,expand=True)
movie_type['Category']=movie_type[0]
movie_type['Certification']=movie_type[1].replace('UA',' UA')
movie_type.drop(columns =[0,1], inplace = True) 
movie_type.sample(3)
df = pd.concat([file,duration,timestamp,movie_type], axis=1)
df.drop(columns =['Movie_type'], inplace = True) 
df.head(5)
plt.figure(figsize=(15,6))
ax = sns.distplot(df['Duration'])
plt.figure(figsize=(16, 6))
sns.distplot(df['Crtitic_ratings'],  kde=True, label='Crtitic ratings')
sns.distplot(df['User_ratings'],  kde=True,label='User ratings')
plt.legend(prop={'size': 15})
plt.title('Histogram of Ratings of Critics and Users')
plt.xlabel('Ratings')
plt.ylabel('Density')
plt.figsize=(11,7)
Image("../input/images/critic_comic.png")
# monthwise movie
plt.figure(figsize=(15,8))
p = sns.countplot(data=df, x = 'Monthname',order = df['Monthname'].value_counts().index)
plt.figure(figsize=(13,5))
p = sns.countplot(data=df, x = 'Day',order = df['Day'].value_counts().index)
# releaseday
plt.figure(figsize=(15,5))
p = sns.countplot(data=df, x = 'Weekday')
from IPython.display import Image
Image("../input/3filehere/3.png")
# certification 
x2=df['Certification'].value_counts(ascending=False)
df2=pd.DataFrame(x2)
df2.reset_index(inplace=True)
df2.columns = ['Certification','Count']
labels = df2['Certification']
sizes = df2['Count']

fig1, ax1 = plt.subplots()
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0) 
ax1.pie(sizes,explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()
# relation betn cretic and user ratings
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
# image of crtitic nd user rating
df3=df['Category'].str.split(",", n=3,expand=True)
df3.head(3)
ls=[' Drama ', ' Romance', None, ' Crime', ' Action', 'Romantic ',
       ' Romance ', ' Thriller ', ' Thriller', ' Action ', 'Drama',
       ' Comedy ', ' Adult', 'Comedy', ' Drama', ' Crime ', ' Comedy',
       'Drama ', ' History ', 'Thriller ', ' Biography ', ' Horror',
       ' Adventure', ' Period', ' Mystery', 'Crime', 'Fantasy ',
       ' History', ' Biography', ' Biopic ', 'Action ', ' Mystery ',
       'Comedy ']
d={}
for i in ls:
    s=df3[df3==i].count().sum()
    d[i]=s
print(d)
df_type=pd.DataFrame(d.items(), columns=['Type', 'Count'])
df_type = df_type.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df_type['Type'].unique()
df_type['Type']=df_type['Type'].replace(['Romance','Romantic'],'Romantic')
df_type['Type']=df_type['Type'].replace(['History','Period'],'History')
df_type['Type']=df_type['Type'].replace(['Biography ','Biopic'], 'Biography')
df_type['Type']=df_type['Type'].dropna()
df_type_pie=df_type.groupby(['Type']).agg('sum').reset_index()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
df_type_pie = df_type_pie.sort_values(by='Count',ascending=False)
df_type_pie["cumpercentage"] = df_type_pie["Count"].cumsum()/df_type_pie["Count"].sum()*100
fig, ax = plt.subplots()
fig.set_size_inches(20,7)
ax.bar(df_type_pie['Type'], df_type_pie["Count"], color="C0")
ax2 = ax.twinx()
ax2.plot(df_type_pie['Type'], df_type_pie["cumpercentage"], color="C1", marker="*", ms=15)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C2")
ax.axes.set_title("Movie Type Cumulative", fontsize=20, y=1.05);
plt.show()
df4=df['Language'].str.split(",", n=4,expand=True)
df4 = df4.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
ls2=[ 'Kannada',None, 'Tamil', 'Telugu', 'Malayalam', 'English','Hindi','nan', 'Urdu', 'Marathi']
d={}
for i in ls2:
    s=df4[df4==i].count().sum()
    d[i]=s

df_lang=pd.DataFrame(d.items(), columns=['Language', 'Count'])
df_lang
df_lang.drop([1,7,6], inplace=True)
sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
#sns.set_context("talk")
ax = sns.barplot(x=df_lang['Count'], y=df_lang['Language'], data=df_lang, orient='h', saturation=0.7)
ax.axes.set_title("Out of 230 Hindi Movies in other Indian languages released", fontsize=20, y=1.05);
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
df_lang = df_lang.sort_values(by='Count',ascending=False)
df_lang["cumpercentage"] = df_lang["Count"].cumsum()/df_lang["Count"].sum()*100
fig, ax = plt.subplots()
ax.bar(df_lang['Language'], df_lang["Count"], color="C0")
ax2 = ax.twinx()
ax2.plot(df_lang['Language'], df_lang["cumpercentage"], color="C1", marker="*", ms=10)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C2")
fig.set_size_inches(15.5, 5.5)
plt.show()
df['Actors'] = df['Actors'].astype(str)
df.head()
print('Movie of Maximum ratings by Critics')
df[df['Crtitic_ratings']==df['Crtitic_ratings'].max()]
print('Movie of Minimum ratings by Critics')
df[df['Crtitic_ratings']==df['Crtitic_ratings'].min()]
print('Movie of Maximum ratings by User')
df[df['User_ratings']==df['User_ratings'].max()]
print('Movie of Minimum ratings by User')
df[df['User_ratings']==df['User_ratings'].min()]
print('Logest Runtime Movie')
df[df['Duration']==df['Duration'].max()]
print('Shortst Runtime Movie')
df[df['Duration']==df['Duration'].min()]
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text = df.Actors.values
text = " ".join(df.Actors.values)
wordcloud = WordCloud(
    width = 3500,
    height = 1500,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
df['Actors'] = df['Actors'].str.replace(" ","")
text = df.Actors.values
text = " ".join(df.Actors.values)
wordcloud = WordCloud(
    width = 3500,
    height = 1500,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (30, 20),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
df['Actors'] = df['Actors'].str.replace(" ","")
text = df.Actors.values
text = " ".join(df.Actors.values)
wordcloud = WordCloud(
    width = 3500,
    height = 1500,
    background_color = 'black',max_words=3,
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (30, 20),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()