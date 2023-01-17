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

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/Mass Shootings Dataset.csv', encoding = "ISO-8859-1")
df.head()
df.isnull().sum()
df.Summary = df.Summary.fillna('Stephen Paddock shot and from Mandalay Bay Hotel Room')

df.Gender = df.Gender.fillna('M')

df.Race = df.Race.fillna('White')
df.isnull().sum()
df.Gender.replace(['M', 'M/F'], ['Male', 'Male/Female'], inplace=True)
t = pd.DataFrame(pd.Series(df['Race'].sort_values(ascending=True).unique()))

v = pd.DataFrame(pd.Series(['Asian', 'Asian', 'Asian','Black','Black','Black','Latino','Native American',

           'Native American', 'Other', 'Other', 'Other', 'Unknown', 'White', 'White', 

           'White', 'White', 'Black', 'Unknown', 'White']))

t.reset_index(inplace=True)

v.reset_index(inplace=True)

u = pd.merge(t,v,how='inner',on='index')
mydict = dict(zip(u['0_x'], u['0_y']))


for i in df.columns:

    if i == 'Race':

        df['Race'] = df[i].map(mydict) 
df.Date = pd.to_datetime(df.Date)
df1 = df.drop(['Total victims', 'Longitude', 'Latitude', 'S#', 'Summary', 'Title'], axis=1)
df1[df1['Location']=='Newtown, Connecticut']
df1.duplicated()

df2 = df1.drop_duplicates()
df2['Total victims'] = df2['Fatalities'] + df['Injured']

df2['Total victims'] = df2['Total victims'].astype(int)
df2.head()
df2['Year'] = df2['Date'].dt.year
yrvictims = df2.groupby('Year').sum()['Injured']

count = df2.groupby('Year').count()['Location']

yrfatalities = df2.groupby('Year').sum()['Fatalities']

e=pd.DataFrame(yrvictims)

e.reset_index(inplace=True)

e.shape
c = pd.DataFrame(count)

c.reset_index(inplace=True)

c.shape

t = pd.DataFrame(yrfatalities)

t.reset_index(inplace=True)

t.shape
z = pd.merge(e,c,how='inner',on='Year')

w = pd.merge(z,t, how='inner', on='Year')
w.head()
n_groups = w.Year



fig, ax = plt.subplots(figsize=(15,7))



bar_width = 0.3



opacity = 0.4



rects1 = plt.bar(n_groups, w['Location'], bar_width ,

                 alpha=opacity,

                 color='b',

                 label='Number of Attacks')



rects2 = plt.bar(n_groups + bar_width, w['Fatalities'], bar_width,

                 alpha=opacity,

                 color='r',

                 label='Total Number of Fatalities')



rects3 = plt.bar(n_groups + bar_width  * 2, w['Injured'], bar_width,

                 alpha=opacity,

                 color='g',

                 label='Total Number of Injured')



plt.xlabel('Year')

plt.ylabel('Count')

plt.title('Total numbers of Attacks, Fatalities and Injured')



plt.legend()



plt.tight_layout()

plt.show()
df2['City'] = df2['Location'].str.rpartition(',')[0]

df2['State'] = df2['Location'].str.rpartition(',')[2]
df2['Mental Health Issues'].replace(['unknown', 'unclear', 'Unclear'], ['Unknown','Unclear', 'Unclear'], inplace=True)
df2.head()
h = df.groupby(['Longitude', 'Latitude', 'Location']).count()['S#']

d = pd.DataFrame(h)

d.reset_index(inplace=True)

n = d['Location']

x = d['Longitude']

y = d['Latitude']



fig, ax = plt.subplots(figsize=(15,10))

ax.scatter(x, y, s=d['S#'] *75)

ax.set_ylabel('Longitude')

ax.set_xlabel('Latitude')

ax.set_title('Plotting Location by Number of Attacks', fontsize=18)

for i, txt in enumerate(n):

    ax.annotate(txt, (x[i], y[i]), fontsize=8)

plt.show()
df3 = df2.copy()



from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in df3.columns:

    df3[col] = labelencoder.fit_transform(df3[col])

df3.head()
plt.figure(figsize=(14,8))

sns.heatmap(df3.corr(),annot=True)

plt.show()
w = df2['Mental Health Issues'].value_counts()



plt.figure(figsize=(15,10))

sns.boxplot(df2['Race'], df2['Fatalities'])

plt.xlabel('Attackers Race')

plt.title('Count of Fatalities by Attacker', fontsize=14)

plt.show()
plt.figure(figsize=(15,8))

plt.plot(df2['Date'], df2['Injured'])

plt.plot(df2['Date'], df2['Fatalities'])

plt.title('Total by the Years', fontsize=24)

plt.ylabel('Number of Victims')

plt.xlabel('Year')

plt.show()
df2[df2['Location']=='Newtown, Connecticut']
plt.figure(figsize=(15,10))

b = df2.groupby('Mental Health Issues').sum()[['Fatalities', 'Injured', 'Total victims']]

sns.barplot(b.index, b.Fatalities)

plt.ylabel('Total Fatalities')

plt.title('Total Fatalities By Mental Health Illness', fontsize=20)

plt.show()
b
health = df2.groupby('Mental Health Issues').sum()['Fatalities']

health = pd.DataFrame(health)

health.reset_index(inplace=True)



plt.figure(figsize=(15,8))

plt.pie(health.Fatalities, labels=health['Mental Health Issues'], autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Percentage of Fatalities by Mental Health', fontsize=18)

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(df2['Mental Health Issues'], df2['Fatalities'])

plt.title('Total Fatalities by Mental Health Illness', fontsize=20)

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(df2['Mental Health Issues'], df['Total victims'])

plt.title('Total victims by Mental Health Illness', fontsize=18)

plt.show()
countState = df2['State'].value_counts()

countState = countState.head(10)

plt.figure(figsize=(15,8))

sns.barplot(countState.index, countState.values)

plt.xticks(rotation='vertical')

plt.title('Total Attacks by State', fontsize=18)

plt.show()
plt.figure(figsize=(15,8))

plt.pie(countState.values, labels=countState.index, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Attacks by top 10 States', fontsize=18)

plt.show()
ga = df2.groupby('State').sum()['Fatalities']

ga = pd.DataFrame(ga.sort_values(ascending=False).head(10))



ga.reset_index(inplace=True)



plt.figure(figsize=(15,8))

plt.pie(ga.Fatalities, labels=ga.State, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Percentage of Fatalitie by State', fontsize=18)

plt.show()
cityCount = df2['City'].value_counts()

cityCount = cityCount.head(10)

plt.figure(figsize=(15,8))

sns.barplot(cityCount.index, cityCount.values)

plt.xticks(rotation='vertical')

plt.title('Total Attacks by Cities', fontsize=18)

plt.show()
df2.head()
raceCounts = df2.groupby('Race').sum()[['Fatalities','Injured', 'Total victims']]
raceCounts.reset_index(inplace = True)
plt.figure(figsize=(15,8))

plt.pie(raceCounts.Fatalities, labels=raceCounts.Race, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Percentage of Fatalitie Cause by Race of Attacker', fontsize=18)

plt.show()
plt.figure(figsize=(15,8))

sns.barplot(raceCounts.Race, raceCounts.Fatalities)

plt.title('Fatalities by Attackers Race', fontsize=18)

plt.show()
n_groups = np.arange(len(raceCounts.Race))



labels = np.array(raceCounts.Race)



fig, ax = plt.subplots(figsize=(15,7))



bar_width = 0.34



opacity = 0.4



rects1 = ax.bar(n_groups, raceCounts.Fatalities, bar_width ,

                 alpha=opacity,

                 color='b',

                 label='Number of Fatalities')



#rects2 = plt.bar(n_groups + bar_width, w['Fatalities'], bar_width,

                 #alpha=opacity,

                 #color='r',

                 #label='Total Number of Fatalities')



rects2 = ax.bar(n_groups + bar_width, raceCounts.Injured, bar_width,

                 alpha=opacity,

                 color='r',

                 label='Total Number of Injured')



ax.set_xlabel('Race')

ax.set_ylabel('Count')

plt.xticks(n_groups,labels)

ax.set_title('Total numbers of Fatalities and Injured by Attackers Race', fontsize=18)



ax.legend()



plt.tight_layout()

plt.show()