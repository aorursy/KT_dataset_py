%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from datetime import datetime 

from mpl_toolkits.basemap import Basemap 
df = pd.read_csv('../input/attacks.csv', sep=',', encoding='ISO-8859-1') 



column_names = list(df.columns)

col_mapping = {'Sex ':'Sex', 'Fatal (Y/N)': 'Fatal', 'Species ' : 'Species'}

df = df.rename(columns=col_mapping, copy=False)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%d-%b-%Y')

df['Date'] = df['Date'].dropna()

df = df[df['Date'] > '1945-01-01']
df['Age'] = np.where(pd.to_numeric(df['Age'], 'coerce').notnull(), df['Age'], np.nan)

clean_age = df[['Age']].dropna()

clean_age = clean_age.astype(int)



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'lightblue'}, font_scale=0.9)

clean_age.plot(kind='hist', figsize=(5, 3), alpha=0.9, bins=25, color='mediumseagreen')

plt.xlabel('Age cathegory (years)')

plt.ylabel('Number of victims', fontsize=10)

plt.title('Victims repartition by age', fontsize=12, fontweight='bold')

plt.grid(color='k', axis='y', alpha=0.4, lw=0.8)

df.Fatal.value_counts()

plt.show()
def clean_g(x):

    if x == 'F'or x=='M':

        return x

    else:

        return np.nan



df['Sex'] = df['Sex'].apply(clean_g)

clean_gender = df['Sex'].dropna()

clean_gender = clean_gender.value_counts().tolist()



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'white'}, font_scale=0.8)



fig = plt.figure(figsize=(8, 6))



ax1 = fig.add_subplot(221)



ax1.pie(clean_gender, 

        labels=['Male', 'Female'], 

        shadow=True, 

        colors=('aquamarine', 'steelblue'), 

        explode=(0, 0.1), 

        startangle=45, 

        autopct='%1.1f%%')



ax1.set_title('Victims repartition by gender', fontsize=12, fontweight='bold')

ax1.axis('equal')

plt.tight_layout()

ax1.legend()



def clean_f(x):

    if x == 'Y'or x=='N':

        return x

    else:

        return np.nan



df['Fatal'] = df['Fatal'].apply(clean_f)

clean_fatal = df['Fatal'].dropna()

clean_fatal = clean_fatal.value_counts().tolist()



ax2 = fig.add_subplot(222)



ax2.pie(clean_fatal, 

        labels=['Fatal', 'Non fatal'], 

        shadow=True, 

        colors=('springgreen', 'deepskyblue'), 

        explode=(0, 0.1), 

        startangle=45, 

        autopct='%1.1f%%')



ax2.set_title('Cases repartition by result', fontsize=12, fontweight='bold')

ax2.axis('equal')

ax2.legend()

plt.tight_layout()

plt.show()
activity = df.Activity.value_counts().head(10)



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'lightsage'}, font_scale=0.9)

activity.plot(kind='bar', figsize=(6, 3), alpha=0.9, color='turquoise', rot=45)

plt.xlabel('Activities')

plt.ylabel('Number of cases', fontsize=10)

plt.title('Top 10 activities most exposed to a shark attack', fontsize=12, fontweight='bold')

plt.grid(color='k', axis='y', alpha=0.4, lw=0.8)

plt.show()
from_re = [r'.*?\bwhite\b\s+\bshark\b.*',r'.*?\bblue\b\s+\bshark\b.*', r'.*?\btiger\b\s.*',

           r'.*?\bbull\b\s.*',r'.*?\bshark\b\s+\binvolvement\b.*',r'.*?\bwobbegong\b\s+\bshark\b.*',

           r'.*?\bblacktip\b\s.*', r'.*?\bbronze\b\s+\bwhaler\b.*', r'.*?\bmako\b\s.*',r'.*?\bnurse\b\s.*',

           r'.*?\bhammerhead\b\s.*', r'.*?\braggedtooth\b\s.*']





to_re = ['White shark', 'Blue shark', 'Tiger shark', 

         'Bull shark', 'Not a shark', 'Wobbegong shark',

         'Blacktip shark','Bronze whaler shark', 'Mako shark', 

         'Nurse shark', 'Hammerhead shark', 'Raggedtooth shark']



df.Species = df.Species.str.lower().replace(from_re, to_re, regex=True)



top = df.Species.value_counts().head(7)



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'palegreen'})

top.plot(kind='bar', figsize=(6, 4), alpha=0.9, color='darkcyan', rot=45, fontsize=9)

plt.xlabel('Species')

plt.ylabel('Number of cases', fontsize=10)

plt.title('Top 7 most dangerous species of shark', fontsize=12, fontweight='bold')

plt.grid(color='k', axis='y', alpha=0.4, lw=0.8)





plt.tight_layout()

plt.show()
table = df.groupby('Date').size()

table = table.resample('A').sum()



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'lavender'})



table.plot(kind='line', figsize=(6, 4), alpha=1, color='darkslateblue', fontsize=10)



plt.ylabel('Number of cases', fontsize=10)

plt.title('Shark attacks repartition by year', fontsize=12, fontweight='bold')

plt.grid(color='k', axis='y', alpha=0.4, lw=0.8)



plt.tight_layout()

plt.show()
country = df['Country'].value_counts()

country = country.sort_values(ascending=False)

country = country.head(5)



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'powderblue'})

country.plot(kind='bar', figsize=(6, 4), alpha=1, color='mediumseagreen', rot=0, fontsize=9)

plt.xlabel('Countries')

plt.ylabel('Number of cases', fontsize=10)

plt.title('Sharks 5 most favorite countries', fontsize=12, fontweight='bold')

plt.grid(color='k', axis='y', alpha=0.4, lw=0.8)

plt.tight_layout()

plt.show()
df = df[df['Country'] == 'USA']



from_re = [r'.*?\bcocoa\s\b.*',r'.*?\bnew\b\s+\bsmyrna\b.*', r'.*?\bdaytona\b\s.*', 

           r'.*?\bponce\s\b.*', r'.*?\bmyrtle\s\b.*' ]

           

to_re = ['Cocoa Beach', 'New Smyrna Beach', 'Daytona Beach', 

         'Ponce Inlet Beach', 'Myrtle Beach']



df.Location = df.Location.str.lower().replace(from_re, to_re, regex=True)

beach = df.Location.value_counts()



sns.set(rc={'axes.facecolor':'w', 'figure.facecolor':'lightsteelblue'})

beach.head(5).plot(kind='barh', figsize=(6, 4), alpha=0.9, color='mediumpurple', rot=0, fontsize=9, width=0.8)

plt.xlabel('Countries')

plt.ylabel('Number of cases', fontsize=10)

plt.title('USA top 5 most dangerous beaches', fontsize=12, fontweight='bold')

plt.grid(color='k', axis='x', alpha=0.4, lw=0.8)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()