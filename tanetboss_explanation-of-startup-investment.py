import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

from wordcloud import WordCloud



warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.2f}'.format

## Function for providing summary in dataframe

%matplotlib inline



def funding_information(data,name):

    company = data[data['name'] == name]

    print ("Company : ", name)

    print ("Total Funding : ", company.funding_total_usd.values[0] , " $")

    print ("Seed Funding : ", company.seed.values[0] , " $")

    print ("Angle Funding :", company.angel.values[0] , " $")

    print ("Grant Funding : ",company.grant.values[0] , " $")

    print ("Product Crowd Funding : ",company.product_crowdfunding.values[0] , " $")

    print ("Equity Crowd Funding : ",company.equity_crowdfunding.values[0] , " $")

    print ("Undisclode Funding : ", company.undisclosed.values[0] , " $")

    print ("Convertible Note : ", company.convertible_note.values[0] , " $")

    print ("Debt Financing : ", company.debt_financing.values[0] , " $")

    print ("Private Equity : ",company.private_equity.values[0] , " $")

    print ("PostIPO Equity : ",company.post_ipo_equity.values[0] , " $")

    print ("PostIPO Debt : ",company.post_ipo_debt.values[0] , " $")

    print ("Secondary Market : ",company.secondary_market.values[0] , " $")

    print ("Venture Funding : ",company.venture.values[0] , " $")

    print ("Round A funding : ",company.round_A.values[0] , " $")

    print ("Round B funding : ",company.round_B.values[0] , " $")

    print ("Round C funding : ",company.round_C.values[0] , " $")

    print ("Round D funding : ",company.round_D.values[0] , " $")

    print ("Round E funding : ",company.round_E.values[0] , " $")

    print ("Round F funding : ",company.round_F.values[0] , " $")

    print ("Round G funding : ",company.round_G.values[0] , " $")

    print ("Round H funding : ",company.round_H.values[0] , " $")



def count_word(df, ref_col, liste):

    keyword_count = dict()

    for s in liste: keyword_count[s] = 0

    for liste_keywords in df[ref_col].str.split('|'):        

        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        

        for s in [s for s in liste_keywords if s in liste]: 

            if pd.notnull(s): keyword_count[s] += 1

    #______________________________________________________________________

    # convert the dictionary in a list to sort the keywords by frequency

    keyword_occurences = []

    for k,v in keyword_count.items():

        keyword_occurences.append([k,v])

    keyword_occurences.sort(key = lambda x:x[1], reverse = True)

    return keyword_occurences, keyword_count





def makeCloud(Dict,name,color):

    words = dict()



    for s in Dict:

        words[s[0]] = s[1]



        wordcloud = WordCloud(

                      width=1500,

                      height=750, 

                      background_color=color, 

                      max_words=50,

                      max_font_size=500, 

                      normalize_plurals=False)

        wordcloud.generate_from_frequencies(words)





    fig = plt.figure(figsize=(12, 8))

    plt.title(name)

    plt.imshow(wordcloud)

    plt.axis('off')



    plt.show()
data = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv',encoding = "ISO-8859-1")
len(data)
data.tail()
## select only data which name is not null



data = data[~data.name.isna()]
len(data)
print( data.columns.values )
data.rename(columns={' funding_total_usd ': "funding_total_usd",

                    ' market ': "market"},inplace=True)
data['status'].value_counts()
plt.rcParams['figure.figsize'] = 10,10

labels = data['status'].value_counts().index.tolist()

sizes = data['status'].value_counts().tolist()

explode = (0, 0, 0.2)

colors = ['#99ff99','#66b3ff','#ff9999']



plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',

        shadow=False, startangle=30)

plt.axis('equal')

plt.tight_layout()

plt.title("What is start up companies current status", fontdict=None, position= [0.48,1], size = 'x-large')

plt.show()
len(data['market'].unique())
data['market'].value_counts()[:5]
plt.rcParams['figure.figsize'] = 15,8



height = data['market'].value_counts()[:15].tolist()

bars =  data['market'].value_counts()[:15].index.tolist()

y_pos = np.arange(len(bars))

plt.bar(y_pos, height , width=0.7 ,color= ['c']+['paleturquoise']*14)

plt.xticks(y_pos, bars)

plt.xticks(rotation=90)

plt.title("Top 15 Start-Up market category", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()

set_keywords = set()

for liste_keywords in data['category_list'].str.split('|').values:

    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN

    set_keywords = set_keywords.union(liste_keywords)

#_________________________

# remove null chain entry

set_keywords.remove('')
keyword_occurences, dum = count_word(data, 'category_list', set_keywords)
makeCloud(keyword_occurences[0:15],"Keywords","White")
data['funding_total_usd'].head()
data['funding_total_usd'] = data['funding_total_usd'].str.replace(',', '')

data['funding_total_usd'] = data['funding_total_usd'].str.replace('-', '')

data['funding_total_usd'] = data['funding_total_usd'].str.replace(' ', '')



data['funding_total_usd'] = pd.to_numeric(data['funding_total_usd'], errors='coerce')
data['funding_total_usd'].head()
plt.rcParams['figure.figsize'] = 15,6

plt.hist(data['funding_total_usd'].dropna(), normed=False, bins=30)

plt.ylabel('Count')

plt.xlabel('Fnding (usd)')

plt.title("Distribution of total funding ", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()
Q1 = data['funding_total_usd'].quantile(0.25)

Q3 = data['funding_total_usd'].quantile(0.75)

IQR = Q3 - Q1



lower_bound = (Q1 - 1.5 * IQR)

upper_bound = (Q3 + 1.5 * IQR)

without_outlier = data[(data['funding_total_usd'] > lower_bound ) & (data['funding_total_usd'] < upper_bound)]
plt.rcParams['figure.figsize'] = 15,6

plt.hist(without_outlier['funding_total_usd'].dropna(), bins=30,color = 'paleturquoise' )



plt.ylabel('Count')

plt.xlabel('Funding (usd)')

plt.title("Distribution of total funding ", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()
Facebook_total_funding = data['funding_total_usd'][data['name']=="Facebook"].values[0]

Uber_total_funding = data['funding_total_usd'][data['name']=="Uber"].values[0]

Alibaba_total_funding = data['funding_total_usd'][data['name']=="Alibaba"].values[0]

Cloudera_total_funding = data['funding_total_usd'][data['name']=="Cloudera"].values[0]
plt.rcParams['figure.figsize'] = 15,6



plt.hist(data['funding_total_usd'][(data['funding_total_usd'] >= 1000000000)&(data['funding_total_usd'] <= 3000000000)].dropna(), bins=30,color = 'lightcyan' )

plt.ylabel('Count')

plt.xlabel('Funding (usd)')

plt.title("Where are the well-known companies ? ", fontdict=None, position= [0.48,1.05], size = 'x-large')



plt.axvline(Facebook_total_funding,color='royalblue',linestyle ="--")

plt.text(Facebook_total_funding+15000000, 2.6,"Facebook")



plt.axvline(Uber_total_funding,color='black',linestyle ="--")

plt.text(Uber_total_funding+10000000, 2.2,"Uber")



plt.axvline(Cloudera_total_funding,color='dodgerblue',linestyle ="--")

plt.text(Cloudera_total_funding+10000000, 1.9,"Cloudera")



plt.axvline(Alibaba_total_funding,color='orange',linestyle ="--")

plt.text(Alibaba_total_funding+10000000, 1.6,"Alibaba")

#plt.ticklabel_format(style='plain')







plt.show()
Verizon_total_funding = data['funding_total_usd'][data['name']=="Verizon Communications"].values[0]

Sberbank_total_funding = data['funding_total_usd'][data['name']=="Sberbank"].values[0]
plt.rcParams['figure.figsize'] = 15,6

plt.hist(data['funding_total_usd'][(data['funding_total_usd'] >= 1000000000)].dropna(), bins=30,color = 'lightcyan' )

plt.ylabel('Count')

plt.xlabel('Funding (usd)')

plt.title("Who get the highest funding ? ", fontdict=None, position= [0.48,1.05], size = 'x-large')



plt.axvline(Facebook_total_funding,color='royalblue',linestyle ="--")

plt.text(Facebook_total_funding+15000000, 11,"Facebook")



plt.axvline(Uber_total_funding,color='black',linestyle ="--")

plt.text(Uber_total_funding+10000000, 9,"Uber")



plt.axvline(Cloudera_total_funding,color='dodgerblue',linestyle ="--")

plt.text(Cloudera_total_funding+10000000, 7,"Cloudera")



plt.axvline(Alibaba_total_funding,color='orange',linestyle ="--")

plt.text(Alibaba_total_funding+10000000, 4,"Alibaba")



plt.axvline(Verizon_total_funding,color='red',linestyle ="--")

plt.text(Verizon_total_funding+100000000, 15,"Verizon Communications")



plt.axvline(Sberbank_total_funding,color='mediumseagreen',linestyle ="--")

plt.text(Sberbank_total_funding+100000000, 12,"Sberbank")





plt.show()
data['founded_at'].head()
data['founded_at'] = pd.to_datetime(data['founded_at'], errors = 'coerce' )
plt.rcParams['figure.figsize'] = 15,6

data['name'].groupby(data["founded_at"].dt.year).count().plot(kind="line")



plt.ylabel('Count')

plt.title("Founded distribution ", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()
Facebook_founded_year = data['founded_at'][data['name']=="Facebook"].dt.year.values[0]

Uber_founded_year  = data['founded_at'][data['name']=="Uber"].dt.year.values[0]

Alibaba_founded_year  = data['founded_at'][data['name']=="Alibaba"].dt.year.values[0]
Uber_founded_year
plt.rcParams['figure.figsize'] = 15,6

data['name'][data["founded_at"].dt.year >= 1990].groupby(data["founded_at"].dt.year).count().plot(kind="line")

plt.ylabel('Count')



plt.axvline(Facebook_founded_year,color='royalblue',linestyle ="--")

plt.text(Facebook_founded_year+0.15, 3000,"Facebook \n (2004)")



plt.axvline(Uber_founded_year,color='black',linestyle ="--")

plt.text(Uber_founded_year+0.15, 4000,"Uber \n(2009)")



plt.axvline(Alibaba_founded_year,color='orange',linestyle ="--")

plt.text(Alibaba_founded_year+0.15, 2000,"Alibaba \n(1999)")





plt.title("When the well-known company found ?", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()
len(data['country_code'].unique())
data['country_code'].value_counts()[:8]
data['count'] = 1

country_market = data[['count','country_code','market']].groupby(['country_code','market']).agg({'count': 'sum'})

# Change: groupby state_office and divide by sum

country_market_pct = country_market.groupby(level=0).apply(lambda x:

                                                 100 * x / float(x.sum()))

country_market_pct.reset_index(inplace = True)
USA_market_pct = country_market_pct[country_market_pct['country_code'] == "USA"]

USA_market_pct = USA_market_pct.sort_values('count',ascending = False)[0:10]
## USA

plt.rcParams['figure.figsize'] =10,10

labels = list(USA_market_pct['market'])+['Other...']

sizes = list(USA_market_pct['count'])+[100-USA_market_pct['count'].sum()]

explode = (0.18, 0.12, 0.09,0,0,0,0,0,0,0,0.01)

colors =  ['royalblue','mediumaquamarine','moccasin'] +['oldlace']*8



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=30)

plt.axis('equal')

plt.tight_layout()

plt.title("USA start up market", fontdict=None, position= [0.48,1.1], size = 'x-large')



plt.show()
THA_market_pct = country_market_pct[country_market_pct['country_code'] == "THA"]

THA_market_pct = THA_market_pct.sort_values('count',ascending = False)[0:10]
plt.rcParams['figure.figsize'] = 10,10

labels = list(THA_market_pct['market'])+['Other...']

sizes = list(THA_market_pct['count'])+[100-USA_market_pct['count'].sum()]

explode = (0.18, 0.12, 0.09,0,0,0,0,0,0,0,0.01)

colors =  ['royalblue','violet','gold'] +['oldlace']*8



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=30)

plt.axis('equal')

plt.tight_layout()

plt.title("THA start up market", fontdict=None, position= [0.48,1.1], size = 'x-large')

plt.show()
funding_information(data,"Dropbox")
funding_information(data,"Uber")
data[['name','seed']].head(5)
print("The average of seed funding stage is around ",data['seed'][data['seed'] != 0].mean(), "$")
data['get_funding_in_seed'] = data['seed'].map(lambda s :1  if s > 0 else 0)
## USA

plt.rcParams['figure.figsize'] =10,10

labels = ['No','Get funding']

sizes = data['get_funding_in_seed'].value_counts().tolist()

explode = (0, 0.1)

colors =  ['lightcoral','palegreen'] 



plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=190)

plt.axis('equal')

plt.tight_layout()

plt.title("How may company get funding in seed stage", fontdict=None, position= [0.48,1.1], size = 'x-large')



plt.show()
## Remove Outlier first 



Q1 = data['seed'][data['seed'] != 0].quantile(0.25)

Q3 = data['seed'][data['seed'] != 0].quantile(0.75)

IQR = Q3 - Q1



lower_bound = (Q1 - 1.5 * IQR)

upper_bound = (Q3 + 1.5 * IQR)

without_outlier = data[(data['seed'] > lower_bound ) & (data['seed'] < upper_bound)]
Facebook_seed_funding = data['seed'][data['name']=="Facebook"].values[0]

Uber_seed_funding   = data['seed'][data['name']=="Uber"].values[0]

Dropbox_seed_funding   = data['seed'][data['name']=="Dropbox"].values[0]
plt.rcParams['figure.figsize'] = 15,6

plt.hist(without_outlier['seed'][without_outlier['seed']!=0].dropna(), bins=50,color = 'cornsilk' )



plt.axvline(Facebook_seed_funding,color='royalblue',linestyle ="--")

plt.text(Facebook_seed_funding+0.15, 200,"Facebook \n ( 0$ )")



plt.axvline(Uber_seed_funding,color='black',linestyle ="--")

plt.text(Uber_seed_funding+0.15, 2000,"      Uber \n ( 200000$ )")



plt.axvline(Dropbox_seed_funding,color='violet',linestyle ="--")

plt.text(Dropbox_seed_funding+0.15, 1000,"  Dropbox \n( 15000$ )")



plt.ylabel('Count')

plt.xlabel('Funding (usd)')

plt.title("Distribution of Seed funding ", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()
print("The average of Angel funding is around ",data['angel'][data['angel'] != 0].mean(), "$")
data['get_funding_in_angel'] = data['angel'].map(lambda s :"Get funding"  if s > 0 else "Not get funding")
print("Only " , data['get_funding_in_angel'].value_counts().values[1], " companies has angel investor")

print("while " , data['get_funding_in_angel'].value_counts().values[0], " are not")

print("~",data['get_funding_in_angel'].value_counts().values[1]/(data['get_funding_in_angel'].value_counts().values[1]+data['get_funding_in_angel'].value_counts().values[0]) *100, "percent")
data['round_A'][data['round_A'] != 0].mean()
data['round_B'][data['round_B'] != 0].mean()
data['round_C'][data['round_C'] != 0].mean()
round_ = ['round_A','round_B','round_C','round_D','round_E','round_F','round_G','round_H']

amount = [data['round_A'][data['round_A'] != 0].mean(),

          data['round_B'][data['round_B'] != 0].mean(),

          data['round_C'][data['round_C'] != 0].mean(),

          data['round_D'][data['round_D'] != 0].mean(),

          data['round_E'][data['round_E'] != 0].mean(),

          data['round_F'][data['round_F'] != 0].mean(),

          data['round_G'][data['round_G'] != 0].mean(),

         data['round_H'][data['round_H'] != 0].mean()]

          
plt.rcParams['figure.figsize'] = 15,8



height = amount

bars =  round_

y_pos = np.arange(len(bars))

plt.bar(y_pos, height , width=0.7, color= ['cornsilk','oldlace','papayawhip','wheat','moccasin','navajowhite','burlywood','goldenrod'] )

plt.xticks(y_pos, bars)

plt.title("Average investment in each round", fontdict=None, position= [0.48,1.05], size = 'x-large')

plt.show()