# Import necessary libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# load the dataset from local storage

df=pd.read_csv('../input/flavors_of_cacao.csv')



# Understanding the basic ground information of our data

def all_about_my_data(df):

    print("Here is some Basic Ground Info about your Data:\n")

    

    # Shape of the dataframe

    print("Number of Instances:",df.shape[0])

    print("Number of Features:",df.shape[1])

    

    # Summary Stats

    print("\nSummary Stats:")

    print(df.describe())

    

    # Missing Value Inspection

    print("\nMissing Values:")

    print(df.isna().sum())



all_about_my_data(df)
df["Review Date"].head()
# Cleaning our feature names



cols = list(df.columns)



### Function to replace newline characters and spaces in the feature names

def rec_features(feature_names):

    rec_feat = []

    for f in feature_names:

        rec_feat.append(((f.casefold()).replace("\n","_")).replace(" ","_"))

    return rec_feat



print("Feature Names before Cleaning:")

print(cols)

print("\nFeature Names after Cleaning:")

print(rec_features(cols))

# Manual Removal



new_feature_names = rec_features(cols)

new_feature_names[0] = "company"



df=df.rename(columns=dict(zip(df.columns,new_feature_names)))

df.dtypes
df.head()
# Checking out if we have missing values

df.info()
df[['bean_type', 'broad_bean_origin']].head()
# What are these missing values in "bean_type" encoded as?



print(df['bean_type'].value_counts().head())

print("Missing Spaces encoded as:")

list(df['bean_type'][0:10])
# Replace the weird spaces with None (Symbolizes no data) 



def repl_space(x):

    if(x is "\xa0"):

        return "None"



# apply()        

df['bean_type'] = df['bean_type'].apply(repl_space)

df.head()
# Making that much needed conversion



df['cocoa_percent']=df['cocoa_percent'].str.replace('%','').astype(float)/100

df.head()
### Cocoa Percentage patterns over the years



d5 = df.groupby('review_date').aggregate({'cocoa_percent':'mean'})

d5 = d5.reset_index()



# Plotting

sns.set()

plt.figure(figsize=(15, 4))

ax = sns.lineplot(x='review_date', y='cocoa_percent', data=d5)

ax.set(xticks=d5.review_date.values)

plt.xlabel("\nDate of Review")

plt.ylabel("Average Cocoa Percentage")

plt.title("Cocoa Percentage patterns over the years \n")

plt.show()
### Rating patterns over the years



d6 = df.groupby('review_date').aggregate({'rating':'mean'})

d6 = d6.reset_index()



# Plotting

sns.set()

plt.figure(figsize=(15, 4))

ax = sns.lineplot(x='review_date', y='rating', data=d6)

ax.set(xticks=d6.review_date.values)

plt.xlabel("\nDate of Review")

plt.ylabel("Average Rating")

plt.title("Average Rating over the years \n")

plt.show()
### Top 5 companies in terms of chocolate bars in this dataset

d = df['company'].value_counts().sort_values(ascending=False).head(5)

d = pd.DataFrame(d)

d = d.reset_index() # dataframe with top 5 companies



# Plotting

sns.set()

plt.figure(figsize=(10,4))

sns.barplot(x='index', y='company', data=d)

plt.xlabel("\nChocolate Company")

plt.ylabel("Number of Bars")

plt.title("Top 5 Companies in terms of Chocolate Bars\n")

plt.show()
# Distribution of Chocolate Bars



sns.set()

plt.figure(figsize=(8,6))

sns.countplot(df['company'].value_counts().sort_values(ascending=False))

plt.xlabel("\nCount of chocolate bars")

plt.ylabel("Number of Companies")

plt.title("Distribution of Chocolate Bars")

plt.show()
### Top 5 companies in terms of average ratings

d2 = df.groupby('company').aggregate({'rating':'mean'})

d2 = d2.sort_values('rating', ascending=False).head(5)

d2 = d2.reset_index()



# Plotting

sns.set()

plt.figure(figsize=(20, 6))

sns.barplot(x='company', y='rating', data=d2)

plt.xlabel("\nChocolate Company")

plt.ylabel("Average Rating")

plt.title("Top 5 Companies in terms of Average Ratings \n")

plt.show()
### Top 5 companies in terms of average Cocoa Percentage

d2 = df.groupby('company').aggregate({'cocoa_percent':'mean'})

d2 = d2.sort_values('cocoa_percent', ascending=False).head(5)

d2 = d2.reset_index()



# Plotting

sns.set()

plt.figure(figsize=(15, 4))

sns.barplot(x='company', y='cocoa_percent', data=d2)

plt.xlabel("\nChocolate Company")

plt.ylabel("Average Cocoa Percentage")

plt.title("Top 5 Companies in terms of Average Cocoa Percentage \n")

plt.show()
### Average rating over the years (Top 5)



top5_dict = {}

for element in list(d['index']):

    temp = df[df['company']==element]

    top5_dict[element]=temp



top5_list = list(top5_dict.keys())



### Rating patterns over the years

d7 = df.groupby(['review_date', 'company']).aggregate({'rating':'mean'})

d7 = d7.reset_index()

d7 = d7[d7['company'].isin(top5_list)]



# Plotting

sns.set()

plt.figure(figsize=(15, 4))

ax = sns.lineplot(x='review_date', y='rating', hue="company", data=d7, palette="husl")

ax.set(xticks=d6.review_date.values)

plt.xlabel("\nDate of Review")

plt.ylabel("Average Rating")

plt.title("Average Rating over the years (Top 5 Producer Companies)\n")

plt.show()
### Preparing Soma for analysis



soma = df[df['company']=='Soma']
### Where does Soma get it's beans from ?



d3 = soma['broad_bean_origin'].value_counts().sort_values(ascending=False).head(5)

d3 = pd.DataFrame(d3)

d3 = d3.reset_index()

# Plotting

sns.set()

plt.figure(figsize=(10, 6))

sns.barplot(x='index', y='broad_bean_origin', data=d3)

plt.xlabel("\nBroad Bean Origin")

plt.ylabel("Number of Chocolate Bars")

plt.title("Where does Soma get it's beans from? \n")

plt.show()
### How are ratings of Chocolate bars by Soma ?



sns.kdeplot(soma['rating'], legend=False, color="brown", shade=True)

plt.xlabel("\nRating of the Chocolate Bar")

plt.ylabel("Proportion of Chocolate Bars")

plt.title("Ratings of Chocolate produced by Soma\n")

plt.show()
### Soma's performance over the years

d4 = soma.groupby('review_date').aggregate({'rating':'mean'})

d4 = d4.reset_index()

# Plotting

plt.figure(figsize=(10, 6))

sns.lineplot(x='review_date', y='rating', data=d4)

plt.xlabel("\nDate of Review")

plt.ylabel("Average Rating")

plt.title("Soma's Average Rating over the years\n")

plt.show()
### Soma's performance over the years

d4 = soma.groupby('review_date').aggregate({'cocoa_percent':'mean'})

d4 = d4.reset_index()

# Plotting

plt.figure(figsize=(10, 6))

sns.lineplot(x='review_date', y='cocoa_percent', data=d4)

plt.xlabel("\nDate of Review")

plt.ylabel("Percentage of Cocoa")

plt.title("Soma's Percentage of Cocoa over the years\n")

plt.show()
# Chocolate Bar levels



unsatisfactory = df[df['rating'] < 3.0]

satisfactory = df[(df['rating'] >= 3.0) & (df.rating < 4)]

pre_elite = df[df['rating'] >= 4.0]

label_names=['Unsatisfactory','Above Satisfactory (Excludes Premium and Elite)','Premium and Elite']

sizes = [unsatisfactory.shape[0],satisfactory.shape[0],pre_elite.shape[0]]

# Now let's make the donut plot

explode = (0.05,0.05,0.05)

my_circle=plt.Circle((0,0),0.7,color='white')

plt.figure(figsize=(7,7))

plt.pie(sizes,labels=label_names,explode=explode,autopct='%1.1f%%',pctdistance=0.85,startangle=90,shadow=True)

fig=plt.gcf()

fig.gca().add_artist(my_circle)

plt.axis('equal')

plt.tight_layout()

plt.show()
# The counts of each rating



r=list(df['rating'].value_counts())

rating=df['rating'].value_counts().index.tolist()

rat=dict(zip(rating,r))

for key,val in rat.items():

    print ('Rating:',key,'Reviews:',val)

plt.figure(figsize=(10,5))

sns.countplot(x='rating',data=df)

plt.xlabel('Rating of chocolate bar',size=12,color='blue')

plt.ylabel('Number of Chocolate bars',size=12,color='blue')

plt.show()
# Cocoa percent and choco bars



plt.figure(figsize=(10,5))

df['cocoa_percent'].value_counts().head(10).sort_index().plot.bar(color=['#d9d9d9','#b3b3b3','#808080','#000000','#404040','#d9d9d9','#b3b3b3','#404040','#b3b3b3'])

plt.xlabel('Percentage of Cocoa',size=12,color='black')

plt.ylabel('Number of Chocolate bars',size=12,color='black')

plt.show()
# Cocoa Percent and Rating



sns.lmplot(x='cocoa_percent',y='rating',fit_reg=False,scatter_kws={"color":"darkred","alpha":0.3,"s":100},data=df)

plt.xlabel('Percentage of Cocoa',size=12,color='darkred')

plt.ylabel('Expert Rating of the Bar',size=12,color='darkred')

plt.show()
#to get the indices

countries=df['broad_bean_origin'].value_counts().index.tolist()[:5]

# countries has the top 5 countries in terms of reviews

satisfactory={} # empty dictionary

for j in countries:

    c=0

    b=df[df['broad_bean_origin']==j]

    br=b[b['rating']>=3] # rating more than 4

    for i in br['rating']:

        c+=1

        satisfactory[j]=c    

# Code to visualize the countries that give best cocoa beans

print(satisfactory)

li=satisfactory.keys()

plt.figure(figsize=(10,5))

plt.bar(range(len(satisfactory)), satisfactory.values(), align='center',color=['#a22a2a','#511515','#e59a9a','#d04949','#a22a2a'])

plt.xticks(range(len(satisfactory)), list(li))

plt.xlabel('\nCountry')

plt.ylabel('Number of chocolate bars')

plt.title("Top 5 Broad origins of the Chocolate Beans with a Rating above 3.0\n")

plt.show()

#to get the indices

countries=df['broad_bean_origin'].value_counts().index.tolist()[:5]

# countries has the top 5 countries in terms of reviews

best_choc={} # empty dictionary

for j in countries:

    c=0

    b=df[df['broad_bean_origin']==j]

    br=b[b['rating']>=4] # rating more than 4

    for i in br['rating']:

        c+=1

        best_choc[j]=c    

# Code to visualize the countries that give best cocoa beans

print(best_choc)

li=best_choc.keys()

plt.figure(figsize=(10,5))

plt.bar(range(len(best_choc)), best_choc.values(), align='center',color=['#a22a2a','#511515','#a22a2a','#d04949','#e59a9a'])

plt.xticks(range(len(best_choc)), list(li))

plt.xlabel('Country')

plt.ylabel('Number of chocolate bars')

plt.title("Top 5 Broad origins of the Chocolate Beans with a Rating above 4.0\n")

plt.show()
df.columns
# Countries



print ('Top Chocolate Producing Countries in the World\n')

country=list(df['company_location'].value_counts().head(10).index)

choco_bars=list(df['company_location'].value_counts().head(10))

prod_ctry=dict(zip(country,choco_bars))

print(df['company_location'].value_counts().head())



plt.figure(figsize=(10,5))

plt.hlines(y=country,xmin=0,xmax=choco_bars,color='skyblue')

plt.plot(choco_bars,country,"o")

plt.xlabel('Country')

plt.ylabel('Number of chocolate bars')

plt.title("Top Chocolate Producing Countries in the World")

plt.show()
#reusing code written before

countries=country

best_choc={} # empty dictionary

for j in countries:

    c=0

    b=df[df['company_location']==j]

    br=b[b['rating']>=4] # rating more than 4

    for i in br['rating']:

        c+=1

        best_choc[j]=c    

# Code to visualize the countries that produce the best choclates

print(best_choc)

li=best_choc.keys()

# The lollipop plot

plt.hlines(y=li,xmin=0,xmax=best_choc.values(),color='darkgreen')

plt.plot(best_choc.values(),li,"o")

plt.xlabel('Country')

plt.ylabel('Number of chocolate bars')

plt.title("Top Chocolate Producing Countries in the World (Ratings above 4.0)")

plt.show()