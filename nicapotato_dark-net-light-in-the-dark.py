# General

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Regex

import re



# Viz

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Warnings OFF

import warnings

warnings.filterwarnings('ignore')



# Load

#df = pd.read_csv("Dark Net/Agora.csv", encoding="latin1")

df = pd.read_csv("../input/Agora.csv", encoding="latin1")

df.columns = [x.replace(" ", "") for x in df.columns] # Strip Column variable names from spaces

print("Data Load Complete")



# Change Dtype

df.Item = df.Item.astype(str)



"""

Regex Cleaning of Destination and Origin Column. Still very Messy and Deceptive.

#1 Remove non-words and strip the strings of outlying blank spaces, and

lowercase all characters, while capitalizing the first letter

#2 Remove instances of "only". Redundant information.

#3 Transform all the iterations of the term "worldwide"

#4&5 Transform all the iterations of united states and united kingdom



Regular Expression Explanation:

- [a-zA-Z]{2} # More than two letters

- |[/] # Or, the character " / "

"""



for x in ["Origin","Destination"]:

    df[x] = df[x].str.capitalize().str.replace('[^\w\s]','') # 1

    df[x] = df[x].str.replace(r"\bonly\b", '').str.strip() # 2 

    df.loc[df[x].str.contains(r"(?i)world|\b(?i)word\b|(?i)global",na=False),x] = "Worldwide" # 3

    df.loc[df[x].str.contains(r"(?i)kingdom",na=False),x] = "Uk" # 4

    df.loc[df[x].str.contains(r"(?i)(\bunited states\b)|\b(?i)us\b",na=False),x] = "Usa" # 5

print("Cleaning of 'Origin' and 'Destination' Complete")

##



# Remove BTC suffix

df["BTC"] = df['Price'].str.replace('BTC', '')



# Remove nan

df = df[pd.notnull(df['BTC'])]



##



"""

Dirty Data:

Here, it appears like non-price information made its way into the `BTC` variable columns. These are to be removed.

Diagnose for random junk in Bitcoin Price Column

Reference- https://stackoverflow.com/questions/10439666/regex-pattern-any-two-letters-followed-by-six-numbers

"""

regex1= r'[a-zA-Z]{2}|[/]'

# Delete Junk

df = df[~df['BTC'].str.contains(regex1)]

# To float.

df.BTC = pd.to_numeric(df.BTC)



##



# Convert to Dollar Value with 2014/2015 Average (399)

# Scaling, and Rounding

df["Value"] = (df.BTC * 399).round(3)

df["LogValue"]= np.log(df.Value).round(3)

print("Bitcoin/Value Variable Cleaned")



##

"""

Extract Sub-Categories and Create New Columns:

The Rating variable has includes is total score: e.i 4.5/5. Also plagued with "Number of Deals",

an unfortunate mash-up of two variables.

"""



# Regex Clean "Score" and extract Deals Count which found itself hiding within it

df["Score"]= df.Rating.str.split('/').str[0]

df["Deals"] = "NaN"

df["Deals"] = df.Score[df.Score.str.contains(r"(deal)", na=False)]

df.Score[df.Score.str.contains(r"(deal)", na=False)] = "NaN"

df.Score = df.Score.str.replace(r'(~)','')

df.Score = df.Score.astype(float)

df.Deals = df.Deals.str.extract('(\d+)')

print("Rating and Deal Variable Cleaned and Expanded")



"""

Parse Out Item Categories:

Currently in the form:  Drugs/Cannabis/Weed

Here, I split the categories up into seperate variables.

"""

df = pd.concat([df,df.Category.str.split('/', expand=True)], axis=1)

df = df.rename(columns={0: 'cat1', 1: 'cat2',2: 'cat3', 3: 'cat4'})

print("Item Category Variable Cleaned and Expanded")



"""

Bitcoin and Dollar Value Mix-Up:

Some products have their drug value unit incorrectly stated as Bitcoin, while they are infact in USD.

Discovered by noticing severe outliers in bitcoin value, aswell as cross-referencing the

Title and Item Description variable. See:

df.loc[[47625,9729,46707], ["Vendor","Item","ItemDescription","Price"]]



Solutions:

Find suspected non-BTC products from the intial "Price" column, and convert them directly to dollars, rather than the Bitcoin to Dollar transformation.



After manually reading the high-valued "Bitcoin" items, I decided that most of these are supposed to be dollars, with some outliers. My solution is to:



- Delete items with costing over 4000 bitcoin (supposibly ~1.7 million dollar), since these are not fault of the "dollars in the price column" problem.

- Convert items between 4000 and 1000 bitcoin to the value column without transformation, thus assuming they are already in dollars.

- Now, the new highest value possible is 399 thousand dollars, rather than 70 million dollars!



"""

# Remove products worth more than 4000 Bitcoin, since these are mostly probably in dollar units

df = df[df.BTC<4000]

df.Value[df.BTC > 1000] = df.BTC

print("Bitcoin and Dollar Value Unit Mix-Up partially delt with..")
def custom_describe(df):

    """

    I am a non-comformist :)

    """

    unique_count = []

    for x in df.columns:

        mode = df[x].mode().iloc[0]

        unique_count.append([x,

                             len(df[x].unique()),

                             df[x].isnull().sum(),

                             mode,

                             df[x][df[x]==mode].count(),

                             df[x].dtypes])

    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))

    return pd.DataFrame(unique_count, columns=["Column","Unique","Missing","Mode","Mode Occurence","dtype"]).set_index("Column").T



custom_describe(df)
# Countries

f, (ax1, ax2) = plt.subplots(1,2,figsize=(13,5))

target = "Origin"

bar_count1 = df.groupby([target])[['LogValue']].count().reset_index().sort_values(by='LogValue',ascending=False)[:15]

ax1 = sns.barplot(x="LogValue", y=target, data=bar_count1, ax=ax1,palette="viridis")

ax1.set_xlabel('Count')

ax1.set_title('Item Count by {}'.format(target))

ax1.set_ylabel('{}'.format(target))



target = "Destination"

bar_count2 = df.groupby([target])[['LogValue']].count().reset_index().sort_values(by='LogValue',ascending=False)[:15]

ax2 = sns.barplot(x="LogValue", y=target, data=bar_count2,ax=ax2,palette="viridis")

ax2.set_xlabel('Count')

ax2.set_title('Item Count by {}'.format(target))

ax2.set_ylabel('{}'.format(target))

plt.tight_layout(pad=0)

plt.show()
df.loc[df.Destination.str.contains(r"(?i)united|(?i)you|(?i)captivity",na=False),"Destination"].unique()[:10]
plt.subplots(figsize=(8,4))

ax = sns.kdeplot(df.Value, shade=True, color="b")

ax.set_xlabel('Log Axis: Dollar Value')

ax.set_title('Dollar Value of All Items')

ax.set_ylim(0,)

ax.set(xscale="log")

plt.show()
f, ax = plt.subplots(1,2,figsize=(12,4))

sns.kdeplot(df.Score, shade=True, color="g", ax=ax[0])

ax[0].set_xlabel('Rating out of Five')

ax[0].set_title('Distribution of Ratings')



sns.kdeplot(df.Deals, shade=True, color="r", ax=ax[1])

ax[1].set_xlabel('Number of Deals Boasted')

ax[1].set_title('Distribution of Deals Count')

plt.show()
n=30

f, ax = plt.subplots(figsize=[9,8])

sns.countplot(y=df.loc[df.Vendor.isin(df["Vendor"].value_counts().index[:n]),"Vendor"],order=df["Vendor"].value_counts().index[:n],palette="inferno",ax=ax)

ax.set_xlabel('Number of Listings by Vendors')

ax.set_title('Number of Listings')

plt.show()
print("Glance into the Group and Subgroup Structure:")

df.loc[:,["Category","cat1","cat2","cat3","cat4"]].sample(7)
# Count and Box for Category Level

def catplot(target, level):

    f, (ax1, ax2) = plt.subplots(1,2,figsize=(15,8), sharey=True)

    bar_count1 = df.groupby([target])[['LogValue']].count().sort_values(by='LogValue',ascending=False)

    ax1 = sns.barplot(x="LogValue", y=bar_count1.index, data=bar_count1, ax=ax1)

    ax1.set_xlabel('Count')

    ax1.set_title('Item Count by {}'.format(level))

    ax1.set_ylabel('{}'.format(level))



    # Box Plot

    ax2 = sns.boxplot(y=target, x="Value",order=bar_count1.index, data=df)

    ax2.set_xlabel('Log Dollar Value')

    ax2.set_title('Log Dollar Value for {}'.format(level))

    ax2.set_ylabel('')



    # Plot

    f.suptitle("Count and Box Plot for {}".format(level),fontsize=20)

    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    

# Simple Count Plot Helper Function

def countplot(level, label):

    plt.subplots(figsize=(8,7))

    bar_count= df.groupby([level])[['LogValue']].count().reset_index().sort_values(by='LogValue',ascending=False)

    ax = sns.barplot(x="LogValue", y=level, data=bar_count,palette="viridis")

    ax.set_xlabel('Count')

    ax.set_title('Item Count by {}'.format(label))

    ax.set_ylabel('General Categories')

    plt.show()

    

# Count and Box for Category Level

def catplot(target, level, size = (13,6)):

    f, (ax1, ax2) = plt.subplots(1,2,figsize=size, sharey=True)

    bar_count1 = df.groupby([target])[['Value']].count().sort_values(by='Value',ascending=False)

    ax1 = sns.barplot(x="Value", y=bar_count1.index, data=bar_count1, ax=ax1,palette="inferno")

    ax1.set_xlabel('Count')

    ax1.set_title('Item Count by {}'.format(level))

    ax1.set_ylabel('{}'.format(level))



    # Box Plot

    ax2 = sns.boxplot(y=target, x="Value",order=bar_count1.index, data=df,palette="inferno")

    ax2.set_xlabel('Log Axis: Dollar Value')

    ax2.set_title('Dollar Value for {}'.format(level))

    ax2.set_ylabel('')

    ax2.set_xscale('log')



    # Plot

    f.suptitle("Count and Box Plot for {}".format(level),fontsize=20)

    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    

# Simple Count Plot Helper Function

def countplot(level, label, size= (10,7)):

    plt.subplots(figsize=size)

    bar_count= df.groupby([level])[['Value']].count().reset_index().sort_values(by='Value',ascending=False)

    ax = sns.barplot(x="Value", y=level, data=bar_count,palette="inferno")

    ax.set_xlabel('Count')

    ax.set_title('Item Count by {}'.format(label))

    ax.set_ylabel('General Categories')

    plt.show()

print("Plot Helper Functions Ready")
catplot(target="cat1", level="Umbrella Categories", size = (10,5))

regex1= r'^((?!/).)*$'

print("Groups without Sub-Categories:",df['Category'][df['Category'].str.contains(regex1, na=True)].unique())
# Plot Funciton

def catinabox_plot(target, parentcat, kitten, level, size = (13,6)):

    f, (ax1, ax2) = plt.subplots(1,2,figsize=size, sharey=True)

    # Count Bar

    bar_count1= df[df[parentcat]==target].groupby([kitten])[['Value']].count().sort_values(by='Value',ascending=False)

    ax1 = sns.barplot(x="Value", y=bar_count1.index, data=bar_count1, ax=ax1,palette="inferno")

    ax1.set_xlabel('Count')

    ax1.set_title('Item Count by {}'.format(target))

    ax1.set_ylabel('{}'.format(target))

    

    # Box

    ax2 = sns.boxplot(y=kitten, x="Value", order=bar_count1.index, data=df[df[parentcat]==target],palette="inferno")

    ax2.set_xlabel('Log Axis: Dollar Value')

    ax2.set_title('Dollar Value for General Categories')

    ax2.set_xscale('log')

    ax2.set_ylabel('')

    

    # Plot

    f.suptitle("{} Categories for {}".format(level,target),fontsize=20)

    f.tight_layout(rect=[0, 0.03, 1, 0.95])
withsub2 = df.loc[df.cat2.notnull(),["cat1"]].reset_index()["cat1"].unique()

print("Unique Subcategory for Tier Two with SubClass:\n", withsub2)



print("\nTier Two Subcategories")

print(df.query("cat1 in @withsub2")["cat2"].unique())



countplot("cat2", "Tier Two Category", size=(10,7))
for x in withsub2:

    catinabox_plot(x, parentcat="cat1",kitten="cat2", level= "Second Level", size = (10,4))
cats = ["cat1","cat2","cat3", "cat4"]

withsub3 = df.loc[df.cat3.notnull(),["cat2"]].reset_index()["cat2"].unique()

print("Unique Subcategory for Tier Three with SubClass\n",  withsub3)



print("\nTier Three Subcategories")

print(df.query("cat2 in @withsub3")["cat3"].unique())



catplot("cat3", level="Tier Three Categories", size=(11,9))



for x in withsub3:

    catinabox_plot(x, parentcat="cat2",kitten="cat3", level="Tier Three Categories", size=(10,4))
withsub4 = df.loc[df.cat4.notnull(),["cat3"]].reset_index()["cat3"].unique()

print("Unique Subcategory for Tier Four with Subclass\n", withsub4)

print("\nTier Four Subcategories")

print(df.query("cat3 in @withsub4")["cat4"].unique())



catplot("cat4", level="Tier Four Categories", size=(10,4))
# Topic Modeling

from nltk import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

from wordcloud import WordCloud, STOPWORDS



size = (10,7)

# Function

def cloud(text, title):

    # Processing Text

    stopwords = set(STOPWORDS) # Redundant

    wordcloud = WordCloud(width=800, height=400,

                          #background_color='white',

                          #stopwords=stopwords,

                         ).generate(" ".join(text))

    

    # Output Visualization

    fig = plt.figure(figsize=size, dpi=80, facecolor='k',edgecolor='k')

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.axis('off')

    plt.title(title, fontsize=50,color='y')

    plt.tight_layout(pad=0)

    plt.show()



lemm = WordNetLemmatizer()

class LemmaCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(LemmaCountVectorizer, self).build_analyzer()

        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

    

# Define helper function to print top words

def print_top_words(model, feature_names, n_top_words):

    for index, topic in enumerate(model.components_):

        message = "Topic #{}: ".format(index)

        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])

        print(message)

    

def LDA(data, name=""):

    # Storing the entire training text in a list

    text = list(data.values)

    # Calling our overwritten Count vectorizer

    tf_vectorizer = LemmaCountVectorizer(max_df=0.95, min_df=2,

                                              stop_words='english',

                                              decode_error='ignore')

    tf = tf_vectorizer.fit_transform(text)





    lda = LatentDirichletAllocation(n_components=8, max_iter=5,

                                    learning_method = 'online',

                                    learning_offset = 50.,

                                    random_state = 0)



    lda.fit(tf)



    n_top_words = 10

    print("\n{} Topics in LDA model: ".format(name))

    tf_feature_names = tf_vectorizer.get_feature_names()

    print_top_words(lda, tf_feature_names, n_top_words)

    

# Output

pd.set_option('max_colwidth', 700)

pd.set_option('max_info_columns', 100)



def topic_view(df, catlevel, x):

    df = df[df["ItemDescription"].notnull()]

    print(LDA(df.ItemDescription[df[catlevel]==x], name=x))

    cloud(df['Item'][df[catlevel]==x].values,"Cloud for {}".format(x))



    return df.loc[df[catlevel]==x,["Item","ItemDescription","Value"]].sort_values(

        by='Value',ascending=False).sample(4)
topic_view(df=df, catlevel="cat1", x="Weapons")
topic_view(df=df, catlevel="cat2", x="Lethal firearms")
topic_view(df=df, catlevel="cat1", x="Data")
topic_view(df=df, catlevel="cat3", x="Relationships")
topic_view(df=df, catlevel="cat1", x="Drugs")
topic_view(df=df, catlevel="cat2", x="Opioids")
topic_view(df=df, catlevel="cat2", x="Cannabis")
topic_view(df=df, catlevel="cat2", x="Prescription")
topic_view(df=df, catlevel="cat3", x="Prescription")
df.cat2[df.cat1 == "Drugs"].unique()
topic_view(df=df, catlevel="cat4", x="UFOs")