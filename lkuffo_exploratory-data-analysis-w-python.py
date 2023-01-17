import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import string
import html
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.stats import spearmanr, pearsonr
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # To see all the outputs in the notebook, not only the last one
pd.set_option('display.max_colwidth', -1) # To see all text in reviews

plt.style.use('ggplot')
# Reading data 
data = [
    pd.read_csv("../input/drugsComTest_raw.csv"),
    pd.read_csv("../input/drugsComTrain_raw.csv")
]

drugs = pd.concat(data, ignore_index=True) # Joining both train and test data

drugs.head(2) # Lets see what we have to work with...
# Let's delete all of these corrupted rows
print("Number of Corrupted Reviews: ", len(drugs[drugs.condition.str.contains(" users found this comment helpful.",na=False)]))
drugs = drugs[~drugs.condition.str.contains(" users found this comment helpful.",na=False)]
drugs.describe() # A basic descriptive analysis of the data 
# We will use this groupings later on...

print("Number of Reviews per Drug")
# Number of reviews per drug
reviews_per_drug = drugs.groupby(["drugName"]).agg({
    "uniqueID": pd.Series.nunique
})
reviews_per_drug.describe()


print("Number of Reviews per Condition")
# Number of reviews per condition
reviews_per_condition = drugs.groupby(["condition"]).agg({
    "uniqueID": pd.Series.nunique
})

reviews_per_condition.describe()
# Top 10 most reviewed drug names
plot = drugs.drugName.value_counts().nlargest(10).plot(kind='bar', title="Top 10 reviewed drugs", figsize=(12,6))
# Top 10 most suffered condition by reviewers
plot = drugs.condition.value_counts().nlargest(10).plot(kind='bar', title="Top 10 conditions in reviews", figsize=(12,6))
drugs_rating = drugs.groupby('drugName').agg({
    'rating': np.mean,
    'uniqueID': pd.Series.nunique
})

print("Significant number of reviews: More than", reviews_per_drug.quantile(q=0.75).values[0], "reviews")

# We only use the drugs which number of reviews is higher than a threshold
drugs_rating = drugs_rating[drugs_rating['uniqueID'] > int(reviews_per_drug.quantile(q=0.75))]

# Top 10
top_drugs_rating = drugs_rating.nlargest(20, 'rating')
plot = top_drugs_rating.plot(y='rating', kind='bar', figsize = (16, 3))
dummy = plt.title("Top 10 'significant' drugs with best rating") # Assigned to variable to prevent output
dummy = plt.ylim(9, 10) # Assigned to variable to prevent output

# Bottom 10
bottom_drugs_rating = drugs_rating.nsmallest(20, 'rating')
plot = bottom_drugs_rating.plot(y='rating', kind='bar', figsize = (16, 3))
dummy = plt.title("Top 10 'significant' drugs with worst rating") # Assigned to variable to prevent output
dummy = plt.ylim(1, 5) # Assigned to variable to prevent output
drugs_condition_rating = drugs.groupby(['drugName', 'condition']).agg({
    'rating': np.mean,
    'uniqueID': pd.Series.nunique
})

print("Number of pairs (Drug, Condition):", len(drugs_condition_rating))

print("Significant number of reviews: More than", drugs_condition_rating['uniqueID'].quantile(q=0.75), "reviews")

drugs_condition_rating = drugs_condition_rating[drugs_condition_rating['uniqueID'] > int(drugs_condition_rating['uniqueID'].quantile(q=0.75))]
# drugs_condition_rating.sort_values('rating', ascending=False)
top_drugs_condition_rating = drugs_condition_rating.nlargest(20, 'rating')
plot = top_drugs_condition_rating.plot(y='rating', kind='bar', figsize = (16, 3))
dummy = plt.title("Top 10 (Drug - Condition) with best rating") # Assigned to variable to prevent output
dummy = plt.ylim(9.5, 10) # Assigned to variable to prevent output

bottom_drugs_condition_rating = drugs_condition_rating.nsmallest(20, 'rating')
plot = bottom_drugs_condition_rating.plot(y='rating', kind='bar', figsize = (16, 3))
dummy = plt.title("Top 10 (Drug - Condition) with worst rating") # Assigned to variable to prevent output
dummy = plt.ylim(1, 4) # Assigned to variable to prevent output
drugs["date_format"] = drugs["date"].apply( lambda x: datetime.strptime(x, '%d-%b-%y')) # Get date as a date object
drugs["month"] = drugs["date_format"].apply(lambda x: x.strftime('%m')) # Extract date month
drugs["year"] = drugs["date_format"].apply(lambda x: x.strftime('%Y')) # Extract date year
drugs["weekday"] = drugs["date_format"].apply(lambda x: x.strftime('%w')) # Extract date weekday
start_date = drugs["date_format"].min()
end_date = drugs["date_format"].max()

print("First review date: ", start_date)
print("Last review date: ", end_date)
days_grouped = drugs.groupby(["year", "month"]) 
days_grouped = days_grouped.agg({
    'rating': np.mean,
    'usefulCount': np.sum,
    'uniqueID': pd.Series.nunique
})

different_months = len(days_grouped)

print("Months on dataset: ", different_months)
MME = MinMaxScaler() # Min-max normalization (0-1) for better visualization

grouped = days_grouped.reset_index(level=1)
index_values = np.unique(grouped.index.values)[1:] # First year (2008) month of January is missing

months = pd.DataFrame()

for year in index_values:
    months[year] = grouped.loc[year,:]["rating"].values # Every column is a year
months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dic"]

months.iloc[:,:] = MME.fit_transform(months) # Min Max Normalization by columns (year)

plots = months.plot(subplots=True, legend=True, figsize=(6,10), lw=2, title="Normalized (min-max) ratings average given in reviews for every month in every year")
for plot in plots:
    x = plot.set_ylim([-0.05, 1.05]) # Just assigning to variable so there is no output on notebook
x = plt.xticks(range(0, len(months_labels)), months_labels)
def encode_reviews(review):
    return html.unescape(review) # Decode in utf-8 and convert HTML characters

print("Review example: ", drugs.loc[0, 'review']) # Example of review text

drugs['review'] = drugs['review'].apply(encode_reviews) # Let's clean the text...

drugs.head(2) # We are ready to go
# This might take some time... (3-5 minutes)
sid = SentimentIntensityAnalyzer() 
drugs["sentiment"] = drugs["review"].apply(lambda x: sid.polarity_scores(x)["compound"]) # Compound is the overall sentiment score
# Let's see how Vader behaves with some reviews... (check text above)
drugs.loc[[215041,215046, 215050, 206473, 215035, 2, 23], ["review", "rating", "sentiment"]]
# Spearman correlation between computed sentiment and given rating
spearmanr(drugs['sentiment'], drugs['rating']) # Low-moderate correlation can be seen
# Let's find how Vader performs for the average rating in the most reviewed drugs... 

drugs_sentiment = drugs.groupby(["drugName"])
drugs_sentiment = drugs_sentiment.agg({
    'sentiment': np.mean, # drug sentiment average 
    'rating': np.mean,  # drug rating average
    'uniqueID': pd.Series.nunique
})
drugs_sentiment = drugs_sentiment[drugs_sentiment["uniqueID"] > reviews_per_drug.quantile(q=0.75)[0]]

plot = sns.jointplot(x="sentiment", y="rating", data=drugs_sentiment, kind="reg", height=8, scatter_kws={"s": 20})

print("Pearson correlation coefficient", pearsonr(drugs_sentiment['sentiment'], drugs_sentiment['rating']))
def generate_wordcloud(df, plot_title):
    
    stopwords_list = stopwords.words('english') + list(STOPWORDS) # We use both wordcloud and NLTK stopwords list
    
    raw_text = " ".join(df['review'].values)
    
    wc = WordCloud(stopwords=stopwords_list, background_color="white", width=1600, height=400).generate(raw_text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    plt.title(plot_title)
    plt.show()

reviews_by_comments = drugs.sort_values(by="usefulCount")

top_1000_useful_comments = reviews_by_comments.tail(1000)
top_100_useful_comments = reviews_by_comments.tail(100)
top_1000_unuseful_comments =reviews_by_comments.head(1000)
top_100_unuseful_comments =reviews_by_comments.head(100)


generate_wordcloud(drugs, "Wordcloud of all reviews")
generate_wordcloud(top_1000_useful_comments, "Wordcloud of Top 1000 most useful reviews")
generate_wordcloud(top_100_useful_comments, "Wordcloud of Top 100 most useful reviews")
generate_wordcloud(top_1000_unuseful_comments, "Wordcloud of Top 1000 most unuseful reviews")
generate_wordcloud(top_100_unuseful_comments, "Wordcloud of Top 100 most unuseful reviews")




