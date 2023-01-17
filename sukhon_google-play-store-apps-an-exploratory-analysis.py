import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import ttest_ind

import warnings

%matplotlib inline

warnings.filterwarnings("ignore")
# Import the googleplaystore.csv into a Pandas dataframe

app_data = pd.read_csv("../input/googleplaystore.csv")



# Show the first 3 rows of the dataframe

app_data.head(3)
# Header

print ("Missing Values"+"\n"+"-"*15)



# Print sum of null values per column

app_data.isnull().sum()
# Sort the original dataset by number of installs to see most popular apps first

app_data = app_data.sort_values(by="Installs", ascending=False)

app_data.head(4)
# Re-sort the data in ascending order to show least installed apps first

app_data = app_data.sort_values(by="Installs")

app_data.head()
# Get indexes of rows with NaN values for Rating column

nan_rows = list(app_data[app_data["Rating"].isna()].index)



# Add the index of the erroneous row

nan_rows.append(10472)



# Remove all rows with missing values

app_data = app_data.drop(nan_rows, axis=0)



# Remove unusable columns

app_data = app_data.drop(columns=["Size", "Current Ver","Android Ver"])



# Re-sort the data in descending order

app_data = app_data.sort_values(by="Installs", ascending=False)
# Generate a series containing count of apps with each bin of number of installs



# Plot each bin of installs with its frequency/occurrence/count in the dataset

plt.figure(figsize=(13,8))

app_data["Installs"].value_counts().plot(kind='bar')

plt.title("Count of Popular Apps in our Dataset")

plt.ylabel("Count")

plt.xlabel("Installs")

plt.show()
# Get Unique values of all bins in the Installs column

top_10 = np.unique(app_data["Installs"])



# Sort the values by length and return the longest 10 values

top_10 = (sorted(top_10,key=len,reverse=True))[:10]



print(top_10)

del top_10
# Remove "+" and "," from Installs column values & Put new values in a variable

installs = [np.int(i.replace("+", "").replace(",","")) for i in app_data["Installs"]]



# Replace the installs column with the new integer values

app_data["Installs"] = [i for i in installs]



del installs
# A new dataframe containing rows in top 10 bins

top_10_df = app_data[app_data["Installs"] >= 50000]



# How much does this data represent of the original data?

print (str(round(len(top_10_df)/len(app_data)*100,0))+"%")
# Parse review column values to integers

top_10_df["Reviews"] = [int(value) for value in top_10_df["Reviews"]]



# Add a new column containing review ratios

top_10_df["ReviewRatio"] = top_10_df["Reviews"]/top_10_df["Installs"]
# A dataframe to contain the most reviewed app from each bin:

most_reviewed = pd.DataFrame()



# Get the most reviewed app from each bin and add it to the most_reviewed dataframe

for bins in np.unique(top_10_df["Installs"]):

    top_row = top_10_df[top_10_df["Installs"] == bins]

    top_row = top_row.sort_values(by="ReviewRatio", ascending=False)

    top_row = top_row.head(1)

    most_reviewed = most_reviewed.append(top_row)



# Clear this dataframe of irrelevant columns for enhanced visibility

most_reviewed = most_reviewed.drop(columns=["Category","Type","Price","Content Rating","Genres","Last Updated"])

most_reviewed
# Sort by rating descending and get the top 3

highest_rated = most_reviewed.sort_values(by="Rating", ascending=False).head(3)

highest_rated
#Plot a histogram from the Rating column

top_10_df["Rating"].hist()

plt.title(" Google Play Apps Rating Distribution")

plt.ylabel("App Count")

plt.xlabel("Rating out of 5.0")

plt.show()
top_10_df["Rating"].describe()
len(top_10_df[top_10_df["Rating"] > 4.7])
top_10_df[top_10_df["Rating"] > 4.7].sort_values(by=["Rating","ReviewRatio"],ascending=False).head()
# Function to remove the $ prefix and parse price values to floats

def usd_2_float(value):

    if value == "0":

        return 0

    return float(value[1:])



# Call function on the Price column values

top_10_df["Price"] = [usd_2_float(value) for value in top_10_df["Price"]]
#First, the paid_apps variable will contain ALL apps, even free ones

#Plot a histogram of the Price column values

paid_apps = top_10_df["Price"]

paid_apps.hist()

plt.title("Pricing Distribution For Apps In Top 10 Bins")

plt.ylabel("App Count")

plt.xlabel("Price (U.S $)")

plt.show()
#Pick apps with a price higher than zero

paid_apps = top_10_df[top_10_df["Price"] > 0]



#Pick apps with a price higher than $50

paid_apps[paid_apps["Price"] > 50]
paid_apps[paid_apps["Price"]<40]["Price"].hist()

plt.title("Pricing Distribution For Apps Costing Less Than $40")

plt.ylabel("App Count")

plt.xlabel("Price (U.S $)")
print("$ {}".format(paid_apps["Price"].min()))
top_10_df[top_10_df["ReviewRatio"] > 1]
top_10_df["ReviewRatio"].hist()

plt.title("Review Ratio Distribution for Apps in Top 10 Bins")

plt.xlabel("Review Ratio")

plt.ylabel("App Count")

plt.show()
#Print percentiles at the borders of 1st, 2nd, and 3rd standard deviations, along with the max value

print("68th: {}".format(round(np.percentile(top_10_df["ReviewRatio"], 68),4)))

print("95th: {}".format(round(np.percentile(top_10_df["ReviewRatio"], 95),4)))

print("99th: {}".format(round(np.percentile(top_10_df["ReviewRatio"], 99),4)))

print("Max: {}".format(max(top_10_df["ReviewRatio"])))
plt.figure(figsize=(10, 5))



plt.subplot(1, 2, 1)

top_10_df["ReviewRatio"].hist()

plt.title("Review Ratio Distribution - Outliers Removed")

plt.xlabel("Review Ratio")

plt.ylabel("App Count")

plt.xlim(0,0.3)

plt.ylim(0,5800)



plt.subplot(1, 2, 2)

top_10_df["ReviewRatio"].hist()

plt.title("More reviewed groups further from the mean")

plt.xlabel("Review Ratio")

plt.ylabel("App Count")

plt.xlim(0.06,0.6)

plt.ylim(0,660)



plt.tight_layout()

plt.show()
top_10_df["Installs"].corr(top_10_df["Reviews"])
# Import the csv file

sentiment_data = pd.read_csv("../input/googleplaystore_user_reviews.csv")



# Drop null values

sentiment_data = sentiment_data.dropna()

sentiment_data.head()
# Take a slice of the original dataset containing app name and type, and merge it with matching apps in this dataset

sentiment_data = pd.merge(sentiment_data, app_data[["App","Type"]] , how='inner', on="App")



# Drop null values from sentiment data

sentiment_data = sentiment_data.dropna()



sentiment_data.head()
# Generate counts of each unique values and print them

(values,counts)= np.unique(sentiment_data["Type"], return_counts=True)



for index in range(len(values)):

    print("{}: {}".format(values[index],counts[index]))
# A dataframe for each app type

free_apps = sentiment_data[sentiment_data["Type"] == "Free"]

paid_apps = sentiment_data[sentiment_data["Type"] == "Paid"]
# Return normalized values (percentages) of each value's occurrence & display them

print("Free Apps - Sentiment Percentage \n"+"-"*30+"\n{}\n\n".format(free_apps['Sentiment'].value_counts(normalize=True) * 100))

print("Paid Apps - Sentiment Percentage \n"+"-"*30+"\n{}".format(paid_apps['Sentiment'].value_counts(normalize=True) * 100))
# Make random, consistent choice of rows from free apps

np.random.seed(777)



# Reset index, then drop the old index column when it is moved to the right as a new column

paid_apps = paid_apps.reset_index().drop(columns=["index"])

free_apps = free_apps.reset_index().drop(columns=["index"])



# Generate a list of random indexes applicable to free_apps

random_indexes = np.random.choice(len(free_apps)-1, len(paid_apps), replace=False)



# Shorten free_apps to the same size of paid_apps & using a random selection

free_apps = free_apps.iloc[random_indexes]



# Reset index of free_apps

free_apps = free_apps.reset_index().drop(columns=["index"])





len(free_apps)
# Generate values & counts for Sentiment columns in our dataframes & put them in dictionaries

val_ct_free = np.unique(free_apps['Sentiment'],return_counts=True)

free_data = {value: count for value, count in zip(val_ct_free[0],val_ct_free[1])}

val_ct_paid = np.unique(paid_apps['Sentiment'],return_counts=True)

paid_data = {value: count for value, count in zip(val_ct_paid[0],val_ct_paid[1])}





# Put values and counts each in a different variable for use in plots, taken from dictionaries

free_names = list(free_data.keys())

free_values = list(free_data.values())

paid_names = list(paid_data.keys())

paid_values = list(paid_data.values())





# Create a figure containing plots for each app type, sharing the y-axis for comparison

fig, axs = plt.subplots(1, 2, figsize=(10, 6),sharey=True)

axs[0].bar(free_names, free_values)

axs[0].set_title("Free App Sentiments")

axs[1].bar(paid_names, paid_values)

axs[1].set_title("Paid App Sentiments")

plt.show()



# Define variables that contain sentiment polarity for each app type

polarity_paid = paid_apps["Sentiment_Polarity"]

polarity_free = free_apps["Sentiment_Polarity"]



# Plot two histograms showing sentiment polarity of each app type

plt.figure(figsize=(14,5))

plt.hist(polarity_paid, normed=True, color="green",alpha=.7, label="Paid Apps") 

plt.hist(polarity_free, normed=True,color="red",alpha=.4, label="Free Apps")

plt.title('Sentiment Polarity Distribution For Free & Paid Apps')

plt.xlabel('Sentiment Polarity')

plt.legend(loc='upper right')

plt.show()
print("Mean polarity of free apps : {}".format(round(polarity_free.mean(),3)))

print("Mean polarity of paid apps : {}".format(round(polarity_paid.mean(),3)))

print("Polarity std. deviation of free apps : {}".format(round(polarity_free.std(),3)))

print("Polarity std. deviation of paid apps : {}".format(round(polarity_paid.std(),3)))
# Test similarity of samples with a t-value & a p-value

sample_comparison = list(ttest_ind(polarity_paid,polarity_free, equal_var=False))

sample_comparison = [round(value,3) for value in sample_comparison]

print("t-value: {}\np-value: {}".format(sample_comparison[0],sample_comparison[1]))
# Define variables that contain sentiment subjectivity for each app type

subjectivity_paid = paid_apps["Sentiment_Subjectivity"]

subjectivity_free = free_apps["Sentiment_Subjectivity"]



# Plot two histograms showing sentiment subjectivity of each app type

plt.figure(figsize=(14,5))

plt.hist(subjectivity_paid, normed=True, color="green",alpha=.7, label="Paid Apps") 

plt.hist(subjectivity_free, normed=True,color="red",alpha=.4, label="Free Apps")

plt.title('Sentiment Subjectivity Distribution For Free & Paid Apps')

plt.xlabel('Sentiment Subjectivity')

plt.legend(loc='upper right')

plt.show()
print("Mean subjectivity of free apps : {}".format(round(subjectivity_free.mean(),3)))

print("Mean subjectivity of paid apps : {}".format(round(subjectivity_paid.mean(),3)))

print("Subjectivity std. deviation of free apps : {}".format(round(subjectivity_free.std(),3)))

print("Subjectivity std. deviation of paid apps : {}".format(round(subjectivity_paid.std(),3)))
# Test similarity of samples with a t-value & a p-value

sample_comparison = list(ttest_ind(subjectivity_paid,subjectivity_free, equal_var=False))

sample_comparison = [round(value,2) for value in sample_comparison]

print("t-value: {}\np-value: {}".format(sample_comparison[0],sample_comparison[1]))