import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv("/kaggle/input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv")

df.head(5)
df.describe()
plt.figure(figsize=(15,10))
sns.heatmap(df.isnull())
plt.title("Missing values?", fontsize = 15)
plt.show()
nb_missing = df[df["lat"].isnull() & df["lng"].isnull()].shape[0]
print(f"Number of reviews with no latitude or longitude: {nb_missing}\nTotal number of reviews: {df.shape[0]}")
print("Number of hotels:",df['Hotel_Name'].nunique())
# Create a column with the rounded reviews
df["Reviewer_Score_Round"] = df["Reviewer_Score"].apply(lambda x: int(round(x)))

# Get the number of reviews with which scores
reviews_dist = df["Reviewer_Score_Round"].value_counts().sort_index()
bar = reviews_dist.plot.bar(figsize =(10,7))
plt.title("Distribution of reviews", fontsize = 18)
plt.axvline(df["Reviewer_Score"].mean()-2, 0 ,1, color = "grey", lw = 3)
plt.text(6, -15000, "average", fontsize = 14, color = "grey")
plt.ylabel("Count", fontsize = 18)
bar.tick_params(labelsize=16)

# Remove the column "Reviewer_Score_Round"
df.drop("Reviewer_Score_Round", axis = 1, inplace = True)
df_corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(df_corr, annot = True)
plt.title("Correlation between the variables", fontsize = 22)
plt.show()
# Get the colors for the graphic
colors = []
dim = df_corr.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,r,0.3))

# Transform each value in a positive value, because what interesses us
# isn't the direction of the correlation but the absolute correlation
df_corr["Reviewer_Score"].apply(lambda x: abs(x)).sort_values().plot.barh(color = colors)
plt.title("Correlation with Reviewer_Score", fontsize = 16)
plt.show()
# Group the data by nationality
group_nationality = df.pivot_table(values = "Reviewer_Score", 
                                   index = "Reviewer_Nationality", 
                                   aggfunc=["mean","count"])
group_nationality.columns = ["mean_review","review_count"]
# Keep only the nationalities with at least 3000 reviews given
reviews_count=group_nationality[group_nationality["review_count"]>3000]["review_count"].sort_values(ascending = False)

# Get the colors for the graphic
colors = []
dim = reviews_count.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,1-r,0.3))

# Display the result
reviews_count.plot.barh(figsize=(10,10), color = colors)
plt.title("Number of reviews given by nationality", fontsize = 18)
plt.ylabel("")
plt.show()
# Keep only the nationalities with at least 1000 reviews
group_nationality = group_nationality[group_nationality["review_count"] > 1000].sort_values(by = "mean_review", ascending = False)

# Get the colors for the graphic
colors = []
dim = group_nationality.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,1-r,0.3))

# Display the result
group_nationality["mean_review"].plot.barh(figsize = (10,20), color = colors)
plt.title("Who gives the worst review scores to hotels?", fontsize = 17)
plt.axvline(df["Reviewer_Score"].mean(), 0 ,1, color = "grey", lw = 3)
plt.text(8, 55, "average", fontsize = 14, c = "grey")
plt.text(8, -2, "average", fontsize = 14, c = "grey")
plt.xlabel("Average review score given", fontsize = 18)
plt.ylabel("")
plt.show()
# Convert the reviews to lower and delete leading/trailing space
df["Negative_Review"] = df["Negative_Review"].str.lower().str.strip()
df["Positive_Review"] = df["Positive_Review"].str.lower().str.strip()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()

rev1 = "The hotel was very good, I love it!"
rev2 = "It was just horrible, the worst ever."

print(f"review 1:\n{rev1}\nScore: {sent_analyzer.polarity_scores(rev1)}")

print(f"\nreview 2:\n{rev2}\nScore: {sent_analyzer.polarity_scores(rev2)}")
# Take only a part of the data to speed up
# df = df[:50000].copy()

start_time = time.time()
pos = df["Positive_Review"].apply(lambda x: abs(sent_analyzer.polarity_scores(x)["compound"]))
neg = df["Negative_Review"].apply(lambda x: -abs(sent_analyzer.polarity_scores(x)["compound"]))

df["sentiment_score"] = pos + neg
df["polarity_pos"] = pos
df["polarity_neg"] = neg

time_model = time.time() - start_time
print(f"Execution time: {int(time_model)} seconds")
df_corr = df.corr()

# Get the colors for the graphic
colors = []
dim = df_corr.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,r,0.3))

# Transform each value in a positive value, because what interesses us
# isn't the direction of the correlation but the absolute correlation
df_corr["Reviewer_Score"].apply(lambda x: abs(x)).sort_values().plot.barh(color = colors)
plt.title("Correlation with Reviewer_Score", fontsize = 16)
plt.show()
# Columns to use to train the models
# Only the columns with the highest correlation were chosen
cols = ['Review_Total_Negative_Word_Counts',
        'polarity_pos',
        'Average_Score',
        'Review_Total_Positive_Word_Counts']
        
X = df[cols].values
y = df["Reviewer_Score"].values

# Use StandardScaler to scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.metrics import mean_squared_error

def plot_res(y_test,pred, model = "LinearRegression"):
# Violinplots with the distribution of real scores and predicted scores

    MSRE = round((mean_squared_error(y_test,pred))**0.5,3)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
    
    sns.violinplot(y_test, ax = axes[0])
    axes[0].set_title("Distribution of\n scores")
    axes[0].set_xlim(0,11)
    
    sns.violinplot(pred, ax = axes[1])
    title = f"Predictions of scores with {model}\nMSRE:{MSRE}"
    axes[1].set_title(title)
    axes[1].set_xlim(0,11)
    plt.show()
    
# LinearRegression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plot_res(y_test,pred, model = "LinearRegression")

# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plot_res(y_test,pred, model = "GradientBoostingRegressor")

# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plot_res(y_test,pred, model = "RandomForestRegressor")