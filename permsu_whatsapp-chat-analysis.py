# getting data into dataframe
import pandas as pd
df = pd.read_csv('/kaggle/input/whatsapp-chat/Whatsapp_chat.csv')
# dataframe information
df.info()
# changing column names
df.columns = ['id', 'hours', 'days', 'months', 'names', 'timestamp']
# dropping unwanted column id and rearranging columns as needed
df.drop('id', axis=1)
df = df[['names', 'timestamp', 'hours', 'days', 'months']]
df.head()
# checking how many peoples data is available
df['names'].unique()
# adding dummy column
df['dummy'] = pd.Series([1]*len(df))
# counting number of messages sent by each person
df.groupby('names').count()["dummy"]
# some Data visualizations using seaborn
import seaborn as sns
sns.set_style('dark')
sns.countplot(x="names", data=df, palette="viridis")

# looks like person1 was using whatsapp to it's full :)
# let's dig into a Person1's activity
sns.countplot(x="hours", data=df[df["names"] == "Person1"], palette="viridis")

# looks like Person1 was active during 11 PM to 2AM the most
# let's dig into a Person2's activity
sns.countplot(x="hours", data=df[df["names"] == "Person2"], palette="viridis")

# Person2 was active during 1PM the most
# let's dig into a Person3's activity
sns.countplot(x="hours", data=df[df["names"] == "Person3"], palette="viridis")

# Person3 was active during 12PM to 2AM
# let's dig into a Person4's activity
sns.countplot(x="hours", data=df[df["names"] == "Person4"], palette="viridis")

# Person4 is nearly same like Person3
df["months"].unique()

# only four months data is available
df.groupby("months").count()["dummy"]

# more chatting is done in July month
# now let's check data against months
sns.countplot(x="months", data=df, hue="names", palette="viridis")

# more chatting is done in July month and Person1 is active in all months :)
# let's check it against hours
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
sns.countplot(x="hours", data=df, hue="names", palette="viridis")

# person2 uses whatsapp very often i believe
# and at morning 6'o clock only Person3 was active. What was he doing ????????
