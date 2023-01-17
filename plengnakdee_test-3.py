import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/swift-2.csv')

#df.head()



# drop columns with NaN or 0

data = df.drop(columns=['comment', 'comment_id', 'order_comment', 'local_code'])

data = data.loc[df['category'] == 'auto\n']

data.head()
# segment by demographic

gender = data.groupby('gender').count().reset_index()

gender
plt.figure(figsize=(10,6))

plt.title("Swift Market Segmentation, by Gender")

ax = sns.barplot(x=gender['gender'], y=gender['user_id'])

plt.xlabel("Gender")

plt.ylabel("Number of People")

ax.set_xticklabels(['Male', 'Female', 'Other'])

plt.show()
# segment by geographic

region = data.groupby('region').count().reset_index()

region
plt.figure(figsize=(10,6))

plt.title("Swift Market Segmentation, by Geographic")

sns.barplot(x=region['region'], y=region['user_id'])

plt.xlabel("Region")

plt.ylabel("Number of People")

plt.xticks(rotation=45)

plt.show()