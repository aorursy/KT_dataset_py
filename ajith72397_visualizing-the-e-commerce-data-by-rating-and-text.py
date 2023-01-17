
df = pd.read_csv('C:\\Users\\ajith\\Desktop\\Kaggle-Project\\Kaggle-women-data\\women.csv')
df.head()
df.info()
df.describe()
df['len_text'] = df['Review Text'].str.len()
#Data after the getting the length of the text.
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#plotting rating and length of the text.
#Here we can see the more it has length the more rating it can get.

g = sns.FacetGrid(df,col='Rating')
g.map(plt.hist,'len_text')
#Uisng boxplot

sns.boxplot(x='Rating',y='len_text',data=df)
#Uisng countplot counting the number of letters vs rating

sns.countplot(x='Rating',data=df)
Rating = df.groupby('Rating').mean()
Rating
#Getting correlation

Rating.corr()
sns.heatmap(Rating.corr(),cmap='coolwarm',annot=True)
