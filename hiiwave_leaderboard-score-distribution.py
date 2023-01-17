import pandas as pd
import seaborn as sns
sns.set()
df = pd.read_csv("../input/leaderboard/order-brushing-shopee-code-league-publicleaderboard.csv")
sns.distplot(df.Score, bins=100, kde=False)
df.groupby('Score').TeamId.count().sort_values(ascending=False).head(20)