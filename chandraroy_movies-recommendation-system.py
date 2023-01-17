import pandas as pd
df = pd.read_csv("../input/movies_metadata.csv")
df.head(2)
df.columns
C = df["vote_average"].mean()
print(C)
m = df['vote_count'].quantile(.80)
print(m)
df_sub = df.copy().loc[df["vote_count"] >= 25.0]
df_sub.shape
df.shape
# Function to calculate new SCORE
def weighted_rating(x=df_sub, m = m, C=C):
    C = C
    m = m
    v = x["vote_count"]
    R = x["vote_average"]
    return (v/(v+m) * R) + (m/(m+v) * C)
df_sub["score"] = df_sub.apply(weighted_rating, axis = 1)
df_sub.head()
df_sub = df_sub.sort_values('score', ascending=False)
df_sub[['title', 'vote_count', 'vote_average', 'score']].head(10)
