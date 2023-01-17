import pandas as pd
import seaborn as sns
df = pd.read_csv("../input/imdb-dataset-of-50k-movie-translated-urdu-reviews/urdu_imdb_dataset.csv")
df.head()
sns.countplot(df["sentiment"]);
df.info()
df.describe()
