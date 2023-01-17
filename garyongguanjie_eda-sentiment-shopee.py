import pandas as pd

from nltk import word_tokenize

from tqdm.notebook import tqdm

from sklearn import model_selection
class config:

    TRAIN_CSV = "../input/student-shopee-code-league-sentiment-analysis/train.csv"

    TEST_CSV = "../input/student-shopee-code-league-sentiment-analysis/test.csv"
df_train = pd.read_csv(config.TRAIN_CSV)

df_test = pd.read_csv(config.TEST_CSV)
df_train.head()
df_test.head()
df_dup = df_train[df_train['review'].duplicated()]
print("num duplicates",df_dup.shape[0])
df_dup['checker'] = df_dup.apply(lambda x: str(x.review)+str(x.rating),axis=1)

print(df_dup['checker'].duplicated().sum(),'of duplicated reviews have the same rating')
df_train.drop_duplicates(subset='review',inplace=True)
count_df = df_train.groupby(['rating']).count()

count_df['percentage'] = 100 * count_df['review']  / count_df['review'].sum()

count_df
df_train['rating'].hist()
def count_len(text):

    return len(word_tokenize(text))
df_train['len'] = df_train['review'].apply(count_len)

df_test['len'] = df_test['review'].apply(count_len)
df_train['len'].hist()
df_test['len'].hist()
def len_stats(df):

    count_df = df.groupby(['len']).count()

    count_df['len_percentage'] = 100 * count_df['review']  / count_df['review'].sum()

    count_df['cumsum'] = count_df['len_percentage'].cumsum()

    return count_df
df_train_len = len_stats(df_train)
df_test_len = len_stats(df_test)
df_train_len
df_test_len
df_train_len[df_train_len['cumsum']<=99.9]
df_test_len[df_test_len['cumsum']<=99.9]