import pandas as pd
class config:

    TRAIN_CSV = "../input/student-shopee-code-league-sentiment-analysis/train.csv"

    TEST_CSV = "../input/student-shopee-code-league-sentiment-analysis/test.csv"
df_train = pd.read_csv(config.TRAIN_CSV)

df_test = pd.read_csv(config.TEST_CSV)
review_set = set()
for row in df_train.itertuples():

    review_set.add(row.review)
count = 0

for row in df_test.itertuples():

    if row.review in review_set:

        print (row.review)

        count += 1
print("Percentage leakage",count/df_test.shape[0])