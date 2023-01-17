import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import textacy
from textacy.preprocess import preprocess_text
import spacy

nlp = spacy.load('en')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",low_memory=False,#index_col="id",
                    parse_dates=["project_submitted_datetime"])
test= pd.read_csv("../input/test.csv",low_memory=False, #index_col="id",
                    parse_dates=["project_submitted_datetime"])
resources = pd.read_csv("../input/resources.csv")
resources.shape
train.columns
print(train.shape)
print(test.shape)
teachers_train = list(set(train.teacher_id.values))
teachers_test = list(set(test.teacher_id.values))
inter = set(teachers_train).intersection(teachers_test)
print("Number teachers train : %s, Number teachers test : %s, Overlap : %s "%(len(teachers_train), len(teachers_test), len(inter)))
print("Percent of test teachers in intersection with train : %f"%(100*len(inter)/len(teachers_test) ))
# add dum colyumn of target
test["project_is_approved"] = -1
df = pd.concat([train,test])
df.shape
# df[df.project_is_approved != -1].shape
#https://www.kaggle.com/mmi333/beat-the-benchmark-with-one-feature
resources['total_price'] = resources.quantity * resources.price

mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum()) 
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index
# all events in  2016-2017 , so post 2010 change in rules is Irrelevant!
df['project_submitted_datetime'].describe()

# df["after_change_date"] = df['project_submitted_datetime']>pd.to_datetime("18/02/2010")
# df["after_change_date"].describe()
# ## Add isi n USA federal holidays:
# ## https://stackoverflow.com/questions/29688899/pandas-checking-if-a-date-is-a-holiday-and-assigning-boolean-value

# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# cal = calendar()
# holidays = cal.holidays(start=df['project_submitted_datetime'].min(),end =  df['project_submitted_datetime'].max())
# df['fedHoliday'] = df['project_submitted_datetime'].isin(holidays)

# df['fedHoliday'] .describe()
# # All negative!  - Drop feature!
def create_features(df):
    df = pd.merge(df, mean_total_price, on='id')
    df = pd.merge(df, sum_total_price, on='id')
    df = pd.merge(df, count_total_price, on='id')
    df["dayOfYear"] = df['project_submitted_datetime'].dt.dayofyear # Day of Year
#     df["Weekday"] = df['project_submitted_datetime'].dt.weekday
    df["dayOfMonth"] = df['project_submitted_datetime'].dt.day
    df["teach_in_test"] = df["teacher_id"].isin(teachers_test)
#     df['year'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[0])
#     df['month'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[1])
#     for col in char_cols:
#         df[col] = df[col].fillna("NA")
#     df['text'] = df.apply(lambda x: " ".join(x[col] for col in char_cols), axis=1)
    return df

# train = create_features(train)
# test = create_features(test)

df = create_features(df)
df.rename(columns={"total_price_x":"mean_price_per_item","total_price":"count_items","total_price_y":"sum_price"},inplace=True)
df["teacher_max_proj_diff"] = df.groupby("teacher_id")["teacher_number_of_previously_posted_projects"].transform("max")
df["teacher_max_proj_diff"] = df["teacher_max_proj_diff"] - df["teacher_number_of_previously_posted_projects"]
df["teacher_count"] = df.groupby(['teacher_id'])["id"].transform("count")
# df["teacher_id"].value_counts()["484aaf11257089a66cfedc9461c6bd0a"] # counts match up!

df["state_count"] = df.groupby(['school_state'])["id"].transform("count")

df["state_count"].describe()
df["teach_sum_price_max"] = df.groupby("teacher_id")["sum_price"].transform("max")
df["teach_mean_price_median"] = df.groupby("teacher_id")["mean_price_per_item"].transform("median")
text_cols  = ['project_title', 'project_essay_1', 'project_essay_2','project_essay_3', 'project_essay_4', 'project_resource_summary']
df[text_cols] = df[text_cols].replace(r"\\n", " \\n ",regex=True)#.str.replace(r"\[a-z]", "  \ ",regex=True)
df[text_cols] = df[text_cols].replace(r"(\\[a-z])", r"  \1 ",regex=True)
# df[text_cols].head(3)
df["joint_essays"] = df.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    str(row['project_resource_summary']),
    str(row['project_title']),
#     str(row['description'])
]),
       axis=1)
df["joint_essays"].head(3)
# Clean text before feeding it to spaCy
# punctuations = string.punctuation

# # Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
# def cleanup_text(docs, logging=False):
#     texts = []
#     counter = 1
#     for doc in docs:
#         if counter % 1000 == 0 and logging:
#             print("Processed %d out of %d documents." % (counter, len(docs)))
#         counter += 1
#         doc = nlp(doc, disable=['parser', 'ner'])
#         tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
#         tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
#         tokens = ' '.join(tokens)
#         texts.append(tokens)
#     return pd.Series(texts)
## disabled in kernels due to timeout!

# df["joint_essays"] = df["joint_essays"].apply(lambda x: preprocess_text(x, fix_unicode=True, lowercase=True, transliterate=False,
#                                                                         no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=True,
#                                                                         no_punct=True, no_contractions=True
#                                                                        ,no_accents=True))
def lemma_text(doc):
    doc = nlp(doc, disable=['ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc  if tok.is_stop == False and tok.is_punct == False]
    tokens = ' '.join(tokens)
    return tokens
df["joint_essays"] = df["joint_essays"].apply(lemma_text)
df["joint_essays"].head()

print("old Train:", train.shape)
train = df[df.project_is_approved != -1]
print("Train:", train.shape)

print("old test:", train.shape)
test = df[df.project_is_approved == -1]
print("test:", test.shape)
train.to_csv("donors_train_aug-v1.csv.gz",index=False,compression="gzip")
test.drop("project_is_approved",axis=1).to_csv("donors_test_aug-v1.csv.gz",index=False,compression="gzip")