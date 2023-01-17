import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import PowerTransformer, MinMaxScaler

#import lime

#from lime import lime_text

#from lime.lime_text import LimeTextExplainer
import string

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.tokenize import word_tokenize

nltk.download('punkt')
df = pd.read_csv("/Users/marievachelard/OneDrive - Capgemini/HACKATHON/HACKATHON_SUPAERO/data/train_set.csv")

df_test = pd.read_csv("/Users/marievachelard/OneDrive - Capgemini/HACKATHON/HACKATHON_SUPAERO/data/test_set.csv")
df.head(2)
df['bin_rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
df = df.dropna(subset=["review"]) # drop NaN values
# word / character count

df['number_words'] = df['review'].str.split().str.len()

df['number_character'] = df['review'].str.len()
# example of normalizing data

pt = PowerTransformer(method='yeo-johnson') # apply a power transform featurewise to make data more Gaussian-like

mms = MinMaxScaler() # transforms features by scaling each feature to a given range



df["useful_review_norm"] = pt.fit_transform(df["useful_review"].values.reshape(-1, 1))

df["useful_review_norm_mms"] = mms.fit_transform(df["useful_review_norm"].values.reshape(-1, 1))



df["funny_review_norm"] = pt.fit_transform(df["funny_review"].values.reshape(-1, 1))

df["funny_review_norm_mms"] = mms.fit_transform(df["funny_review_norm"].values.reshape(-1, 1))



df["cool_review_norm"] = pt.fit_transform(df["cool_review"].values.reshape(-1, 1))

df["cool_review_norm_mms"] = mms.fit_transform(df["cool_review_norm"].values.reshape(-1, 1))
numerical_col = [

"useful_review_norm_mms",

"funny_review_norm_mms",

"cool_review_norm_mms"

]



for col in numerical_col:

    plt.figure()

    sns.boxplot(x="bin_rating", y=col,data=df)

    plt.show()
def convert_text_to_lowercase(df, colname):

    df[colname] = df[colname].str.lower()

    return df

    



def text_cleaning(df, colname):

    """

    Takes in a string of text, then performs the following:

    1. convert text to lowercase

    2. ??

    """

    df = (

        df

        .pipe(convert_text_to_lowercase, colname)

    )

    return df
# clean the review column

df_cleaned = text_cleaning(df, 'review')
x_train, x_val, y_train, y_val = train_test_split(df_cleaned['review'], 

                                                    df_cleaned['bin_rating'], 

                                                    test_size=0.2, 

                                                    random_state=2018,

                                                    stratify=df_cleaned['bin_rating'])
Count_Vectorizer = CountVectorizer(

    analyzer='word',

    ngram_range=(1, 1),

    max_features=20000,

    max_df=1.0,

    min_df=10)



logit = LogisticRegression(solver='lbfgs', verbose=2, n_jobs=-1)



pipeline = Pipeline([

    ('vectorizer', Count_Vectorizer),

    ('model', logit)])



# fit model

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_val)
print('accuracy {}'.format(accuracy_score(y_pred, y_val)))

print('f1_score {}'.format(f1_score(y_val, y_pred)))

print(classification_report(y_val, y_pred))
# clean the review column

df_test_cleaned = text_cleaning(df_test, 'review')

x_test = df_test_cleaned['review']
# apply model to test dataset

predictions = pipeline.predict(x_test)

soumission = pd.DataFrame({"review_id": df_test['review_id'], "prediction": predictions})
soumission['prediction'] = soumission['prediction'].astype('bool')
soumission.head()
soumission.to_csv('/Users/marievachelard/OneDrive - Capgemini/HACKATHON/HACKATHON_SUPAERO/data/prediction_1.csv', index=False)