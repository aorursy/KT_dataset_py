import pandas as pd
import catboost as cb
import numpy as np

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
MAX_TFIDF_FEATURES = 75
stop_words = stopwords.words('russian')
rs = RussianStemmer()
train_data = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
y = train_data.deal_probability.copy()
selected_columns = ["item_id", "user_id", "region", "price", "item_seq_number", 
                    "user_type", "image_top_1", "category_name", "description",
                    "title", "activation_date", "param_1", "param_2", "param_3"]
label_column = "deal_probability"

train_labels = train_data[label_column]
train_data = train_data[selected_columns]
def preprocess(df):
    df["price"].fillna(train_data["price"].mean(), inplace=True)
    df["image_top_1"].fillna(train_data["image_top_1"].mode()[0], inplace=True)
    df['title'].fillna(' ', inplace=True)
    df['description'].fillna(' ', inplace=True)
    df['param_1'].fillna(' ', inplace=True)
    df['param_2'].fillna(' ', inplace=True)
    df['param_3'].fillna(' ', inplace=True)
    df["Weekday"] = df['activation_date'].dt.weekday
    df["Day of Month"] = df['activation_date'].dt.day
    # text preparation
    df["txt"] = df["title"] + " " + df["description"] + " " + df["param_2"] + " " + df["param_3"]
    # lower everything
    df["txt"] = df["txt"].str.lower() 
    # remove punctuation
    df["txt"] = df["txt"].str.replace('[^\w\s]',' ')
    # remove stopwords
    df["txt"] = df["txt"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    # stem
    #df["stem_txt"] =df["txt"][0:len(df)].apply(lambda x: " ".join([rs.stem(word) for word in x.split()]))
    df.drop(["activation_date", "title", "description", "param_2", "param_3"], axis = 1, inplace = True)
    return df
%%time
train_data = preprocess(train_data)
train_data.head()
def tfidf_vectorize(series, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    return np.array(vectorizer.fit_transform(series).todense(), dtype=np.float16)

def feature_engineering(df):
    txt_vectors = tfidf_vectorize(df['txt'], MAX_TFIDF_FEATURES)

    for i in range(MAX_TFIDF_FEATURES):
        df.loc[:, 'txt_tfidf_' + str(i)] = txt_vectors[:, i]
    df.drop("txt", axis = 1, inplace = True)
    return df
train_data = feature_engineering(train_data)
X = train_data
X.head()
test_data = pd.read_csv("../input/test.csv", parse_dates = ["activation_date"])
test_data = test_data[selected_columns]
test_data = preprocess(test_data)
test_data = feature_engineering(test_data)
test_data.head()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, random_state=167)
model = cb.CatBoostRegressor(iterations=400,
                             learning_rate=0.05,
                             depth=10,
                             #loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed = 167, 
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20) 
model.fit(X_train, y_train,
          eval_set=(X_valid,y_valid),
          use_best_model=True,
          cat_features=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
preds = model.predict(test_data)
submission = pd.DataFrame(columns=["item_id", "deal_probability"])
submission["item_id"] = test_data["item_id"]
submission["deal_probability"] = preds
submission["deal_probability"].clip(0.0, 1.0, inplace=True)
submission.to_csv("submission.csv", index=False)