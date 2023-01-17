import pandas as pd
import catboost as cb
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = stopwords.words('russian')
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
selected_columns = ["item_id", "user_id", "region", "price", "item_seq_number", 
                    "user_type", "image_top_1", "category_name", "description", "title", "activation_date"]
label_column = "deal_probability"

train_labels = train_data[label_column]
train_data = train_data[selected_columns]
test_data = test_data[selected_columns]
def preprocess(df):
    df["price"].fillna(df["price"].mean(), inplace=True)
    df["image_top_1"].fillna(df["image_top_1"].mode()[0], inplace=True)
    df['description'].fillna(' ', inplace=True)
    df['title'].fillna(' ', inplace=True)
    
    return df
from sklearn.model_selection import train_test_split

train_data = preprocess(train_data)
test_data = preprocess(test_data)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, train_size=0.8, test_size=0.2)
def tfidf_vectorize(series, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words, min_df=1)
    return np.array(vectorizer.fit_transform(series).todense(), dtype=np.float16)


def price_feature(df, column_name):
    df.loc[:, column_name + "_mean_price"] = df.groupby(column_name)["price"].transform("mean")
    df.loc[:, column_name + "_max_price"] = df.groupby(column_name)["price"].transform("max")
    df.loc[:, column_name + "_min_price"] = df.groupby(column_name)["price"].transform("min")
    return df


def date_features(df):
    df.loc[:, "weekday"] = pd.to_datetime(df['activation_date']).dt.weekday
    df.loc[:, "month"] = pd.to_datetime(df['activation_date']).dt.month
    df.loc[:, "month_day"] = pd.to_datetime(df['activation_date']).dt.day
    df.drop(['activation_date'], inplace=True, axis=1)
    return df


def feature_engineering(df):
    description_vectors = tfidf_vectorize(df['description'], 100)
    title_vectors = tfidf_vectorize(df['title'], 200)

    for i in range(100):
        df.loc[:, 'description_tfidf_' + str(i)] = description_vectors[:, i]
    for i in range(200):
        df.loc[:, 'title_tfidf_' + str(i)] = title_vectors[:, i]
    
    df.drop(["description"], inplace=True, axis=1)
    
    df = price_feature(df, "category_name")
    df = price_feature(df, "user_id")
    df = price_feature(df, "region")
    df = date_features(df)
    
    df.loc[:, "title_len"] = df.title.apply(lambda x: len(x))
    df.drop(["title"], inplace=True, axis=1)
    return df
X_train_features = feature_engineering(X_train)
X_val_features = feature_engineering(X_val)
CAT_FEATURES=[0, 1, 2, 4, 5, 6, 7]
model = cb.CatBoostRegressor(iterations=200, 
                             learning_rate=0.05, 
                             depth=5, 
                             loss_function='RMSE', 
                             eval_metric='RMSE', 
                             random_seed=23, 
                             od_type='Iter', 
                             metric_period=50, 
                             od_wait=20)
valid_pool = cb.Pool(data=X_val_features, label=y_val, cat_features=CAT_FEATURES)
model.fit(X=X_train_features, y=y_train,  eval_set=valid_pool, cat_features=CAT_FEATURES)
test_data = feature_engineering(test_data)
preds = model.predict(test_data)
submission = pd.DataFrame(columns=["item_id", "deal_probability"])
submission["item_id"] = test_data["item_id"]
submission["deal_probability"] = preds
submission["deal_probability"].clip(0.0, 1.0, inplace=True)
submission.to_csv("submission.csv", index=False)