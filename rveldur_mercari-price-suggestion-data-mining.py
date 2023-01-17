#importing the reuired packages for data processing
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
#importing data to train and test data frame
train = pd.read_csv("../input/train.tsv",sep="\t")
test = pd.read_csv("../input/test.tsv",sep="\t")
NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000
full_df = pd.concat([train, test], 0)
nrow_train = train.shape[0]
y = np.log1p(train['price'])
del train
full_df["category_name"] = full_df["category_name"].fillna("Other").astype("category")
full_df["brand_name"] = full_df["brand_name"].fillna("unknown")

pop_brands = full_df["brand_name"].value_counts().index[:NUM_BRANDS]
full_df.loc[~full_df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

full_df["item_description"] = full_df["item_description"].fillna("None")
full_df["item_condition_id"] = full_df["item_condition_id"].astype("category")
full_df["brand_name"] = full_df["brand_name"].astype("category")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(full_df["name"])
unique_categories = pd.Series("/".join(full_df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(full_df["category_name"])
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(full_df["item_description"])
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(full_df["brand_name"])
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(full_df[["item_condition_id", "shipping"]], sparse = True).values)
X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         X_brand,
                         X_category,
                         X_name)).tocsr()
X_train = X[:nrow_train]
X_validate = X[nrow_train:]
X_train1,X_test,y_train,y_test = train_test_split(X_train, y, test_size=0.25, random_state=42)
gbm = GradientBoostingRegressor(loss='ls', learning_rate=0.15, n_estimators=255, subsample=1.0,min_samples_split=1.0, min_samples_leaf=1,max_depth=20)
model = gbm.fit(X_train1,y_train)
predict_test = model.predict(X_test)
rmsle_gbm = np.sqrt(np.square(np.log(predict_test + 1) - np.log(y_test + 1)).mean())
rmsle_gbm
predict_Validate = model.predict(X_validate)
submission = pd.read_csv("../input/sample_submission.csv")
submission.columns
submission['price'] = predict_Validate
submission.to_csv("GBM_Submission.csv")