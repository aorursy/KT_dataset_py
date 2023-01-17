import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import Ridge, RidgeCV, Lasso

from sklearn import metrics

from sklearn.model_selection import cross_val_score, train_test_split

import xgboost as xgb







%config InlineBackend.figure_format = 'png'
train = pd.read_csv("../input/listings.csv")
columns_to_keep = ["price", "neighbourhood_cleansed", "bedrooms",

                   "property_type", "room_type", "name", "summary",

                   "amenities", "latitude", "longitude", "number_of_reviews",

                   "require_guest_phone_verification", "minimum_nights"]



train = train[columns_to_keep]
train.head(3)
def clean(train):



    train["bedrooms"] = train["bedrooms"].fillna(0.5) #these are studios

    train["summary"] = train["summary"].fillna("")

    train["bedrooms"] = train["bedrooms"].astype("str")



    #replace unpopular types with other 

    popular_types = train["property_type"].value_counts().head(6).index.values

    train.loc[~train.property_type.isin(popular_types), "property_type"] = "Other"



    #make price numeric:

    train["price"] = train["price"].str.replace("[$,]", "").astype("float")

    #eliminate crazy prices:

    train = train[train["price"] < 600]

    

    return train
train = clean(train)
train["price"].hist(bins = 30)

train["price"].std()
(train.pivot(columns = "bedrooms", values = "price")

         .plot.hist(bins = 30, stacked = True))
sns.barplot(x = "bedrooms", y = "price", data = train)
(train.pivot(columns = "room_type", values = "price")

         .plot.hist(bins = 30, stacked = False, alpha = 0.8))
train.groupby("room_type")["price"].mean()
y = train["price"]

train_num_cat = train[["neighbourhood_cleansed", "bedrooms",

                   "property_type", "room_type", "latitude", "longitude",

                   "number_of_reviews", "require_guest_phone_verification",

                    "minimum_nights"]]



train_text = train[["name", "summary", "amenities"]]
X_num = pd.get_dummies(train_num_cat)
train_text.head()
train.amenities = train.amenities.str.replace("[{}]", "")
amenity_ohe = train.amenities.str.get_dummies(sep = ",")
train.amenities = train.amenities.str.replace("[{}]", "")

amenity_ohe = train.amenities.str.get_dummies(sep = ",")
amenity_ohe.head(3)
train["text"] = train["name"].str.cat(train["summary"], sep = " ")
vect = CountVectorizer(stop_words = "english", min_df = 10)

X_text = vect.fit_transform(train["text"])
#metric:

def rmse(y_true, y_pred):

    return(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))



#evaluates rmse on a validation set:

def eval_model(model, X, y, state = 3):

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state = state)

    preds = model.fit(X_tr, y_tr).predict(X_val)

    return rmse(y_val, preds)
(X_num.shape, X_text.shape, amenity_ohe.shape)
#this is numeric + amenities:

X = np.hstack((X_num, amenity_ohe))



#this is all of them:

X_full = np.hstack((X_num, amenity_ohe, X_text.toarray()))
models_rmse = [eval_model(xgb.XGBRegressor(), X_num, y),

 eval_model(xgb.XGBRegressor(), X, y),

 eval_model(Ridge(), X_num, y),

 eval_model(Ridge(), X, y)]
models_rmse = pd.Series(models_rmse, index = ["xgb_num", "xgb_ame", "ridge", "ridge_ame"] )
models_rmse
models_rmse.plot(kind = "barh")
results = []

for i in range(30):

    X_tr, X_val, y_tr, y_val = train_test_split(X_num, y)

    y_baseline = [np.mean(y_tr)]*len(y_val)



    model = Ridge(alpha = 5)

    preds_logit = model.fit(X_tr, y_tr).predict(X_val)





    model = xgb.XGBRegressor()  

    preds_xgb = model.fit(X_tr, y_tr).predict(X_val)

    

    results.append((rmse(y_baseline, y_val),

                    rmse(preds_logit, y_val),

                    rmse(preds_xgb, y_val)

                    ))
results = pd.DataFrame(results, columns = ["baseline", "ridge", "xgb"])

results.plot.hist(bins = 15, alpha = 0.5)
pd.DataFrame([results.mean(), results.std()])