%matplotlib inline

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelBinarizer

import re

from matplotlib import pyplot as plt
inp_dataset = pd.read_csv("../input/imdbreviewsentiment/IMDB Dataset_V1.csv")

inp_dataset.head(2)
Cat_Count_df = inp_dataset.sentiment.value_counts().reset_index() 

Cat_Count_df["Percentage"] = Cat_Count_df["sentiment"]/Cat_Count_df["sentiment"].sum()*100

plt.bar(Cat_Count_df["index"], Cat_Count_df["sentiment"])

plt.xticks(Cat_Count_df["index"])

plt.xlabel("Sentiment")

plt.ylabel("Count")

plt.title("Distribution - Movie Sentiment")

plt.show()
def text_clean(text_series):

    text_series = text_series.str.lower()

    clean_2 = text_series.str.replace(r"<.*>|[^a-zA-Z\s]","")

    clean_3 = clean_2.str.replace(r"\s+", " ")

    return clean_3
inp_dataset["Text_Clean"] = text_clean(inp_dataset["review"])
train_x, test_x, train_y, test_y = train_test_split(inp_dataset["Text_Clean"], 

                                                    inp_dataset["sentiment"], 

                                                    test_size = 0.3, 

                                                    random_state = 8)
Cnt_Vec = CountVectorizer(stop_words="english")

BOW_train = Cnt_Vec.fit_transform(train_x).toarray()

BOW_train_Df = pd.DataFrame(BOW_train, columns=Cnt_Vec.get_feature_names())

BOW_train_Df[:4]
BOW_test = Cnt_Vec.transform(test_x).toarray()

BOW_test_Df = pd.DataFrame(BOW_test, columns=Cnt_Vec.get_feature_names())

BOW_test_Df[:4]
BOW_train_Df["Category_Values"] = train_y.reset_index()["sentiment"]

BOW_test_Df["Category_Values"] = test_y.reset_index()["sentiment"]

BOW_train_Df.head(5)
Cons_df = BOW_train_Df.groupby("Category_Values",as_index = False).sum().reset_index(drop=True)

Category_Count_df = BOW_train_Df["Category_Values"].value_counts().reset_index()

Category_Count_df.columns = ["Category_Values", "Category_Count"]

Cons_df = pd.merge(Cons_df,Category_Count_df,on="Category_Values",how = "left")

Cons_df["sum_all_words"] = Cons_df.drop(["Category_Count","Category_Values"],axis = 1).sum(axis = 1)
alpha = 1

prob_table = pd.DataFrame()

prob_table["Category_Values"] = Cons_df["Category_Values"]

prob_table["p_C"] = Cons_df["Category_Count"]/Cons_df["Category_Count"].sum()

cols = [col for col in Cons_df.columns if col not in ["Category_Values", "Category_Count", "sum_all_words"]]

no_of_cols = len(cols)

for col in cols:

    prob_table[col] = np.log((Cons_df[col]+alpha)/(Cons_df["sum_all_words"] + (alpha*no_of_cols)))

prob_table["p_C"] = np.log(prob_table["p_C"])
train_array = prob_table.drop(["Category_Values","p_C",],axis = 1)

train_array = np.array(train_array)

predict_df = pd.DataFrame(np.dot(BOW_test,train_array.T) + np.array(prob_table["p_C"]),columns=["negative", "positive"])

predict_df["final_category"] = predict_df.idxmax(axis = 1)

predict_df["Original_Cateogry"] = BOW_test_Df["Category_Values"]
pd.crosstab(predict_df["final_category"], predict_df["Original_Cateogry"] )
alpha = 1
Cnt_Vec = CountVectorizer(stop_words="english")

BOW_train = Cnt_Vec.fit_transform(train_x).toarray()

BOW_test = Cnt_Vec.transform(test_x).toarray()
lbl = LabelBinarizer()

train_Y = lbl.fit_transform(train_y)

if train_Y.shape[1] == 1:

    train_Y = np.concatenate([1 - train_Y, train_Y], axis=1)

cat_count_arr = np.log(np.sum(train_Y,axis = 0)/np.sum(train_Y))

classes = lbl.classes_
consolidated_train_df = np.dot(np.transpose(train_Y),BOW_train)
prob_table_numer = consolidated_train_df + alpha

prob_table_denom = np.sum(prob_table_numer,axis=1)

prob_table = np.log(prob_table_numer) - np.log(prob_table_denom.reshape(-1,1))
predict_arr = classes[np.argmax(np.dot(BOW_test,np.transpose(prob_table))+cat_count_arr,axis=1)]
pd.crosstab(predict_arr, test_y)
MNB_Model = MultinomialNB(alpha=1)

MNB_Model.fit(BOW_train, train_y)
prediction = MNB_Model.predict(BOW_test)
pd.crosstab(prediction, test_y)