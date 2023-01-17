import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv("../input/train.tsv",sep="\t")

test = pd.read_csv("../input/test.tsv",sep="\t")
train
whole = pd.concat([train[train.columns[1:]],test[test.columns[1:]]],axis=0)

whole = whole[train.columns[1:]].reset_index(drop=True)
#cateory_name

not_error_ind = whole.category_name.dropna().index

whole.category_name.ix[not_error_ind] = whole.category_name.ix[not_error_ind].apply(lambda x:x.split('/'))



#カテゴリーを抽出&ID付与

first_category = [whole.category_name.ix[i][0] for i in not_error_ind]

first_dic = {list(set(first_category))[i]:i+1 for i in range(len(set(first_category)))}

whole["First_category_id"] = [first_dic[row[0]] if type(row) == list else np.nan for row in whole.category_name]

print("First-finish")



sec_category = [whole.category_name.ix[i][1] for i in not_error_ind]

sec_dic = {list(set(sec_category))[i]:i+1 for i in range(len(set(sec_category)))}

whole["Second_category_id"] = [sec_dic[row[1]] if type(row) == list else np.nan for row in whole.category_name]

print("Second-finish")



last_category = [whole.category_name.ix[i][-1] for i in not_error_ind]

last_dic = {list(set(last_category))[i]:i+1 for i in range(len(set(last_category)))}

whole["Last_category_id"] = [last_dic[row[-1]] if type(row) == list else np.nan for row in whole.category_name]

print("Last-finish")



whole = whole.drop("category_name",axis=1)
#brand_name

whole["Brand_name_01"] = [1 if type(row) !=float else 0 for row in whole.brand_name]

whole = whole.drop("brand_name",axis=1)
#item_description

whole = whole.drop("item_description",axis=1)
#name

whole = whole.drop("name",axis=1)
X = whole.ix[:1482534].dropna()

Z = whole.ix[1482535:].drop("price",axis=1).reset_index(drop=True)
for col in Z.columns:

    if len(Z[col].dropna()) != Z.shape[0]:

        Z[col] = Z[col].fillna(Z[col].median())
forest = RandomForestRegressor()
forest.fit(X.drop("price",axis=1),X.price)
result = forest.predict(Z)
submission_df = pd.DataFrame({"test_id":test.test_id,

                              "price":result})

submission_df = submission_df[["test_id","price"]]
submission_df.to_csv("mercari_bench_submission.csv",index=False)