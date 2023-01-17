import pandas as pd
train = pd.read_csv("../input/train.csv")
valid = pd.read_csv("../input/validation.csv")
test  = pd.read_csv("../input/test.csv")

whole = pd.concat([train, valid, test]).reset_index()
whole["_id"].head()
print("Number of rows:", len(whole))
print("Number of _ids:", len(whole["_id"].unique()))
whole.iloc[6279:, :].head()
is_faulty = whole["_canary"].str.isnumeric().fillna(False)

fix = pd.DataFrame({
    "_canary":     whole[is_faulty]["_started_at"],
    "_id":         whole[is_faulty]["_canary"],
    "_started_at": whole[is_faulty]["_id"]
})

whole.loc[is_faulty, ["_canary", "_id", "_started_at"]] = fix
whole.loc[is_faulty, ["_canary", "_id", "_started_at"]].head()
print("Number of rows: ", len(whole))
print("Number of _ids: ", len(whole["_id"].unique()))
was_faulty = is_faulty
del is_faulty
print("Number of unique      _id:", len(whole["_id"].unique()))
print("Number of unique _unit_id:", len(whole["_unit_id"].unique()))
print("Number of unique sentence:", len(whole["sentence"].unique()))
unique_unit_sentence_pairs = whole[["_unit_id", "sentence"]].drop_duplicates()
print("Number of unique unit_id-sentence pairs: ", len(unique_unit_sentence_pairs))
print("Number of unique unit_id:  ", len(unique_unit_sentence_pairs["_unit_id"].unique()))
print("Number of unique sentences:", len(unique_unit_sentence_pairs["sentence"].unique()))
print("Number of unique sent_id:", len(whole["sent_id"].unique()))
unique_unit_sent_pairs = whole[["_unit_id", "sent_id"]].drop_duplicates()
len(unique_unit_sent_pairs)
unique_sentid_sentence_pairs = whole[["sent_id", "sentence"]].drop_duplicates()

has_many_sent_id   = unique_sentid_sentence_pairs["sentence"].value_counts() > 1
has_many_sentences = unique_sentid_sentence_pairs["sent_id" ].value_counts() > 1

num_sentences_with_many_sent_id = has_many_sent_id.value_counts()[True]
num_sent_id_with_many_sentences = has_many_sentences.value_counts()[True]

print("Sentence can have many sent_id: ", num_sentences_with_many_sent_id > 0)
print("sent_id can have many sentences:", num_sent_id_with_many_sentences > 0)
def get_b1_and_b2(row):
    return (
        row["sentence"].find(row["term1"]) == row["b1"],
        row["sentence"].find(row["term2"]) == row["b2"],
    )

whole.apply(get_b1_and_b2, axis=1).all()
diff_values = pd.DataFrame(whole[["e1", "e2"]].values - whole[["b1", "b2"]].values)
diff_values.mean()
term_lengths = whole[["term1", "term2"]].applymap(len).rename(columns=lambda name: name + "_len")
term_diff_eq = (diff_values.as_matrix() + 1) == term_lengths.as_matrix()
term_diff_eq.all()
term_diff_eq = pd.DataFrame(term_diff_eq)
term_diff_eq.apply(pd.Series.value_counts)
e1_e2_error = pd.concat([
    diff_values,
    term_lengths,
    whole[["b1", "b2", "e1", "e2", "term1", "term2", "sentence"]]
], axis=1)[~term_diff_eq[1]]
e1_e2_error.head()
e1_e2_error.iloc[0]["sentence"]
e1_e2_error.iloc[0]["term2"]
len(e1_e2_error.iloc[0]["term2"])
e1_e2_error.iloc[0]["sentence"].find(e1_e2_error.iloc[0]["term2"])
pd.DataFrame({
    "letter": list("IM CEFTRIAXONE"),
    "index":  range(128, 142)
})
whole["relation"].value_counts()
whole["twrex"].value_counts()
unique_relation_twrex_pairs = whole[["relation", "twrex"]].drop_duplicates()
unique_relation_twrex_pairs.head()
print(len(whole[["_unit_id",    "twrex"]].drop_duplicates()))
print(len(whole[["_unit_id", "relation"]].drop_duplicates()))
whole[["term1", "term2", "relation", "direction"]].head()
whole.loc[whole["direction"] == "no_relation" ,["term1", "term2", "relation", "direction"]].head()
whole["direction_gold"].value_counts()
new_train = whole[           : len(train)]
new_valid = whole[len(train) : len(train) + len(valid)]
new_test  = whole[len(train) + len(valid) :]

new_train.to_csv("train.csv")
new_valid.to_csv("validation.csv")
new_test.to_csv( "test.csv")