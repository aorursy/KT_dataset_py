import sys, csv

import pandas as pd

import numpy as np



import seaborn as sn

import matplotlib.pyplot as plt



import plotly.express as px



import plotly

from plotly import graph_objs as go, offline as po, tools



import math

import re

import copy

from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score

from correct import *

from pandas.testing import assert_frame_equal

biochem1 = pd.read_csv("../input/sequence/sequencing_data_biochem1.csv")

biochem2 = pd.read_csv("../input/sequence/sequencing_data_biochem2.csv")
bases = ["A", "T", "C", "G", "N"]
biochem1.head(10)
biochem1[biochem1["ref_1"] != biochem1["call_1"]]
len(biochem1[biochem1["ref_1"] != biochem1["call_1"]])
len(biochem1)
len(biochem1[biochem1["ref_1"] != biochem1["call_1"]]) / len(biochem1)
n_in_biochem1 = len(biochem1[biochem1["A_1"] + biochem1["G_1"] + biochem1["C_1"] + biochem1["T_1"] == 0]) + len(biochem1[biochem1["A_2"] + biochem1["G_2"] + biochem1["C_2"] + biochem1["T_2"] == 0])



print (f"There are {n_in_biochem1} N in biochem1")
n_in_biochem2 = len(biochem2[biochem2["A_1"] + biochem2["G_1"] + biochem2["C_1"] + biochem2["T_1"] == 0]) + len(biochem2[biochem2["A_2"] + biochem2["G_2"] + biochem2["C_2"] + biochem2["T_2"] == 0])



print (f"There are {n_in_biochem2} N in biochem2")
cycle1_temp = biochem1["ref_1"].value_counts().rename_axis('ref').reset_index(name='counts')

cycle2_temp = biochem1["ref_2"].value_counts().rename_axis('ref').reset_index(name='counts')



pd.merge(cycle1_temp, cycle2_temp, on=['ref']).set_index(['ref']).sum(axis=1)
cycle1_temp["relative_1"] = cycle1_temp["counts"] * 100 / len(biochem1)



fig = px.bar(cycle1_temp, x = "ref", y="relative_1", color="ref", category_orders={"ref": ["A", "T", "C", "G"]})

fig.show()

#plt.title('Biochem1 cycle 1 base distribution')
cycle2_temp["relative_2"] = cycle2_temp["counts"] * 100 / len(biochem1)



fig = px.bar(cycle2_temp, x = "ref", y="relative_2", color="ref", category_orders={"ref": ["A", "T", "C", "G"]})

fig.show()
b2_cycle1_temp = biochem2["ref_1"].value_counts().rename_axis('ref').reset_index(name='counts')

b2_cycle2_temp = biochem2["ref_2"].value_counts().rename_axis('ref').reset_index(name='counts')



pd.merge(b2_cycle1_temp, b2_cycle2_temp, on=['ref']).set_index(['ref']).sum(axis=1)
b2_cycle1_temp["relative_1"] = b2_cycle1_temp["counts"] * 100 / len(biochem2)



fig = px.bar(b2_cycle1_temp, x = "ref", y="relative_1", color="ref", category_orders={"ref": ["A", "T", "C", "G"]})

fig.show()

#plt.title('Biochem1 cycle 1 base distribution')
b2_cycle2_temp["relative_2"] = b2_cycle2_temp["counts"] * 100 / len(biochem2)



fig = px.bar(b2_cycle2_temp, x = "ref", y="relative_2", color="ref", category_orders={"ref": ["A", "T", "C", "G"]})

fig.show()

#plt.title('Biochem1 cycle 1 base distribution')
biochem2.head(10)
error_mapping = {}



for index, row in biochem2.iterrows():

     #print(index, row["ref_1"], row["call_1"])

    ref = row["ref_1"]

    call = row["call_1"]



    if ref not in error_mapping:

        error_mapping[ref] = {}



    if call not in error_mapping[ref]:

        error_mapping[ref][call] = 0

    

    error_mapping[ref][call] += 1
error_mapping
error_dict_to_array (error_mapping)
biochem2_error = error_dict_to_array (error_mapping)[0]

biochem2_error_row = error_dict_to_array (error_mapping)[1]

biochem2_error_column = error_dict_to_array (error_mapping)[2]



plot_confusion_matrix(biochem2_error, biochem2_error_row, bases, "call", "ref", "Biochem2 error confusion matrix")
### generate error correction matrix



ref_call_2 = {}

call_ref_2 = {}



for ref in error_mapping:

    max_mistake = 0

    max_mistake_base = ""

    for c in error_mapping[ref]:

        if error_mapping[ref][c] > max_mistake:

            max_mistake = error_mapping[ref][c]

            max_mistake_base = c

    

    ref_call_2[ref] = max_mistake_base

    call_ref_2[max_mistake_base] = ref
call_ref_2
rx = re.compile(r'([ATCG])_(\d)')

biochem2_rename_column = {}



for c in biochem2.columns:

    match = rx.match(c)

    if match:

        base = match.group(1)

        cycle = match.group(2)

        for call in call_ref_2:

            biochem2_rename_column[c] = call_ref_2[base] + "_" + cycle
### rename biochem2 column to correct the bug

my_biochem1 = biochem1.copy(deep=True)



my_biochem2 = biochem2.rename(columns = biochem2_rename_column)
### Sanity check: to check whether my implementation of basecalling is 100% the same with call_1 in Biochem1

temp_df = my_biochem1.copy(deep=True)



temp_df["my_call"] = temp_df[["A_1", "C_1", "G_1", "T_1"]].idxmax(axis=1)

temp_df.loc[temp_df["A_1"] +  temp_df["C_1"] + temp_df["G_1"] + temp_df["T_1"] == 0, "my_call"] = "N"



temp_df["my_call"] = temp_df["my_call"].str.replace('_1','')



temp_df[temp_df["my_call"] != temp_df["call_1"]]

my_biochem1["call_1"] = my_biochem1[["A_1", "C_1", "G_1", "T_1"]].idxmax(axis=1)

my_biochem1["call_2"] = my_biochem1[["A_2", "C_2", "G_2", "T_2"]].idxmax(axis=1)

my_biochem2["call_1"] = my_biochem2[["A_1", "C_1", "G_1", "T_1"]].idxmax(axis=1)

my_biochem2["call_2"] = my_biochem2[["A_2", "C_2", "G_2", "T_2"]].idxmax(axis=1)



my_biochem1.loc[my_biochem1["A_1"] +  my_biochem1["C_1"] + my_biochem1["G_1"] + my_biochem1["T_1"] == 0, "call_1"] = "N"



my_biochem1.loc[my_biochem1["A_2"] +  my_biochem1["C_2"] + my_biochem1["G_2"] + my_biochem1["T_2"] == 0, "call_2" ] = "N"



my_biochem2.loc[my_biochem2["A_1"] +  my_biochem2["C_1"] + my_biochem2["G_1"] + my_biochem2["T_1"] == 0, "call_1" ] = "N"



my_biochem2.loc[my_biochem2["A_2"] +  my_biochem2["C_2"] + my_biochem2["G_2"] + my_biochem2["T_2"] == 0, "call_2" ] = "N"
my_biochem1["call_1"] = my_biochem1["call_1"].str.replace('_1','')

my_biochem1["call_2"] = my_biochem1["call_2"].str.replace('_2','')



my_biochem2["call_1"] = my_biochem2["call_1"].str.replace('_1','')

my_biochem2["call_2"] = my_biochem2["call_2"].str.replace('_2','')
error_biochem1_cycle1 = len(my_biochem1[my_biochem1["ref_1"] != my_biochem1["call_1"]]) / len(my_biochem1)

error_biochem1_cycle2 = len(my_biochem1[my_biochem1["ref_2"] != my_biochem1["call_2"]]) / len(my_biochem1)



error_biochem2_cycle1 = len(my_biochem2[my_biochem2["ref_1"] != my_biochem2["call_1"]]) / len(my_biochem2)

error_biochem2_cycle2 = len(my_biochem2[my_biochem2["ref_2"] != my_biochem2["call_2"]]) / len(my_biochem2)







print (f"error_biochem1_cycle1: {error_biochem1_cycle1}, total: {len(my_biochem1)}")

print (f"error_biochem1_cycle2: {error_biochem1_cycle2}, total: {len(my_biochem1)}")

print (f"error_biochem2_cycle1: {error_biochem2_cycle1}, total: {len(my_biochem2)}")

print (f"error_biochem2_cycle2: {error_biochem2_cycle2}, total: {len(my_biochem2)}")





print (f"error_biochem1: {round(error_biochem1_cycle1 + error_biochem1_cycle2, 3)}, total: {len(my_biochem1) * 2}")



print (f"error_biochem2: {round(error_biochem2_cycle1 + error_biochem2_cycle2, 3)}, total: {len(my_biochem2) * 2}")

error_rows_b1_c1 = my_biochem1[my_biochem1["ref_1"] != my_biochem1["call_1"]][["ref_1", "call_1"]]

error_rows_b1_c2 = my_biochem1[my_biochem1["ref_2"] != my_biochem1["call_2"]][["ref_2", "call_2"]]



error_rows_b1_c1.rename(columns={"ref_1": "ref", "call_1": "call"}, inplace=True)

error_rows_b1_c2.rename(columns={"ref_2": "ref", "call_2": "call"}, inplace=True)



frames = [error_rows_b1_c1, error_rows_b1_c2] 



error_rows_b1 = pd.concat(frames)



error_rows_b2_c1 = my_biochem2[my_biochem2["ref_1"] != my_biochem2["call_1"]][["ref_1", "call_1"]]

error_rows_b2_c2 = my_biochem2[my_biochem2["ref_2"] != my_biochem2["call_2"]][["ref_2", "call_2"]]





error_rows_b2_c1.rename(columns={"ref_1": "ref", "call_1": "call"}, inplace=True)

error_rows_b2_c2.rename(columns={"ref_2": "ref", "call_2": "call"}, inplace=True)



frames = [error_rows_b2_c1, error_rows_b2_c2] 



error_rows_b2 = pd.concat(frames)



#error_rows_b2.rename(columns={"ref": "ref", "call_correct": "call"}, inplace=True)
error_rows_b1




cal_error_mapping(error_rows_b1, "ref", "call")
cal_error_mapping(error_rows_b2, "ref", "call")
cal_error_count(error_rows_b1, "ref", "call")
for base in ["A", "T", "C", "G"]:

    error = cal_error_count(error_rows_b1, "ref", "call").get(base)

    base_error_rate_value = cal_base_error_rate(my_biochem1, base, error)



    print (f"Biochem1, {base} error rate: {round(100 * base_error_rate_value, 2)}%")
for base in ["A", "T", "C", "G"]:

    error = cal_error_count(error_rows_b2, "ref", "call").get(base)

    base_error_rate_value = cal_base_error_rate(my_biochem2, base, error)



    print (f"Biochem2, {base} error rate: {round(100 * base_error_rate_value, 2)}%")
b1_error_array, b1_row, b1_column = error_dict_to_array (cal_error_mapping(error_rows_b1, "ref", "call"))



plot_confusion_matrix(b1_error_array, b1_row, b1_column, "call", "ref", "Biochem1 error confusion matrix")

cal_error_count(error_rows_b2, "ref", "call")


b2_error_array, b2_row, b2_column = error_dict_to_array (cal_error_mapping(error_rows_b2, "ref", "call"))



plot_confusion_matrix(b2_error_array, b2_row, b2_column, "call", "ref", "Biochem2 error confusion matrix")

pair_error_1 = my_biochem1[my_biochem1["ref_2"] != my_biochem1["call_2"]][["ref_1", "ref_2"]]





b1_error_array, b1_row, b1_column = error_dict_to_array (cal_error_mapping(pair_error_1, "ref_1", "ref_2"))



#b1_column.remove("N")



plot_confusion_matrix(b1_error_array, b1_row, b1_column, "2nd base", "1st base", "Biochem1 error pairs")
pair_error_2 = my_biochem2[my_biochem2["ref_2"] != my_biochem2["call_2"]][["ref_1", "ref_2"]]



b2_error_array, b2_row, b2_column = error_dict_to_array (cal_error_mapping(pair_error_2, "ref_1", "ref_2"))



#b2_column.remove("N")



plot_confusion_matrix(b2_error_array, b2_row, b2_column, "2nd base", "1st base", "Biochem2 error pairs")

len(my_biochem1[(my_biochem1["ref_2"] != my_biochem1["call_2"])  & (my_biochem1["ref_1"] == "G") & (my_biochem1["ref_2"] == "A")][["ref_1", "ref_2"]])
len(my_biochem1[(my_biochem1["ref_1"] == "G") & (my_biochem1["ref_2"] == "A")][["ref_1", "ref_2"]])
plot_ref_channel_intensity(my_biochem1, "ref_1", ["A_1", "C_1", "T_1", "G_1"], True)
plot_ref_channel_intensity(my_biochem1, "ref_2", ["A_2", "C_2", "T_2", "G_2"], True)
plot_ref_channel_intensity(my_biochem2, "ref_1", ["A_1", "C_1", "T_1", "G_1"], True)
plot_ref_channel_intensity(my_biochem2, "ref_2", ["A_2", "C_2", "T_2", "G_2"], True)
print (f'biochem1 cycle1, ref A, A_1 below 0.2, len: {len(my_biochem1[(my_biochem1["ref_1"] == "A") & (my_biochem1["A_1"] < 0.2)])}')



print (f'biochem1 cycle2, ref A, A_2 below 0.2, len: {len(my_biochem1[(my_biochem1["ref_2"] == "A") & (my_biochem1["A_2"] < 0.2)])}')





print (f'biochem2 cycle1, ref A, A_1 below 0.2, len: {len(my_biochem2[(my_biochem2["ref_1"] == "A") & (my_biochem2["A_1"] < 0.2)])}')



print (f'biochem2 cycle2, ref A, A_2 below 0.2, len: {len(my_biochem2[(my_biochem2["ref_2"] == "A") & (my_biochem2["A_2"] < 0.2)])}')
#show the outcomes of ref_1 == A & A_1 < 0.2 

my_biochem1[(my_biochem1["ref_1"] == "A") & (my_biochem1["A_1"] < 0.2)]["call_1"].value_counts()
#show the outcomes of ref_2 == A & A_2 < 0.2 



my_biochem1[(my_biochem1["ref_2"] == "A") & (my_biochem1["A_2"] < 0.2)]["call_2"].value_counts()
detect_weak_signal(my_biochem1)
detect_weak_signal(my_biochem2)
# Biochem1: how many signals are weaker than 0.2



my_biochem1_c1_x = my_biochem1[["A_1", "C_1", "G_1", "T_1"]].rename(columns = {"A_1": "A", "C_1": "C", "G_1": "G", "T_1": "T"})





my_biochem1_c2_x = my_biochem1[["A_2", "C_2", "G_2", "T_2"]].rename(columns = {"A_2": "A", "C_2": "C", "G_2": "G", "T_2": "T"})





my_biochem1_x = pd.concat([my_biochem1_c1_x, my_biochem1_c2_x], axis = 0)



my_biochem1_x[my_biochem1_x.max(axis=1) < 0.2]
# Biochem1: how many signals are weaker than 0.2



my_biochem2_c1_x = my_biochem2[["A_1", "C_1", "G_1", "T_1"]].rename(columns = {"A_1": "A", "C_1": "C", "G_1": "G", "T_1": "T"})





my_biochem2_c2_x = my_biochem2[["A_2", "C_2", "G_2", "T_2"]].rename(columns = {"A_2": "A", "C_2": "C", "G_2": "G", "T_2": "T"})





my_biochem2_x = pd.concat([my_biochem2_c1_x, my_biochem2_c2_x], axis = 0)



my_biochem2_x[my_biochem2_x.max(axis=1) < 0.2]
my_biochem1_c1_error_above_02 = len(my_biochem1[((my_biochem1["A_1"] > 0.2) | (my_biochem1["C_1"] > 0.2) | (my_biochem1["G_1"] > 0.2) | (my_biochem1["T_1"] > 0.2)) & (my_biochem1["ref_1"] != my_biochem1["call_1"])])



my_biochem1_c2_error_above_02 = len(my_biochem1[((my_biochem1["A_2"] > 0.2) | (my_biochem1["C_2"] > 0.2) | (my_biochem1["G_2"] > 0.2) | (my_biochem1["T_2"] > 0.2)) & (my_biochem1["ref_2"] != my_biochem1["call_2"])])



my_biochem1_error_above_02 = my_biochem1_c1_error_above_02 + my_biochem1_c2_error_above_02



my_biochem1_error = len(my_biochem1[my_biochem1["ref_1"] != my_biochem1["call_1"]]) + len(my_biochem1[my_biochem1["ref_2"] != my_biochem1["call_2"]])



print (f"For biochem1, the current errors is {my_biochem1_error}. But if we remove bases that have signals below or equal 0.2 in all 4 channels, the errors became {my_biochem1_error_above_02}")

my_biochem2_c1_error_above_02 = len(my_biochem2[((my_biochem2["A_1"] > 0.2) | (my_biochem2["C_1"] > 0.2) | (my_biochem2["G_1"] > 0.2) | (my_biochem2["T_1"] > 0.2)) & (my_biochem2["ref_1"] != my_biochem2["call_1"])])



my_biochem2_c2_error_above_02 = len(my_biochem2[((my_biochem2["A_2"] >= 0.2) | (my_biochem2["C_2"] > 0.2) | (my_biochem2["G_2"] > 0.2) | (my_biochem2["T_2"] > 0.2)) & (my_biochem2["ref_2"] != my_biochem2["call_2"])])



my_biochem2_error_above_02 = my_biochem2_c1_error_above_02 + my_biochem2_c2_error_above_02



my_biochem2_error = len(my_biochem2[my_biochem2["ref_1"] != my_biochem2["call_1"]]) + len(my_biochem2[my_biochem2["ref_2"] != my_biochem2["call_2"]])



print (f"For biochem2, the current errors is {my_biochem2_error}. But if we remove bases that have signals below or equal 0.2 in all 4 channels, the errors became {my_biochem2_error_above_02}")

my_biochem1[(my_biochem1["A_1"] < 0.2) & (my_biochem1["C_1"] < 0.2) & (my_biochem1["G_1"] < 0.2) & (my_biochem1["T_1"] < 0.2) & (my_biochem1["A_2"] < 0.2) & (my_biochem1["C_2"] < 0.2) & (my_biochem1["G_2"] < 0.2) & (my_biochem1["T_2"] < 0.2)]
my_biochem2[(my_biochem2["A_1"] < 0.2) & (my_biochem2["C_1"] < 0.2) & (my_biochem2["G_1"] < 0.2) & (my_biochem2["T_1"] < 0.2) & (my_biochem2["A_2"] < 0.2) & (my_biochem2["C_2"] < 0.2) & (my_biochem2["G_2"] < 0.2) & (my_biochem2["T_2"] < 0.2)]
assert_frame_equal(my_biochem1[((my_biochem1["A_1"] > 0.2) | (my_biochem1["C_1"] > 0.2) | (my_biochem1["G_1"] > 0.2) | (my_biochem1["T_1"] > 0.2)) & (my_biochem1["ref_1"] == my_biochem1["call_1"])], my_biochem1[(my_biochem1["A_1"] > 0.2) | (my_biochem1["C_1"] > 0.2) | (my_biochem1["G_1"] > 0.2) | (my_biochem1["T_1"] > 0.2)])

assert_frame_equal(my_biochem1[((my_biochem1["A_2"] > 0.2) | (my_biochem1["C_2"] > 0.2) | (my_biochem1["G_2"] > 0.2) | (my_biochem1["T_2"] > 0.2)) & (my_biochem1["ref_2"] == my_biochem1["call_2"])], my_biochem1[(my_biochem1["A_2"] > 0.2) | (my_biochem1["C_2"] > 0.2) | (my_biochem1["G_2"] > 0.2) | (my_biochem1["T_2"] > 0.2)])
assert_frame_equal(my_biochem2[((my_biochem2["A_1"] > 0.2) | (my_biochem2["C_1"] > 0.2) | (my_biochem2["G_1"] > 0.2) | (my_biochem2["T_1"] > 0.2)) & (my_biochem2["ref_1"] == my_biochem2["call_1"])], my_biochem2[(my_biochem2["A_1"] > 0.2) | (my_biochem2["C_1"] > 0.2) | (my_biochem2["G_1"] > 0.2) | (my_biochem2["T_1"] > 0.2)])

assert_frame_equal(my_biochem2[((my_biochem2["A_2"] > 0.2) | (my_biochem2["C_2"] > 0.2) | (my_biochem2["G_2"] > 0.2) | (my_biochem2["T_2"] > 0.2)) & (my_biochem2["ref_2"] == my_biochem2["call_2"])], my_biochem2[(my_biochem2["A_2"] > 0.2) | (my_biochem2["C_2"] > 0.2) | (my_biochem2["G_2"] > 0.2) | (my_biochem2["T_2"] > 0.2)])

###remove all 0 lines



temp = my_biochem1[my_biochem1["A_1"] +my_biochem1["C_1"] + my_biochem1["G_1"] +my_biochem1["T_1"] > 0]



my_biochem1_c1_x = temp[["A_1", "C_1", "G_1", "T_1"]].rename(columns = {"A_1": "A", "C_1": "C", "G_1": "G", "T_1": "T"})



my_biochem1_c1_y = temp[["ref_1"]].rename(columns = {"ref_1": "ref"})



cycle_1_error = len(temp[temp["ref_1"] != temp["call_1"] ])

cycle_1_total = len(temp)





temp = my_biochem1[my_biochem1["A_2"] +my_biochem1["C_2"] + my_biochem1["G_2"] +my_biochem1["T_2"] > 0]





my_biochem1_c2_x = temp[["A_2", "C_2", "G_2", "T_2"]].rename(columns = {"A_2": "A", "C_2": "C", "G_2": "G", "T_2": "T"})



my_biochem1_c2_y = temp[["ref_2"]].rename(columns = {"ref_2": "ref"})



my_biochem1_x = pd.concat([my_biochem1_c1_x, my_biochem1_c2_x], axis = 0)

my_biochem1_y = pd.concat([my_biochem1_c1_y, my_biochem1_c2_y], axis = 0)



cycle_2_error = len(temp[temp["ref_2"] != temp["call_2"] ])

cycle_2_total = len(temp)
cycle_1_error + cycle_2_error
cycle_1_total + cycle_2_total
### original error rate:

(cycle_1_error + cycle_2_error) / (cycle_1_total + cycle_2_total)
mlb = MultiLabelBinarizer(classes=["A", "C", "G", "T"])



y_onehot = mlb.fit_transform(my_biochem1_y["ref"])
kf = KFold(n_splits=4)



results = []



for fold, (train_index, test_index) in enumerate(kf.split(my_biochem1_x)):

    X_train, X_test = my_biochem1_x.iloc[train_index], my_biochem1_x.iloc[test_index]

    y_train, y_test = y_onehot[train_index], y_onehot[test_index]



    clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

    clf.fit(X_train, y_train)



    predictions = mlb.inverse_transform(clf.predict(X_test))

    y_test = mlb.inverse_transform(y_test)



    result = calculate_accuracy(predictions, y_test)

    results.append(result)

    print (f"fold {fold}: {result}")



    if fold == 3:

        df = X_test.copy(deep=True)



        p = [x[0] if len(x) > 0 else "N" for x in predictions]

        r = [x[0] if len(x) > 0 else "N" for x in y_test]



        df["pred"] = p

        df["ref"] = r

        print ("Errors in the last fold: ")

        print (df[df["pred"] != df["ref"]])



print (f"Average error rate: {sum(results)/len(results)}")
###remove all 0 lines



temp = my_biochem2[my_biochem2["A_1"] +my_biochem2["C_1"] + my_biochem2["G_1"] +my_biochem2["T_1"] > 0]



my_biochem2_c1_x = temp[["A_1", "C_1", "G_1", "T_1"]].rename(columns = {"A_1": "A", "C_1": "C", "G_1": "G", "T_1": "T"})



my_biochem2_c1_y = temp[["ref_1"]].rename(columns = {"ref_1": "ref"})



cycle_1_error = len(temp[temp["ref_1"] != temp["call_1"] ])

cycle_1_total = len(temp)





temp = my_biochem2[my_biochem2["A_2"] +my_biochem2["C_2"] + my_biochem2["G_2"] +my_biochem2["T_2"] > 0]





my_biochem2_c2_x = temp[["A_2", "C_2", "G_2", "T_2"]].rename(columns = {"A_2": "A", "C_2": "C", "G_2": "G", "T_2": "T"})



my_biochem2_c2_y = temp[["ref_2"]].rename(columns = {"ref_2": "ref"})



my_biochem2_x = pd.concat([my_biochem2_c1_x, my_biochem2_c2_x], axis = 0)

my_biochem2_y = pd.concat([my_biochem2_c1_y, my_biochem2_c2_y], axis = 0)



cycle_2_error = len(temp[temp["ref_2"] != temp["call_2"] ])

cycle_2_total = len(temp)
cycle_1_error + cycle_2_error
cycle_1_total + cycle_2_total
(cycle_1_error + cycle_2_error) / (cycle_1_total + cycle_2_total)
mlb = MultiLabelBinarizer(classes=["A", "C", "G", "T"])



y_onehot = mlb.fit_transform(my_biochem2_y["ref"])
kf = KFold(n_splits=4)



results = []



for fold, (train_index, test_index) in enumerate(kf.split(my_biochem1_x)):

    X_train, X_test = my_biochem2_x.iloc[train_index], my_biochem2_x.iloc[test_index]

    y_train, y_test = y_onehot[train_index], y_onehot[test_index]



    clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

    clf.fit(X_train, y_train)



    predictions = mlb.inverse_transform(clf.predict(X_test))

    y_test = mlb.inverse_transform(y_test)



    result = calculate_accuracy(predictions, y_test)

    results.append(result)

    print (f"fold {fold}: {result}")



    if fold == 3:

        df = X_test.copy(deep=True)



        p = [x[0] if len(x) > 0 else "N" for x in predictions]

        r = [x[0] if len(x) > 0 else "N" for x in y_test]



        df["pred"] = p

        df["ref"] = r

        print ("Errors in the last fold: ")

        print (df[df["pred"] != df["ref"]])



print (f"Average error rate: {sum(results)/len(results)}")