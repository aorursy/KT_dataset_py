import pandas as pd 
# df_sample_submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

df_test = pd.read_csv("../input/cat-in-the-dat/test.csv")

df_train = pd.read_csv("../input/cat-in-the-dat/train.csv")
df_train.head()
df_test.head()
print("The training set has {count_train} samples and the test set has {count_test} samples.".format(count_train=len(df_train), count_test=len(df_test)))
unique_vals = []

for col in df_train.columns:

    unique_vals.append([col, df_train[col].nunique()])

unique_vals
unique_test_vals = []

for col in df_test.columns:

    unique_test_vals.append([col, df_test[col].nunique()])

unique_test_vals
same_unique_vals = []

for col in df_test.columns:

    same_unique_vals.append([col, set(df_train[col].value_counts().index.tolist()) == set(df_test[col].value_counts().index.tolist())])

same_unique_vals
diff_cols = ['nom_7', 'nom_8', 'nom_9']

vals_diff = []

for col in diff_cols:

    vals_diff.append([col, 

                      set(df_train[col].value_counts().index.tolist()) - set(df_test[col].value_counts().index.tolist()), 

                      set(df_test[col].value_counts().index.tolist()) - set(df_train[col].value_counts().index.tolist())])

vals_diff
diff_sizes = []

for val_diff in vals_diff:

    diff_sizes.append([val_diff[0], len(val_diff[1]), len(val_diff[2])])



for var in diff_sizes:

    print("The number of values in {col} that occur in the training set but not the test set is {count}".format(col=var[0], count=var[1]))

    print("The number of values in {col} that occur in the test set but not the training set is {count}".format(col=var[0], count=var[2]))

len(df_test.loc[df_test['nom_7'].isin(vals_diff[0][2])]['nom_7'])



perc_diff = []

for var in vals_diff:

    test_perc = len(df_test.loc[df_test[var[0]].isin(var[2])])/len(df_test)

    train_perc = len(df_train.loc[df_train[var[0]].isin(var[1])])/len(df_train)

    perc_diff.append([var[0], test_perc, train_perc])

perc_diff

for perc in perc_diff:

    print("The percentage of values in {col} in the training set that do not occur in the test set is {freq}".format(col=perc[0], freq=perc[2]))

    print("The percentage of values in {col} in the test set that do not occur in the training set is {freq}".format(col=perc[0], freq=perc[1]))