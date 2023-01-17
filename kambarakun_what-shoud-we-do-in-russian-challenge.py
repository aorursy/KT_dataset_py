import pandas as pd
df_train = pd.read_csv('../input/ru_train.csv')

df_test = pd.read_csv('../input/ru_test.csv')
df_train[0:10]
df_test[0:10]
se_class_sum = df_train['class'].value_counts()

se_class_sum.name='class_sum'

se_class_ratio = 100 * se_class_sum / se_class_sum.sum()

se_class_ratio.name = '%class_sum'



df_class_ratio = pd.concat([se_class_sum, se_class_ratio], axis=1)

df_class_ratio
df_train_mismatch = df_train[df_train.before != df_train.after]

print('%f%% of df_train' % (100 * len(df_train_mismatch) / len(df_train)))

df_train_mismatch
pd_columns = ['train', 'test']

pd_index = ['same', 'mismatch', '%mismatch', 'total']

pd_data = [[len(df_train) - len(df_train_mismatch), '?'], [len(df_train_mismatch), '?'], ['%f%%' % (100 * len(df_train_mismatch) / len(df_train)), '?'], [len(df_train), len(df_test)]]



pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
se_class_sum_mismatch = df_train_mismatch['class'].value_counts()

se_class_sum_mismatch.name='class_sum'

se_class_ratio_mismatch = 100 * se_class_sum_mismatch / se_class_sum_mismatch.sum()

se_class_ratio_mismatch.name = '%class_sum'



df_class_ratio_mismatch = pd.concat([se_class_sum_mismatch, se_class_ratio_mismatch], axis=1)

df_class_ratio_mismatch
def func_str_int_str(str_query):

    try:

        return str(int(str_query))

    except:

        return 'FAIL!'



df_train_int_simple = df_train[df_train.before == df_train.before.map(func_str_int_str)]

print('%f%% of df_train' % (100 * len(df_train_int_simple) / len(df_train)))

df_train_int_simple
se_class_sum_int_simple = df_train_int_simple['class'].value_counts()

se_class_sum_int_simple.name='class_sum'

se_class_ratio_int_simple = 100 * se_class_sum_int_simple / se_class_sum_int_simple.sum()

se_class_ratio_int_simple.name = '%class_sum'



df_class_ratio_int_simple = pd.concat([se_class_sum_int_simple, se_class_ratio_int_simple], axis=1)

df_class_ratio_int_simple
se_class_sum_rest = (df_class_ratio_mismatch.class_sum - df_class_ratio_int_simple.class_sum).dropna().astype(int)

se_class_sum_rest.name = 'class_sum'

se_class_ratio_rest = 100 * se_class_sum_rest / se_class_sum_rest.sum()

se_class_ratio_rest.name = '%class_sum'



df_rest = pd.concat([se_class_sum_rest, se_class_ratio_rest], axis=1).sort_values(by='class_sum', ascending=False)

print('%f%% of df_train' % (100 * df_rest.class_sum.sum() / len(df_train)))



df_rest