import pandas as pd





df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
pd_columns = ['length']

pd_index   = ['train', 'test']

pd_data    = [len(df_train), len(df_test)]



pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
pd_columns = ['question1', 'question2']

pd_index   = ['train', 'test']

pd_data    = [

    [

        max([len(str(x).split(' ')) for x in df_train.question1.tolist()]),

        max([len(str(x).split(' ')) for x in df_train.question2.tolist()])

    ],

    [

        max([len(str(x).split(' ')) for x in df_test.question1.tolist()]),

        max([len(str(x).split(' ')) for x in df_test.question2.tolist()])

    ]]



pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
pd_columns = ['duplicate', 'not duplicate', 'total', '%duplication']

pd_index   = ['train']

pd_data    = [len(df_train), len(df_test)]



pd_data    = [

    [

        df_train.is_duplicate.sum(),

        len(df_train) - df_train.is_duplicate.sum(),

        len(df_train),

        df_train.is_duplicate.sum() / len(df_train) * 100

    ]]



pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)