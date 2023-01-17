import pandas as pd



from csv import QUOTE_NONE
def read_and_reformat(csv_path):

    df = pd.read_csv(csv_path,

                     sep='|',

                     encoding='iso-8859-1',

                     dtype=object,

                     header=None,

                     quoting=QUOTE_NONE,

                     names=['Surah', 'Ayah', 'Text'])    

    df['Text'] = df['Text'].str.replace('#NAME\?', '')

    df['Text'] = df['Text'].str.strip(',')

    return df
df = read_and_reformat('../input/English.csv')

df.head()
df.info()
df[df.isnull().any(axis=1)]
df.loc[1894]