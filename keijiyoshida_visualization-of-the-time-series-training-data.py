import pandas as pd

import kagglegym as kg



env = kg.make()

observation = env.reset()

train = observation.train



def extract_df(train: pd.core.frame.DataFrame, id: int) -> pd.core.frame.DataFrame:

    df = train.loc[train.id == id, ['timestamp', 'y']]

    df.index = df.timestamp

    del df['timestamp']

    df.rename(columns={'y': str(id)}, inplace=True)

    return df



def extract_time_series_df(train: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    df = pd.DataFrame(index=train.timestamp.unique())

    for id in train.id.unique():

        df = df.join(extract_df(train, id))

    return df



df = extract_time_series_df(train)

df.head()
df[['10', '11', '12', '25']].plot()
df[['1906', '1919', '2081', '2097']].plot()
df.mean(axis=1).plot()
df.std(axis=1).plot()