import pandas as pd

import cufflinks as cf

cf.go_offline()
df = pd.read_parquet("/kaggle/input/omniture-visit/visit.parquet")

df.head()
N_TOP = 20 

top_users = list(df.user.value_counts().sort_values(ascending=False).keys())[:N_TOP]
df = df_orig[df_orig.user.isin(top_users)]
df['location_text'] = df['country'] + " " + df['city'] + " " + df["state"]
unique_location_user_series = df.groupby('user').location_text.nunique().sort_values(ascending=False)

unique_location_user_series = unique_location_user_series[unique_location_user_series > 1]

travellers = unique_location_user_series.keys()

unique_location_user_series
FREQ='1h'



def draw_user_by_location(username):

    df_user = df[df.user == username]

    df_grouped = df_user.groupby([pd.Grouper(freq=FREQ), 'location_text']).count().reset_index().pivot_table(index="ts",columns="location_text")["city"]

    df_grouped.iplot(title=username, kind='bar')
for user in travellers:

    draw_user_by_location(user)