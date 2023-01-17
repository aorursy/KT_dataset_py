import pandas as pd

df_google_play = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df_app_store = pd.read_csv('../input/app-store-apple-data-set-10k-apps/AppleStore.csv', index_col=0)
print("Google Play:",df_google_play.shape,"\n", list(df_google_play), "\n\nApp Store:",df_app_store.shape,"\n", list(df_app_store))
df_google_play.head()
df_app_store.head()
print("Google Play:\n",df_google_play.nunique(), df_google_play.shape,"\n\nApp Store:\n", df_app_store.nunique(), df_app_store.shape)
df_google_play[10472:10473]
df_google_play = df_google_play.loc[df_google_play['App'] != 'Life Made WI-Fi Touchscreen Photo Frame']

df_google_play.shape
df_google_play = df_google_play.loc[df_google_play['App'] != 'Life Made WI-Fi Touchscreen Photo Frame']

df_google_play.shape
col_list = list(df_google_play)

okay_to_drop_non_dupe = ['Reviews']

drop_dupe_list = [col for col in col_list if col not in okay_to_drop_non_dupe]

df_google_play_test = df_google_play.drop_duplicates(subset=drop_dupe_list)

df_google_play_test.shape
df_google_play_test = df_google_play.sort_values(by="Reviews", ascending=False)

df_google_play_test = df_google_play_test.drop_duplicates(subset="App")

df_google_play_test.shape
df_google_play = df_google_play_test.copy()

df_google_play.shape
df_google_play = df_google_play.loc[df_google_play['Price'] == '0']

df_google_play.shape, df_google_play['Price'].value_counts()
import string



def nonEnglishCharacterCount(app_name):

    non_eng_char_ct = 0

    for character in app_name:

        if ord(character) > 127:

            non_eng_char_ct += 1

    return non_eng_char_ct
df_google_play['num_non_eng_chars'] = [nonEnglishCharacterCount(i) for i in df_google_play['App']]
df_google_play = df_google_play.loc[df_google_play['num_non_eng_chars'] <= 3]

df_google_play.shape
df_app_store.shape
df_app_store['num_non_eng_chars'] = [nonEnglishCharacterCount(i) for i in df_app_store['track_name']]

df_app_store = df_app_store.loc[df_app_store['num_non_eng_chars'] <= 3]

df_app_store.shape
df_app_store.dtypes
df_app_store = df_app_store.loc[df_app_store['price'] == 0]

df_app_store.shape, df_app_store['price'].value_counts()
dupe_list = ['Mannequin Challenge', 'VR Roller Coaster']

df_check1 = df_app_store.loc[df_app_store['track_name'] == dupe_list[0]]

df_check2 = df_app_store.loc[df_app_store['track_name'] == dupe_list[1]]

df_check1
df_check2
df_app_store = df_app_store.sort_values(by='ver', ascending=False)

df_app_store = df_app_store.drop_duplicates(subset='track_name')

df_app_store.shape
# print(df_google_play['Genres'].value_counts().to_dict())

df_google_play['Genres'].value_counts(normalize=True)
# print(df_google_play['Category'].value_counts().to_dict())

df_google_play['Category'].value_counts(normalize=True)
# print(df_app_store['prime_genre'].value_counts().to_dict())

df_app_store['prime_genre'].value_counts(normalize=True)
df_family = df_google_play.loc[df_google_play['Category'] == 'FAMILY']

df_family['Genres'].value_counts(normalize=True)
df_family_entertainment = df_family.loc[df_family['Genres'] == 'Entertainment']

df_family_entertainment = pd.DataFrame(df_family_entertainment['App'].value_counts()).reset_index().drop(columns='App')

list_family_entertainment = df_family_entertainment['index'].tolist()

print(list_family_entertainment)
df_genre_category_relationship = df_google_play.loc[:,['Genres','Category']]

df_genre_category_relationship = df_genre_category_relationship.drop_duplicates()

len(df_genre_category_relationship)
df_g_c_r_counts = pd.DataFrame(df_genre_category_relationship['Genres'].value_counts()).reset_index()

df_g_c_r_counts = df_g_c_r_counts.rename(columns={'index':'Genres','Genres':'Counts'})

df_g_c_r_counts = df_g_c_r_counts.loc[df_g_c_r_counts['Counts'] >= 2]

len(df_g_c_r_counts)
check_list = list(df_g_c_r_counts['Genres'])

check_df = df_google_play.loc[df_google_play['Genres'].isin(check_list)]

check_df = check_df.loc[:,['Genres','Category']].drop_duplicates().sort_values('Genres')

check_df
df_google_play['Installs_count'] = [i.replace(',','').replace('+','') for i in df_google_play['Installs']]

df_google_play['Installs_count'] = df_google_play['Installs_count'].astype(int)

# df_google_play['Installs_count'].value_counts()
# Average installs (in millions) by Genre on Google Play

df_gp_avg_installs_g = df_google_play.groupby('Genres', as_index=False)['Installs_count'].mean().sort_values('Installs_count', 

                                                                                                          ascending=False)

df_gp_avg_installs_g['Installs_count'] = [i/1000000 for i in df_gp_avg_installs_g['Installs_count']]

df_gp_avg_installs_g = df_gp_avg_installs_g.rename(columns={'Installs_count':'Average_installs_count_in_millions'})

df_gp_avg_installs_g
# Average installs (in millions) by Category on Google Play

df_gp_avg_installs_c = df_google_play.groupby('Category', as_index=False)['Installs_count'].mean().sort_values('Installs_count', 

                                                                                                          ascending=False)

df_gp_avg_installs_c['Installs_count'] = [i/1000000 for i in df_gp_avg_installs_c['Installs_count']]

df_gp_avg_installs_c = df_gp_avg_installs_c.rename(columns={'Installs_count':'Average_installs_count_in_millions'})

df_gp_avg_installs_c
# Average number of ratings by prime genre on the App Store

df_a_avg_rat = df_app_store.groupby('prime_genre', as_index=False)['rating_count_tot'].mean().sort_values('rating_count_tot', ascending=False)

df_a_avg_rat['rating_count_tot'] = df_a_avg_rat['rating_count_tot'].astype(int)

df_a_avg_rat