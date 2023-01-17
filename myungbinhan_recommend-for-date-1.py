from tqdm import tqdm
import pandas as pd

df_total = pd.read_json('../input/arena-res/fe/df_total.json')
df_song_meta = pd.read_json('../input/arena-res/song_meta.json')

df_total_train = df_total[df_total['class']==0]

# month = 1
# rate = 0
# rep_range = 100

# for i in tqdm(range(rep_range)):
#     updt_date = df_total_train.iloc[i]['updt_date']
#     df1=df_total_train[
#         (df_total_train['updt_date'] < updt_date + month*15) & 
#         (df_total_train['updt_date'] >= updt_date - month*15)
#     ][['songs']].explode('songs')
    
#     gnr = pd.merge(
#         df1, df_song_meta[['id', 'song_gn_gnr_basket']], 
#         left_on='songs', right_on='id', how='left'
#     ).explode('song_gn_gnr_basket')
    
#     gnr_counts = gnr['song_gn_gnr_basket'].value_counts()
    
# #     song_counts = pd.merge(
# #         df1, df_song_meta[['id']], 
# #         left_on='songs', right_on='id', how='left'
# #     )['songs'].value_counts()

    

#     gnr_top_1 = gnr[gnr['song_gn_gnr_basket'] == gnr_counts.index[0]]['id'].value_counts()
#     #print(gnr_top_1)
    
#     song_list = gnr_top_1.index[:100]
#     #print(song_list)
    
    
#     correct = 0
    
#     target = df_total_train.iloc[i]['songs']
    
    
#     for song in target:
#         if song in song_list:
#             correct += 1
    
#     rate += correct / len(target) * 100
    
# print(rate / rep_range)
    
    
    
# #     print(gnr_counts)
# #     print(gnr_counts.index)
# #     print(gnr_counts.index[0])
# #     print(gnr_counts.index[1])
# #     print(gnr_counts.index[2])
    
# #     print(counts[0])
# #     print(counts[1])
# #     print(counts[2])
    
month = 1
rate = 0
rep_range = 100

for i in tqdm(range(rep_range)):
    updt_date = df_total_train.iloc[i]['updt_date']
    df1=df_total_train[
        (df_total_train['updt_date'] < updt_date + month*15) & 
        (df_total_train['updt_date'] >= updt_date - month*15)
    ][['songs']].explode('songs')
    
    
    song_counts = pd.merge(
        df1, df_song_meta[['id']], 
        left_on='songs', right_on='id', how='left'
    )['songs'].value_counts()

    
    
    song_list = song_counts.index[:100]
    
    
    correct = 0
    
    target = df_total_train.iloc[i]['songs']
    
    
    for song in target:
        if song in song_list:
            correct += 1
    
    rate += correct / len(target) * 100
    
print(rate / rep_range)

    
month = 1
rate = 0
rep_range = 50000

for i in tqdm(range(rep_range)):
    updt_date = df_total_train.iloc[i]['updt_date']
    df1=df_total_train[
        (df_total_train['updt_date'] < updt_date + month*15) & 
        (df_total_train['updt_date'] >= updt_date - month*15)
    ][['songs']].explode('songs')
    
    
    song_counts = pd.merge(
        df1, df_song_meta[['id']], 
        left_on='songs', right_on='id', how='left'
    )['songs'].value_counts()

    
    
    song_list = song_counts.index[:100]
    
    
    correct = 0
    
    target = df_total_train.iloc[i]['songs']
    
    
    for song in target:
        if song in song_list:
            correct += 1
    
    rate += correct / len(target) * 100
    
print(rate / rep_range)
    
month = 2
rate = 0
rep_range = 50000

for i in tqdm(range(rep_range)):
    updt_date = df_total_train.iloc[i]['updt_date']
    df1=df_total_train[
        (df_total_train['updt_date'] < updt_date + month*15) & 
        (df_total_train['updt_date'] >= updt_date - month*15)
    ][['songs']].explode('songs')
    
    
    song_counts = pd.merge(
        df1, df_song_meta[['id']], 
        left_on='songs', right_on='id', how='left'
    )['songs'].value_counts()

    
    
    song_list = song_counts.index[:100]
    
    
    correct = 0
    
    target = df_total_train.iloc[i]['songs']
    
    
    for song in target:
        if song in song_list:
            correct += 1
    
    rate += correct / len(target) * 100
    
print(rate / rep_range)
    
    
month = 3
rate = 0
rep_range = 50000

for i in tqdm(range(rep_range)):
    updt_date = df_total_train.iloc[i]['updt_date']
    df1=df_total_train[
        (df_total_train['updt_date'] < updt_date + month*15) & 
        (df_total_train['updt_date'] >= updt_date - month*15)
    ][['songs']].explode('songs')
    
    
    song_counts = pd.merge(
        df1, df_song_meta[['id']], 
        left_on='songs', right_on='id', how='left'
    )['songs'].value_counts()

    
    
    song_list = song_counts.index[:100]
    
    
    correct = 0
    
    target = df_total_train.iloc[i]['songs']
    
    
    for song in target:
        if song in song_list:
            correct += 1
    
    rate += correct / len(target) * 100
    
print(rate / rep_range)
    

month = 4
rate = 0
rep_range = 50000

for i in tqdm(range(rep_range)):
    updt_date = df_total_train.iloc[i]['updt_date']
    df1=df_total_train[
        (df_total_train['updt_date'] < updt_date + month*15) & 
        (df_total_train['updt_date'] >= updt_date - month*15)
    ][['songs']].explode('songs')
    
    
    song_counts = pd.merge(
        df1, df_song_meta[['id']], 
        left_on='songs', right_on='id', how='left'
    )['songs'].value_counts()

    
    
    song_list = song_counts.index[:100]
    
    
    correct = 0
    
    target = df_total_train.iloc[i]['songs']
    
    
    for song in target:
        if song in song_list:
            correct += 1
    
    rate += correct / len(target) * 100
    
print(rate / rep_range)
    

month = 5
rate = 0
rep_range = 50000

for i in tqdm(range(rep_range)):
    updt_date = df_total_train.iloc[i]['updt_date']
    df1=df_total_train[
        (df_total_train['updt_date'] < updt_date + month*15) & 
        (df_total_train['updt_date'] >= updt_date - month*15)
    ][['songs']].explode('songs')
    
    
    song_counts = pd.merge(
        df1, df_song_meta[['id']], 
        left_on='songs', right_on='id', how='left'
    )['songs'].value_counts()

    
    
    song_list = song_counts.index[:100]
    
    
    correct = 0
    
    target = df_total_train.iloc[i]['songs']
    
    
    for song in target:
        if song in song_list:
            correct += 1
    
    rate += correct / len(target) * 100
    
print(rate / rep_range)
    
