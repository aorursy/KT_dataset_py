# 태그 시리즈 -> 태그 형태소 2차원 배열 반환
def get_tag_morphs(series, morph_dict):
    morph_lists = []
    for idx, li in enumerate(series):
        morph_list = []
        if len(li) == 0:
            morph_lists.append([])
        else:
            for i in li:
                try:
                    morph_list = morph_list + morph_dict[i]
                except:
                    print(idx, '번 인덱스에 이상태그 발견')
            morph_lists.append(morph_list)
    return morph_lists


# 타이틀 시리즈 -> 타이틀 형태소 2차원 배열로 반환
def get_title_morphs(series, morph_dict):
    morph_lists = []
    for idx, title in enumerate(series):
        morph_list = []
        if title == '':
            morph_lists.append([])
        else:
            for t in title.split():
                try:
                    morph_list = morph_list + morph_dict[t]
                except:
#                     print(idx, '번 인덱스에 이상 타이틀 발견')
                    pass
            morph_lists.append(morph_list)
    return morph_lists


# 형태소 -> tfidf & labeling 한 series 리턴
def morph_to_tfidf(df_id_morph):
    col_id, col_morph = df_id_morph.columns
    col_tfidf = col_morph + '_tfidf'
    col_label = col_tfidf+'_label'
    
    vectorizer = TfidfVectorizer(token_pattern=r'[가-힣|a-z|A-Z|0-9]+$')
    tfidf = vectorizer.fit_transform(df_id_morph[col_morph].str.join(" ").to_list())
    feature_names = vectorizer.get_feature_names()
    
    df_id_morph_ex = df_id_morph.explode(col_morph).dropna()
    df_id_morph_tfidf = pd.DataFrame(
        df_id_morph_ex[df_id_morph_ex[col_morph].isin(feature_names)]
    ).rename(columns={col_morph:col_tfidf})
    
    
    le = LabelEncoder()
    df_id_morph_tfidf[col_label]=le.fit_transform(df_id_morph_tfidf[col_tfidf])
    df_id_label = pd.DataFrame(df_id_morph_tfidf.groupby('id')[col_label].apply(list)).reset_index(level=0)
    df_id_tfidf = pd.DataFrame(df_id_morph_tfidf.groupby('id')[col_tfidf].apply(list)).reset_index(level=0)
    df_merged = pd.merge(df_id_morph, df_id_tfidf, on=col_id, how='left')
    df_merged = pd.merge(df_merged, df_id_label, on=col_id, how='left')


    for idx in df_merged.loc[df_merged[col_label].isnull(), col_label].index:
        df_merged.at[idx, col_label] = []
        df_merged.at[idx, col_tfidf] = []
        
#     return df_merged[col_label]
    return df_merged, le


# 형태소 -> tfidf & labeling 한 series 리턴
def morph_to_tfidf2(df_id_morph):
    col_id, col_morph = df_id_morph.columns
    col_tfidf = col_morph + '_tfidf'
    col_label = col_tfidf+'_label'
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df_id_morph[col_morph].str.join(" ").to_list())
    feature_names = vectorizer.get_feature_names()
    
    df_id_morph_ex = df_id_morph.explode(col_morph).dropna()
    df_id_morph_tfidf = pd.DataFrame(
        df_id_morph_ex[df_id_morph_ex[col_morph].isin(feature_names)]
    ).rename(columns={col_morph:col_tfidf})
    
    
    le = LabelEncoder()
    df_id_morph_tfidf[col_label]=le.fit_transform(df_id_morph_tfidf[col_tfidf])
    df_id_label = pd.DataFrame(df_id_morph_tfidf.groupby('id')[col_label].apply(list)).reset_index(level=0)
    df_id_tfidf = pd.DataFrame(df_id_morph_tfidf.groupby('id')[col_tfidf].apply(list)).reset_index(level=0)
    df_merged = pd.merge(df_id_morph, df_id_tfidf, on=col_id, how='left')
    df_merged = pd.merge(df_merged, df_id_label, on=col_id, how='left')


    for idx in df_merged.loc[df_merged[col_label].isnull(), col_label].index:
        df_merged.at[idx, col_label] = []
        df_merged.at[idx, col_tfidf] = []
        
#     return df_merged[col_label]
    return df_merged, le

def get_issuedate_mean_std(songs_ndarray):

    mean_list = []
    std_list = []

    for song_list in tqdm(songs_ndarray, desc='adding issue_date mean&std'):
        mean_list.append(df_song_meta[df_song_meta['id'].isin(song_list)].issue_date.mean())
        std_list.append(df_song_meta[df_song_meta['id'].isin(song_list)].issue_date.std())

    return np.nan_to_num(np.array(mean_list), copy=False), np.nan_to_num(np.array(std_list), copy=False)





# 입력한 song 과 feature로 sparse matrix 생성
def make_sparse_matrix(df_song_fe, shape):
    col_song, col_fe = df_song_fe.columns
    data = np.ones(len(df_song_fe))
    row = df_song_fe[col_song]
    col = df_song_fe[col_fe]
    sparse_matrix = sparse.csr_matrix((data, (row, col)), shape=(shape[0], shape[1]))
    return sparse_matrix


# 현재 곡에 5개 이상 포함된 앨범리스트 반환
def get_imp_album_list(cur_song_list, ge = 5):
    cur_album_cnt = df_song_meta.iloc[cur_song_list]['album_id'].value_counts()
    return cur_album_cnt[cur_album_cnt >= ge].index.to_list()


# 현재 곡에 10개 이상 포함된 가수리스트 반환
def get_imp_artist_list(cur_song_list, ge = 10):
    cur_artist_cnt = df_song_meta.iloc[cur_song_list][['artist_id_basket']].explode('artist_id_basket')['artist_id_basket'].value_counts()
    return cur_artist_cnt[cur_artist_cnt >= ge].index.to_list()


def rec_songs_by_id(cosine_similarity_obj, cur_song_list, cur_updt_date, gt=0, largest_n = 100):
    # 현재곡 col 선택
    df_cur = pd.DataFrame(cosine_similarity_obj[:, cur_song_list].A)

    # (중복횟수 + 유사도 최대값) 내림차순 정렬
    # ex) 3번 중복해서 나오고 각각의 유사도가 0.3, 0.4, 0.5 이면, 3 + 0.5 = 3.5
    df_cur_trans = pd.DataFrame(((df_cur != 0).sum(axis=1) + df_cur.max(axis=1)), columns=['cos'])
    
    # 유사도가 gt 초과인 곡 선택
    df_cur_trans = df_cur_trans[df_cur_trans['cos'] > gt]

    # 플레이리스트 생성일까지 발매된 곡 선택
    date_songs_list = df_song_meta[['id', 'issue_date']][df_song_meta['issue_date'] > cur_updt_date]['id'].to_list()
    df_cur_trans = df_cur_trans[~df_cur_trans.index.isin(date_songs_list)]
    
    # 현재 플레이리스트에 없는 곡으로 상위 largest_n 개 곡 선택
    rec_song_list = df_cur_trans[~df_cur_trans.index.isin(cur_song_list)].sort_values('cos', ascending=False)[:largest_n].index.to_list()
    return rec_song_list


def rec_songs_by_one_vector(sparse_matrix, cur_vec_list, cur_song_list, cur_updt_date, calc_size=1000, gt=0, largest_n=100):
    cur_vec_len = len(cur_vec_list)
    
    cos_sum = sparse.csr_matrix(np.array([[]]).reshape(0, 1)) # empty sparse
    
    data = np.ones(cur_vec_len)
    row = np.array([0]*cur_vec_len)
    col = cur_vec_list
    cur_sparse = sparse.csr_matrix((data, (row, col)), shape=(1, sparse_matrix.shape[1]))
    
    for i in range(0, sparse_matrix.shape[0], calc_size): 
        tar_sparse = sparse_matrix[i:i+calc_size, :]
        calc_sparse = sparse.vstack([cur_sparse, tar_sparse])

        cos = cosine_similarity(calc_sparse, dense_output=False)[1:, :1]
        cos_sum = sparse.vstack([cos_sum, cos])
        
    df_cos = pd.DataFrame(cos_sum.A, columns=['cos'])
    
    # 유사도가 gt 초과인 곡 선택
    df_cos_filtered = df_cos[df_cos.cos > gt]
    
    # 플레이리스트 생성일까지 발매된 곡 선택
    date_songs_list = df_song_meta[['id', 'issue_date']][df_song_meta['issue_date'] > cur_updt_date]['id'].to_list()
    df_cos_filtered = df_cos_filtered[~df_cos_filtered.index.isin(date_songs_list)]
    
    # 현재리스트에 있는 곡 제거
    df_cos_filtered = df_cos_filtered[~df_cos_filtered.index.isin(cur_song_list)]
    
    # 유명도 컬럼 불러오기
    df_cos_filtered = pd.merge(df_cos_filtered, df_song_meta[['id', 'popular']], left_index=True, right_on='id', how='left')
    
    # 유사도 1000곡 뽑고 유명도로 정렬해줄까...
#     return df_cos_filtered.sort_values('cos', ascending=False)[:1000].sort_values('popular', ascending=False)[:largest_n]#.index.to_list()
    # 현재는 전체에서 유사도 > 유명도 순서로 정렬
    return df_cos_filtered.sort_values(by=['cos', 'popular'], ascending=False)[:largest_n].index.to_list()


def rec_song_sum(id_rec_song_list, tag_rec_song_list, n_vs_1):
    if tag_rec_song_list:
        cnt = 0
        len_id_rec = len(id_rec_song_list)
        len_tag_rec = len(tag_rec_song_list)
        rec_sum = []
        for i in range(len_id_rec):
            rec_sum.append(id_rec_song_list[i])
            if (i+1) % n_vs_1 == 0:
                rec_sum.append(tag_rec_song_list[cnt])
                cnt += 1
                if cnt >= len_tag_rec:
                    rec_sum = rec_sum + id_rec_song_list[i+1:]
                    break
            if i == len_id_rec-1:
                rec_sum = rec_sum + tag_rec_song_list[cnt:]
        rec_sum = list(pd.unique(rec_sum))        
    else:
        rec_sum = id_rec_song_list
        
    return rec_sum[:100]


def genre_most_popular(song_list, cur_updt_date):

    df_song_meta_ex = df_song_meta[['id', 'song_gn_gnr_basket', 'popular']].explode('song_gn_gnr_basket')
    top_genre = df_song_meta_ex[df_song_meta_ex['id'].isin(song_list)]['song_gn_gnr_basket'].value_counts().index[0]

    df_genre_song = df_song_meta_ex[df_song_meta_ex['song_gn_gnr_basket']==top_genre]
    df_genre_song_filtered = df_genre_song[~df_genre_song['id'].isin(song_list)]
    
    date_songs_list = df_song_meta[['id', 'issue_date']][df_song_meta['issue_date'] > cur_updt_date]['id'].to_list()
    df_genre_song_filtered = df_genre_song_filtered[~df_genre_song_filtered['id'].isin(date_songs_list)]

    return df_genre_song_filtered.sort_values('popular', ascending=False)['id'][:100].to_list()




import numpy as np
import pandas as pd
import distutils.dir_util
import json

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import datetime as dt
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


df_train = pd.read_json('../input/arena-res/train_test_split/orig/train.json')
df_val = pd.read_json('../input/arena-res/train_test_split/questions/val.json')
df_song_meta = pd.read_json('../input/arena-res/song_meta.json')[['id', 'song_gn_gnr_basket', 'album_id', 'artist_id_basket', 'issue_date']]

with open('../input/arena-res/morphs/all_tags_morphs.json') as json_file:
    tag_morph_dict = json.load(json_file)
    
with open('../input/arena-res/morphs/all_plylst_title_morphs.json') as json_file:
    title_morph_dict = json.load(json_file)
    

#popular 컬럼 추가
df_song_meta.loc[:,'popular'] = df_song_meta[['id']].merge(
    pd.DataFrame(df_train[['songs']].explode('songs')['songs'].value_counts()).rename(columns={'songs': 'popular'}), 
    left_on='id', 
    right_index=True, how='left'
).fillna(0)['popular']


# issue_date -> ordinal date
# 이상치 처리
df_song_meta.loc[df_song_meta.issue_date == 0, 'issue_date'] = 19000101  # 최소값
df_song_meta.loc[df_song_meta.id == 692325, 'issue_date'] = 20150130  # 검색결과: Broken Ones - Jacquie / 20105130
df_song_meta.loc[df_song_meta.id == 168071, 'issue_date'] = 20010930 # 9월 31일
df_song_meta.loc[df_song_meta.issue_date == 20220113, 'issue_date'] = 20200113 # 2022년 1월 13일

year = df_song_meta['issue_date'].astype(str).str.slice(0,4)
month = df_song_meta['issue_date'].astype(str).str.slice(4,6)
day = df_song_meta['issue_date'].astype(str).str.slice(6,8)

month[month == '00'] = '01'
day[day == '00'] = '01'
day[day == '32'] = '01'

df_song_meta['issue_date'] = pd.to_datetime(year + month + day, format='%Y%m%d').map(dt.datetime.toordinal)

df_song_meta_artist_ex = df_song_meta[['id', 'artist_id_basket', 'popular', 'issue_date']].explode('artist_id_basket')


df_train['class'] = 0
df_val['class'] = 1
df_total = pd.concat([df_train, df_val]).reset_index(drop=True)

    
# 태그, 타이틀 형태소 & tf-idf & label

# tag 1글자 포함한 버전
df_total['tag_morphs'] = get_tag_morphs(df_total['tags'], tag_morph_dict)
df_ta_m, le_tag = morph_to_tfidf(df_total[['id', 'tag_morphs']])
df_total['tag_morphs_tfidf'] = df_ta_m['tag_morphs_tfidf']
df_total['tag_morphs_tfidf_label'] = df_ta_m['tag_morphs_tfidf_label']

# tag 1글자 삭제한 버전
df_ta_m2, le_tag2 = morph_to_tfidf2(df_total[['id', 'tag_morphs']])
df_total['tag_morphs_tfidf2'] = df_ta_m2['tag_morphs_tfidf']
df_total['tag_morphs_tfidf_label2'] = df_ta_m2['tag_morphs_tfidf_label']

# title 1글자 포함한 버전
df_total['title_morphs'] = get_title_morphs(df_total['plylst_title'], title_morph_dict)
df_ti_m, le_title = morph_to_tfidf(df_total[['id', 'title_morphs']])
df_total['title_morphs_tfidf'] = df_ti_m['title_morphs_tfidf']
df_total['title_morphs_tfidf_label'] = df_ti_m['title_morphs_tfidf_label']

# title 1글자 삭제한 버전
df_ti_m2, le_title2 = morph_to_tfidf2(df_total[['id', 'title_morphs']])
df_total['title_morphs_tfidf2'] = df_ti_m2['title_morphs_tfidf']
df_total['title_morphs_tfidf_label2'] = df_ti_m2['title_morphs_tfidf_label']


# updt_date 를 ordinal 로 변경
df_total['updt_date'] = pd.to_datetime(df_total.updt_date.str.slice(0,10), format='%Y-%m-%d').map(dt.datetime.toordinal)
# df_total => df_train_use, df_val_use, df_test_use 로 나누기
df_train_use = df_total[df_total['class'] == 0][
    [
        'id', 'songs', 'tag_morphs_tfidf_label2', 
        'title_morphs_tfidf_label2', 'updt_date',
    ]
].reset_index(drop=True)

df_val_use = df_total[df_total['class'] == 1][
    [
        'id', 'songs', 'tag_morphs_tfidf_label2', 
        'title_morphs_tfidf_label2', 'updt_date',
    ]
].reset_index(drop=True)


# df_val_use, df_test_use 날짜의 평균, 표준편차 컬럼 추가
df_val_use['issue_date_mean'], df_val_use['issue_date_std'] = get_issuedate_mean_std(df_val_use.songs)

df_ques = pd.read_json('../input/arena-res/train_test_split/questions/val.json')
df_ques
df_ans = pd.read_json('../input/arena-res/train_test_split/answers/val.json')
df_ans
df_ques['song_len'] = None
df_ques['tag_len'] = None
df_ques['title_len'] = None

def return_song_len(row):
    return len(row['songs'])

def return_tag_len(row):
    return len(row['tags'])

def return_title_len(row):
    if row['tags'] == '':
        return 0
    else:
        return 1

df_ques['song_len'] = df_ques.apply(return_song_len, axis=1)
df_ques['tag_len'] = df_ques.apply(return_tag_len, axis=1)
df_ques['title_len'] = df_ques.apply(return_tag_len, axis=1)

df_ques
#df_val_use[df_val_use['song_len'] > 0]
title = df_ques[(df_ques['title_len'] > 0)]
title = title[['tags', 'id', 'plylst_title', 'songs', 'like_cnt', 'updt_date']]
title

# date = tag['updt_date']

# year = date.astype(str).str.slice(0,4)
# month = date.astype(str).str.slice(4+1,6+1)
# day = date.astype(str).str.slice(6+2,8+2)

# only_tag['updt_date'] = pd.to_datetime(year + month + day, format='%Y%m%d').map(dt.datetime.toordinal)

# only_tag
df_total_2 = df_total[['id', 'songs', 'tag_morphs_tfidf_label2', 'title_morphs_tfidf_label2', 'updt_date']]
df_total_2

df_total_2 = df_total_2[df_total_2['id'].isin(title.id.to_list())].reset_index(drop=True)
df_total_2

# total_only_tag_dummy = pd.merge(only_tag, df_total_2, how='inner', on='id')
# total_only_tag = pd.DataFrame()
# total_only_tag[['id', 'songs', 'tag_morphs_tfidf_label2', 'title_morphs_tfidf_label2', 'updt_date', 'issue_date_mean', 'issue_date_std']] = total_only_tag_dummy[['id', 'songs_x', 'tag_morphs_tfidf_label2', 'title_morphs_tfidf_label2', 'updt_date_x', 'issue_date_mean', 'issue_date_std']]
# total_only_tag
# ans_only_song_dummy = pd.merge(only_song, df_ans, how='inner', on='id')
# ans_only_song = pd.DataFrame()
# ans_only_song[['tags', 'id', 'plylst_title', 'songs', 'like_cnt','updt_date']] = ans_only_song_dummy[['tags_y', 'id', 'plylst_title_y', 'songs_y', 'like_cnt_y','updt_date_y']]
# ans_only_song
total_length = 500

n_songs = len(df_song_meta)
n_ids = df_total['id'].max()+1
n_tags = len(le_tag2.classes_)
n_titles = len(le_title2.classes_)

#sp_id = make_sparse_matrix(df_train_use[['songs', 'id']].explode('songs'), shape=(n_songs, n_ids))
#id_cos_sim = cosine_similarity(sp_id, dense_output=False)

sp_tag = make_sparse_matrix(
    df_train_use[['songs', 'tag_morphs_tfidf_label2']].explode('songs').explode('tag_morphs_tfidf_label2').dropna(), 
    shape=(n_songs, n_tags)
)
sp_tag_scaled = sp_tag / sp_tag.max()

sp_title = make_sparse_matrix(
    df_train_use[['songs', 'title_morphs_tfidf_label2']].explode('songs').explode('title_morphs_tfidf_label2').dropna(), 
    shape=(n_songs, n_titles)
)
sp_title_scaled = sp_title / sp_title.max()

sample_songs = [144663, 116573, 357367, 366786, 654757, 133143, 675115, 349492, 463173,
    396828, 42155, 461341, 174749, 701557, 610933, 520093, 13281, 418935, 449244, 650494,
    680366, 485155, 549178, 11657, 169984, 523521, 648628, 422915, 187047, 547967, 422077,
    625875, 350309, 215411, 442014, 132994, 427724, 300087, 627363, 581799, 253755, 668128,
    339802, 348200, 663256, 26083, 505036, 643628, 582252, 448116, 37748, 199262, 235773,
    339124, 140867, 341513, 68348, 407828, 209135, 209993, 493762, 105140, 487911, 509998,
    531820, 672550, 27469, 157055, 232874, 152422, 75842, 473514, 519391, 377243, 224921,
    295250, 446812, 678762, 351342, 464051, 246531, 146989, 117595, 15318, 205179, 108004,
    645489, 152475, 302646, 590012, 95323, 13198, 343974, 236393, 333595, 6546, 88503,
    443914, 459256, 640657]
sample_tags = ['기분전환', '감성', '휴식', '발라드', '잔잔한', '드라이브', '힐링', '사랑', '새벽', '밤']


df = df_total_2


answers = []

for idx in tqdm(df.index[0 : total_length], desc='rec songs'):
    cur_row = df.iloc[idx]

    cur_id = cur_row['id']
    cur_song_list = cur_row['songs']
    cur_tag_list = cur_row['tag_morphs_tfidf_label2']
    cur_title_list = cur_row['title_morphs_tfidf_label2']
    cur_updt_date = cur_row['updt_date']
    #cur_issue_date_mean = cur_row['issue_date_mean']
    #cur_issue_date_std = cur_row['issue_date_std']

    len_cur_song = len(cur_song_list)
    len_cur_tag = len(cur_tag_list)
    len_cur_title = len(cur_title_list)

    rec_song_list = []


    rec_song_list = rec_songs_by_one_vector(sp_title_scaled, cur_title_list, cur_song_list, cur_updt_date, largest_n=100)

    if len(rec_song_list) == 0:
        # 아무것도 없으면 updt_date 기준 추천곡
        rec_song_list = sample_songs
    elif len(rec_song_list) < 100:
        # 부족하면 rec_song 기준 장르 most popular 고고
        genre_popular_songs = genre_most_popular(rec_song_list, cur_updt_date)
        rec_song_list = list(pd.unique(rec_song_list+genre_popular_songs))[:100]


    if len(rec_song_list) <100:
        print('cur_id:', cur_id, ',', len(rec_song_list),'개')
    answers.append({
        "id": int(cur_id),
        "songs": rec_song_list,
        "tags": sample_tags
    })
distutils.dir_util.mkpath("./cos")
df_result = pd.DataFrame(answers)
df_result.to_json('./cos/results.json', orient='records', force_ascii=False)
df_answer = pd.read_json('../input/arena-res/train_test_split/answers/val.json')
df_answer = df_answer[df_answer['id'].isin(df_result.id.to_list())]
df_answer.to_json('./cos/answer.json', orient='records', force_ascii=False)
# -*- coding: utf-8 -*-
import io
import os
import json
import distutils.dir_util
from collections import Counter

import numpy as np


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))


def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]


def most_popular(playlists, col, topk_count):
    c = Counter()

    for doc in playlists:
        c.update(doc[col])

    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]
# import fire
import numpy as np




class ArenaEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = load_json(rec_fname)

        gt_ids = set([g["id"] for g in gt_playlists])
        rec_ids = set([r["id"] for r in rec_playlists])

        if gt_ids != rec_ids:
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        rec_song_counts = [len(p["songs"]) for p in rec_playlists]
        rec_tag_counts = [len(p["tags"]) for p in rec_playlists]

        if set(rec_song_counts) != set([100]):
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(rec_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

        rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]
        rec_unique_tag_counts = [len(set(p["tags"])) for p in rec_playlists]

        if set(rec_unique_song_counts) != set([100]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(rec_unique_tag_counts) != set([10]):
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")

        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)


# if __name__ == "__main__":
#     fire.Fire(ArenaEvaluator)
a = ArenaEvaluator()
a.evaluate('./cos/answer.json', './cos/results.json') #0.25581

# 500
# Music nDCG: 0.013194
# Tag nDCG: 0.145789
# Score: 0.0330832
ans_song_mat = []
for songs in df_answer.songs:
    ans_song_mat.append(pd.unique(songs)[:100])    
    
ans_tag_mat = []
for tags in df_answer.tags:
    ans_tag_mat.append(pd.unique(tags)[:10])
    
    
song_mat = []
for songs in df_result.songs:
    song_mat.append(pd.unique(songs)[:100])    
    
tag_mat = []
for tags in df_result.tags:
    tag_mat.append(pd.unique(tags)[:10])
    
#rate = 0
li = []

for i in range(total_length):
    target = ans_song_mat[i]
    song_list = song_mat[i]
    
    correct = 0

    for song in target:
        if song in song_list:
            correct += 1
            
    #print(correct / len(target) * 100)
    li.append(correct / len(target) * 100)
    #rate += correct / len(target) * 100

print(np.mean(li))

#print(rate / total_length)
pd.DataFrame({
    'id':df_ques[:total_length].id,
    'cor':li
}).to_json('./cos/rate.json', orient='records', force_ascii=False)


rate = pd.read_json('./cos/rate.json')
rate
# b.iloc[9]#['songs']
# # b.iloc[9]['id']



# df_ans = pd.read_json('./cos/results100.json')
# ids = []
# for idx in tqdm(df_ans.index):
#     song_list = df_ans.iloc[idx]['songs']
#     row_song_len=len(list(set(song_list)))
#     if row_song_len < 100:
#         id =  df_ans.iloc[idx]['id']
#         print('id:', id, ', row_song_len:',row_song_len)
        
#         ids.append(id)
# ids
