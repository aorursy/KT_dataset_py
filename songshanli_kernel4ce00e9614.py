import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
raw_tr = pd.read_csv("/kaggle/input/expedia-hotel-ranking-exercise/train.csv")
test_data = pd.read_csv("/kaggle/input/expedia-hotel-ranking-exercise/test.csv")

raw_tr.head()

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

search_features = [
    'srch_id',
 #'srch_date_time', 
 #'srch_visitor_id',
 #'srch_visitor_visit_nbr', excluded because I am not sure what does it mean
 #'srch_visitor_loc_country',excluded because it is coverred by 'srch_hcom_destination_id'
 #'srch_visitor_loc_region',excluded because it is coverred by 'srch_hcom_destination_id'
 #'srch_visitor_loc_city',excluded because it is coverred by 'srch_hcom_destination_id'
 #'srch_visitor_wr_member',
 #'srch_posa_continent', excluded because it is coverred by 'srch_hcom_destination_id'
 #'srch_posa_country', excluded because it is coverred by 'srch_hcom_destination_id'
 'srch_hcom_destination_id',
 #'srch_dest_longitude',excluded because it is coverred by 'srch_hcom_destination_id'
 #'srch_dest_latitude',excluded because it is coverred by 'srch_hcom_destination_id'
 #'srch_ci', excluded because property doesn't provide this information. 
 #'srch_co',excluded because property doesn't provide this information.
 #'srch_ci_day',excluded because property doesn't provide this information.
 #'srch_co_day',excluded because property doesn't provide this information.
 #'srch_los',
 #'srch_bw',
 'srch_adults_cnt',
 'srch_children_cnt',
 'srch_rm_cnt',
 #'srch_mobile_bool',
 #'srch_mobile_app',
 #'srch_device',
 #'srch_currency',
 #'srch_local_date'
                  ]
search_features
search_tr = raw_tr.loc[:, search_features]
search_tr.drop_duplicates(inplace=True)
search_tr.head()
print(search_tr.shape)
print(len(search_tr.srch_id.unique()))
assert search_tr.shape[0] == len(search_tr.srch_id.unique()) # make sure there is no duplicated prop_key
prop_features = [
    'prop_key',           
    #'prop_travelad_bool',  excluded because the same prop_key has multiple different values        
    #'prop_dotd_bool', excluded because the same prop_key has multiple different values 
    #'prop_price_without_discount_local', excluded because search doesn't cover this feature
    #'prop_price_without_discount_usd',excluded because search doesn't cover this feature
    #'prop_price_with_discount_local', excluded because search doesn't cover this feature
    #'prop_price_with_discount_usd',excluded because search doesn't cover this feature
    #'prop_imp_drr', excluded because the same prop_key has multiple different values
    #'prop_booking_bool', 
    'prop_brand_bool',
    'prop_starrating', #numeric feature
    #'prop_super_region', excluded because it is covered by prop_submarket_id
    #'prop_continent',excluded because it is covered by prop_submarket_id
    #'prop_country',  excluded because it is covered by prop_submarket_id
    #'prop_market_id',  excluded because it is covered by prop_submarket_id
    'prop_submarket_id', #rigid categorical feature
    'prop_room_capacity', 
    #'prop_review_score',  excluded because it is covered by prop_starrating, also the same prop_key has multiple different values
    #'prop_review_count', excluded for because the same prop_key has multiple different values. the better approach is to merge them.
    'prop_hostel_bool'
]
prop_features
prop_tr = raw_tr.loc[:, prop_features]
print(prop_tr.shape)
prop_tr.head()
prop_tr.drop_duplicates(inplace=True)
print(prop_tr.shape)
print(len(prop_tr.prop_key.unique()))

assert prop_tr.shape[0] == len(prop_tr.prop_key.unique()) # make sure there is no duplicated prop_key
booked_columns = ["srch_id", "prop_key", "prop_booking_bool" ]
search_prop_booking = raw_tr.loc[:, booked_columns]
search_prop_booked = search_prop_booking[search_prop_booking["prop_booking_bool"] > 0]
search_prop_booked.head()
print(search_prop_booked.shape)
print(len(search_prop_booked.prop_key.unique()))
print(len(search_prop_booked.srch_id.unique()))

assert search_prop_booked.shape[0] == len(search_prop_booked.srch_id.unique())
def compute_cosine(r1, r2, numeric_columns = ['prop_starrating'], excluded_columns = ['srch_id', 'prop_key'], 
                   rigid_features=['srch_hcom_destination_id', 'prop_submarket_id']):
    xy, xx, yy = 0,0,0
    for column_name in r1.columns:
        if column_name in excluded_columns:
            continue
        r1_v = r1.iloc[0][column_name]
        r2_v = r2.iloc[0][column_name]
        if column_name in numeric_columns:
            xy += r1_v * r2_v
            xx += r1_v * r1_v
            yy += r2_v * r2_v
        else:
            xx += 1
            yy += 1
            if str(r1_v) == str(r2_v):
                xy += 1
            elif column_name in rigid_features:
                return 0
                
    return xy/xx**0.5/yy**0.5
r1 = search_tr[0:1]
search_excluded_columns = ['srch_id']
coes = compute_cosine(r1,r1)
assert coes == 1.0
r1 = search_tr[0:1]
r2 = search_tr[51:52]
coes = compute_cosine(r1,r2)
print(r1)
print(r2)
print(coes)
r1 = search_tr[61:62]
r2 = search_tr[67:68]
coes = compute_cosine(r1,r2)
print(r1)
print(r2)
print(coes)
p1 = prop_tr[0:1]
p2 = prop_tr[51:52]
coes = compute_cosine(p1,p2)
print(p1)
print(p2)
print(coes)
p1 = prop_tr[61:62]
p2 = prop_tr[67:68]
coes = compute_cosine(p1,p2)
print(p1)
print(p2)
print(coes)
def ranking_score_by_cosine(s, p):
    rank_score = 0
    for index, row in search_prop_booked.iterrows():
        srch_id = row['srch_id']
        prop_key = row['prop_key']
        s_tmp = search_tr[search_tr['srch_id'] == srch_id]
        p_tmp = prop_tr[prop_tr['prop_key'] == prop_key]
        s_to_s_tmp_sim = compute_cosine(s,s_tmp)
        p_to_p_tmp_sim = compute_cosine(p,p_tmp)
        rank_score += (s_to_s_tmp_sim - 0.5) * (p_to_p_tmp_sim - 0.5) # 0.5s are seperate hyper-parameters. should search for better ones.
    return rank_score
def compute_prop_sim_matrix():
    matrix = {}
    size = prop_tr.shape[0]
    for index_r, r in prop_tr.iterrows():
        key_r = int(r['prop_key'])
        p1 = prop_tr[prop_tr['prop_key'] == key_r]
        for index_l, l in prop_tr.iterrows():
            key_l = int(l['prop_key'])
            if (key_l, key_r) in matrix:
                matrix[(key_r, key_l)] = matrix[(key_l, key_r)]
                continue
            p2 = prop_tr[prop_tr['prop_key'] == key_l]
            sim = compute_cosine(p1,p2)
            matrix[(key_r, key_l)] = sim
    return matrix
p_to_p_sim_matrix = compute_prop_sim_matrix()
def ranking_score_by_cosine_use_p_matrix(s, p):
    rank_score = 0
    target_prop_key = p['prop_key'][0]
    for index, row in search_prop_booked.iterrows():
        srch_id = row['srch_id']
        prop_key = row['prop_key']
        s_tmp = search_tr[search_tr['srch_id'] == srch_id]
        p_tmp = prop_tr[prop_tr['prop_key'] == prop_key]
        s_to_s_tmp_sim = compute_cosine(s,s_tmp)
        p_to_p_tmp_sim = p_to_p_sim_matrix(target_prop_key,prop_key)# compute_cosine(p, p_tmp)
        rank_score += (s_to_s_tmp_sim - 0.5) * (p_to_p_tmp_sim - 0.5) # 0.5s are seperate hyper-parameters. should search for better ones.
    return rank_score
def ranking_test(t_ds):
    t_ds.srch_id.unique()
    s_p_scores_list = []
    for srch_id in t_ds.srch_id.unique():
        search_part = t_ds.loc[:, search_features]
        s = search_part[search_part['srch_id'] == srch_id][0:1]
        s_p_scores = []
        for prop_key in prop_tr.prop_key.unique():
            p = prop_tr[prop_tr['prop_key'] == prop_key]
            s_p_score = ranking_score_by_cosine_use_p_matrix(s, p)
            s_p_scores.append((srch_id, prop_key, s_p_score))
            print('s_p_score', (srch_id, prop_key, s_p_score))
        s_p_scores.sort(key = lambda el : - e[2])
        s_p_scores_list.append(s_p_scores)
    return s_p_scores_list
import pandas as pd
test_ds = pd.read_csv("C:\\Dev\\study\\rl\\interview\\expedia\\test.csv")
p = test_ds[:1]
print(p['prop_key'][0])

result = ranking_test(test_ds[:1])
#print(result)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load




