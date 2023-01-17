# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))

import io

import math

from operator import itemgetter



# Any results you write to the current directory are saved as output.





def parse_feature_id(feat_id):

    words = feat_id.split('_')

    line_num = int(words[0][1:])

    station_num = int(words[1][1:])

    feat_num = int(words[2][1:])

    return (line_num, station_num, feat_num)





def get_vector_of_visited_stations(row, *station_numbers):

    vec = {st_num: 0 for st_num in station_numbers}

    

    fields_dict = row.to_dict()

    Id = fields_dict.pop('Id', None)

    Response = fields_dict.pop('Response', None)

    

    for field_name, field_value in fields_dict.items():

        if not np.isnan(field_value):

            station_num = parse_feature_id(field_name)[1]

            vec[station_num] = 1

    

    vec = sorted(vec.items(), key = itemgetter(0))

    

    vec = tuple([tup[1] for tup in vec])

    

    return pd.Series({'id': Id, 'label': Response, 'sequence': vec})

    

    

def get_list_of_nonempty_fields(row):

    fields_dict = row.to_dict()



    Id = fields_dict.pop('Id', None)

    Response = fields_dict.pop('Response', None)



    nonempty_feature_names = []

    for field_name, field_value in fields_dict.items():

        if not np.isnan(field_value): nonempty_feature_names.append(parse_feature_id(field_name))



    # convert list of feature names into a tuple to make it hashable

    sequence_tuple = tuple(sorted(nonempty_feature_names))

    return pd.Series({'id': Id, 'label': Response, 'sequence': sequence_tuple})





def get_rid_of_feats(line_stat_feat):

    line_stat = [(l, s) for (l, s, f) in line_stat_feat]

    uniq_line_stat = set(line_stat)

    return tuple(sorted(uniq_line_stat))





def get_list_of_stations(col_names):

    st_numbers = [parse_feature_id(col_name)[1] for col_name in col_names]

    uniq_st_numbers = list(set(st_numbers))

    return sorted(uniq_st_numbers)



num_filename = '../input/train_numeric.csv'

cat_filename = '../input/train_categorical.csv'



example_data = ('one,two,three,four,Id\n'

                '1,,3,4,1\n'

                '2,20,,40,2\n'

                '100,200,300,400,3\n'

                '1000,,3000,,4\n'

                '7,,90,90,5\n'

                '8,,8,8,6\n'

                '5,,5,,7\n'

                '53,53,,53,8')





buf = io.StringIO(example_data)



# df = pd.read_csv(buf, header=0, dtype=np.float64)



#print(df)

#print(df.applymap(lambda x: np.isnan(x) if isinstance(x, float) else False))



num_df = pd.read_csv(num_filename, header=0, nrows=20000, dtype=np.float64)

cat_df = pd.read_csv(cat_filename, header=0, nrows=20000, dtype=str)



# change Id column type to float:

cat_df["Id"] = cat_df["Id"].astype(np.float64)



            

cat_df = cat_df.applymap(lambda x: int(x[1:]) if isinstance(x, str) and x[0] == 'T' else x)





df = num_df.merge(cat_df, on='Id', how='inner')



print(num_df.shape)

print(cat_df.shape)

print(df.shape)





col_names = list(df.columns.values)

if 'Id' in col_names: col_names.remove('Id')

if 'Response' in col_names: col_names.remove('Response')

    

list_of_stations = get_list_of_stations(col_names)



# to strings of 1's and 0's

nonempty_df = df.apply(func = get_vector_of_visited_stations, axis = 1, args = (list_of_stations))



# to lists of visited stations:

# nonempty_df = df.apply(func = get_list_of_nonempty_fields, axis = 1)

# nonempty_df['feature_ids'] = nonempty_df['feature_ids'].apply(get_rid_of_feats)



seq_df = nonempty_df.groupby('sequence')



# for name, group in seq_df:

#     print("NAME:")

#     print(name)

#     print("GROUP:")

#     print("////////////////////")

#     print(group)

#     print(group.__class__.__name__)

#     print("////////////////////")

#     print("^^^^")

# print()



groups_df = seq_df['id'].apply(func=list)



print()

print(groups_df.head(5))

print()

print(groups_df.shape)
# cluster sequence vectors



from sklearn.cluster import KMeans



num_clusters = 5

kmeans = KMeans(n_clusters = num_clusters)



sequence_vectors = np.array(sorted(groups_df.axes[0].tolist()))

kmeans.fit(sequence_vectors)
# create vector column of cluster labels

labels_col = np.array(kmeans.labels_)[:, np.newaxis]



# add 1 to every label so there is no zeros

labels_col += 1



print(sequence_vectors.shape)

print(labels_col.shape)



labeled_vectors = np.multiply(sequence_vectors, labels_col)



print(labeled_vectors.shape)



print(labeled_vectors[:5])

import matplotlib.pyplot as plt



#fig = plt.figure(figsize=(5, 5))

#ax = fig.add_subplot(111)



fig = plt.figure(figsize = (15,8))

ax = fig.add_subplot(111)



ax.matshow(labeled_vectors, aspect='auto')
# let's count how many samples there are in each cluster



def get_cluster(row, *station_numbers):

    row = get_vector_of_visited_stations(row, *station_numbers)

    vec = np.array(row['sequence'])

    cluster = kmeans.predict([vec])[0]

#    return row.append(pd.Series({'cluster': cluster}))

    return pd.Series({'cluster': cluster, 'id': row['id']})

    

clusters_df = df.apply(func = get_cluster, axis = 1, args = (list_of_stations))

print("Done")
counts_df = clusters_df.groupby('cluster').count()

print(counts_df)