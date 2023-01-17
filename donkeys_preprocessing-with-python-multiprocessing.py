import pandas as pd

import pyarrow.parquet as pq

import os

import numpy as np

import matplotlib.pyplot as plt
train_meta = pd.read_csv("../input/metadata_train.csv")

#train_meta.head(6)
test_meta = pd.read_csv("../input/metadata_test.csv")

#test_meta.head(6)
#I use bkt as short for bucket, rather than bin for bin since I tend to read bin as binary

bkt_count = 160

data_size = 800000

bkt_size = int(data_size/bkt_count)



def summarize_df_np(meta_df, data_type, p_id):

    count = 0

    measure_rows = []



    for measurement_id in meta_df["id_measurement"].unique():

        count += 1

        idx1 = measurement_id * 3

        input_col_names = [str(idx1), str(idx1+1), str(idx1+2)]

        df_sig = pq.read_pandas('../input/'+data_type+'.parquet', columns=input_col_names).to_pandas()

        df_sig = df_sig.clip(upper=127, lower=-127)



        df_diff = pd.DataFrame()

        for col in input_col_names:

            df_diff[col] = df_sig[col].diff().abs()

        

        data_measure = df_sig.values

        data_diffs = df_diff.values

        sig_rows = []

        sig_ts_rows = []

        for sig in range(0, 3):

            #take the data for each 3 signals in a measure separately

            data_sig = data_measure[:,sig]

            data_diff = data_diffs[:,sig]

            bkt_rows = []

            diff_avg = np.nanmean(data_diff)

            for i in range(0, data_size, bkt_size):

                # cut data to bkt_size (bucket size)

                bkt_data_raw = data_sig[i:i + bkt_size]

                bkt_avg_raw = bkt_data_raw.mean() #1

                bkt_sum_raw = bkt_data_raw.sum() #1

                bkt_std_raw = bkt_data_raw.std() #1

                bkt_std_top = bkt_avg_raw + bkt_std_raw #1

                bkt_std_bot = bkt_avg_raw - bkt_std_raw #1



                bkt_percentiles = np.percentile(bkt_data_raw, [0, 1, 25, 50, 75, 99, 100]) #7

                bkt_range = bkt_percentiles[-1] - bkt_percentiles[0] #1

                bkt_rel_perc = bkt_percentiles - bkt_avg_raw #7



                bkt_data_diff = data_diff[i:i + bkt_size]

                bkt_avg_diff = np.nanmean(bkt_data_diff) #1

                bkt_sum_diff = np.nansum(bkt_data_diff) #1

                bkt_std_diff = np.nanstd(bkt_data_diff) #1

                bkt_min_diff = np.nanmin(bkt_data_diff) #1

                bkt_max_diff = np.nanmax(bkt_data_diff) #1



                raw_features = np.asarray([bkt_avg_raw, bkt_std_raw, bkt_std_top, bkt_std_bot, bkt_range])

                diff_features = np.asarray([bkt_avg_diff, bkt_std_diff, bkt_sum_diff])

                bkt_row = np.concatenate([raw_features, diff_features, bkt_percentiles, bkt_rel_perc])

                bkt_rows.append(bkt_row)

            sig_rows.extend(bkt_rows)

        measure_rows.extend(sig_rows)

    df_sum = pd.DataFrame(measure_rows)

    #df_sum = df_sum.astype("float32")

    return df_sum

def process_subtrain(arg_tuple):

    meta, idx = arg_tuple

    df_sum = summarize_df_np(meta, "train", idx)

    return idx, df_sum
from sklearn.preprocessing import MinMaxScaler



minmax = MinMaxScaler(feature_range=(-1,1))
def create_chunk_indices(meta_df, chunk_idx, chunk_size):

    start_idx = chunk_idx * chunk_size

    end_idx = start_idx + chunk_size

    meta_chunk = meta_df[start_idx:end_idx]

    print("start/end "+str(chunk_idx+1)+":" + str(start_idx) + "," + str(end_idx))

    print(len(meta_chunk))

    #chunk_idx in return value is used to sort the processed chunks back into original order,

    return (meta_chunk, chunk_idx)
from multiprocessing import Pool



num_cores = 4



def process_train():

    #splitting here by measurement id's to get all signals for a measurement into single chunk

    measurement_ids = train_meta["id_measurement"].unique()

    df_split = np.array_split(measurement_ids, num_cores)

    chunk_size = len(df_split[0]) * 3

    

    chunk1 = create_chunk_indices(train_meta, 0, chunk_size)

    chunk2 = create_chunk_indices(train_meta, 1, chunk_size)

    chunk3 = create_chunk_indices(train_meta, 2, chunk_size)

    chunk4 = create_chunk_indices(train_meta, 3, chunk_size)



    #list of items for multiprocessing, 4 since using 4 cores

    all_chunks = [chunk1, chunk2, chunk3, chunk4]

    

    pool = Pool(num_cores)

    #this starts the (four) parallel processes and collects their results

    #-> process_subtrain() is called concurrently with each item in all_chunks 

    result = pool.map(process_subtrain, all_chunks)

    #parallel processing can be non-deterministic in timing, so here I sort results by their chunk id

    #to maintain results in same order as in original files (to match metadata from other file)

    print("sorting")

    result = sorted(result, key=lambda tup: tup[0])

    print("sorted")

    sums = [item[1] for item in result]

    

    df_train = pd.concat(sums)

    df_train = df_train.reset_index(drop=True)

    #np.save() would be another option but this works for now

    df_train.to_csv("my_train.csv.gz", compression="gzip")



    df_train_scaled = pd.DataFrame(minmax.fit_transform(df_train))

    df_train_scaled.to_csv("my_train_scaled.csv.gz", compression="gzip")

    return df_train, df_train_scaled
ps = process_train()
#first 10 rows of raw feature data

ps[0].head(10)
#same first 10 rows in scaled format

ps[1].head(10)
ps[1].values.shape
bkt_count*len(train_meta)
ps[1][0:160].plot(figsize=(8,5))
ps[1].iloc[:,0:5][0:160].plot()
ps[1].iloc[:,5:8][0:160].plot()
ps[1].iloc[:,8:15][0:160].plot()
ps[1].iloc[:,15:22][0:160].plot()
measurement_ids = train_meta["id_measurement"].unique()

rows = []

for mid in measurement_ids:

    idx1 = mid*3

    idx2 = idx1 + 1

    idx3 = idx2 + 1

    sig1_idx = idx1 * bkt_count

    sig2_idx = idx2 * bkt_count

    sig3_idx = idx3 * bkt_count

    sig1_data = ps[1][sig1_idx:sig1_idx+bkt_count]

    sig2_data = ps[1][sig2_idx:sig2_idx+bkt_count]

    sig3_data = ps[1][sig3_idx:sig3_idx+bkt_count]

    #this combines the above read 3*160 rows for 3 signals into 1 combined set with with 160 rows

    #and from 22 features on 3*160 to 66 (=22*3) features on 160 rows.

    row = np.concatenate([sig1_data, sig2_data, sig3_data], axis=1).flatten().reshape(bkt_count, sig1_data.shape[1]*3)

    rows.append(row)

df_train_combined = pd.DataFrame(np.vstack(rows))

df_train_combined.to_csv("my_train_combined_scaled.csv.gz", compression="gzip")

#slot 1 (measurement 1, signal 1, rows 0-159) for single signal version

ps[1].iloc[:,15:22][0:160].plot()
#slot 2 (measurement 1, signal 2, rows 160-319) for single signal version

ps[1].iloc[:,15:22][160:320].plot()
#slot 1 (measurement 1, signal 1) for combined signal version

df_train_combined.iloc[:,15:22][0:160].plot()
#slot 2 (measurement 1, signal 2) for combined signal version

df_train_combined.iloc[:,37:44][0:160].plot()
#signal 1, single signal version

ps[1].iloc[0:4]
#signal 2, single signal version

ps[1].iloc[160:164]
#signal 3, single signal version

ps[1].iloc[320:324]
#signal 4 (or signal 1 for measurement id 2))

ps[1].iloc[480:484]
df_train_combined.iloc[0:4]
df_train_combined.iloc[160:164]
del ps

del df_train_combined
def process_subtest(arg_tuple):

    meta, idx = arg_tuple

    df_sum = summarize_df_np(meta, "test", idx)

    return idx, df_sum
from multiprocessing import Pool



num_cores = 4



def process_test():

    measurement_ids = test_meta["id_measurement"].unique()

    df_split = np.array_split(measurement_ids, num_cores)

    chunk_size = len(df_split[0]) * 3

    

    chunk1 = create_chunk_indices(test_meta, 0, chunk_size)

    chunk2 = create_chunk_indices(test_meta, 1, chunk_size)

    chunk3 = create_chunk_indices(test_meta, 2, chunk_size)

    chunk4 = create_chunk_indices(test_meta, 3, chunk_size)



    all_chunks = [chunk1, chunk2, chunk3, chunk4]

    

    pool = Pool(num_cores)

    result = pool.map(process_subtest, all_chunks)

    result = sorted(result, key=lambda tup: tup[0])



    sums = [item[1] for item in result]



    df_test = pd.concat(sums)

    df_test = df_test.reset_index(drop=True)

    df_test.to_csv("my_test.csv.gz", compression="gzip")



    df_test_scaled = pd.DataFrame(minmax.transform(df_test))

    df_test_scaled.to_csv("my_test_scaled.csv.gz", compression="gzip")

    return df_test, df_test_scaled
pst = process_test()
pst[0].head(10)
pst[1].head(10)
pst[1].values.shape
pst[1][0:160].plot()
pst[1].iloc[:,0:5][0:160].plot()
pst[1].iloc[:,5:8][0:160].plot()
pst[1].iloc[:,8:15][0:160].plot()
pst[1].iloc[:,15:22][0:160].plot()
measurement_ids = test_meta["id_measurement"].unique()

start = measurement_ids[0]

rows = []

for mid in measurement_ids:

    #test measurement id's start from 2904 and indices at 0, so need to align

    mid = mid - start

    idx1 = mid*3

    idx2 = idx1 + 1

    idx3 = idx2 + 1

    sig1_idx = idx1 * bkt_count

    sig2_idx = idx2 * bkt_count

    sig3_idx = idx3 * bkt_count

    sig1_data = pst[1][sig1_idx:sig1_idx+bkt_count]

    sig2_data = pst[1][sig2_idx:sig2_idx+bkt_count]

    sig3_data = pst[1][sig3_idx:sig3_idx+bkt_count]

    row = np.concatenate([sig1_data, sig2_data, sig3_data], axis=1).flatten().reshape(bkt_count, sig1_data.shape[1]*3)

    rows.append(row)

df_test_combined = pd.DataFrame(np.vstack(rows))

df_test_combined.to_csv("my_test_combined_scaled.csv.gz", compression="gzip")

pst[1].iloc[0:4]
df_test_combined.head(4)
df_test_combined.shape