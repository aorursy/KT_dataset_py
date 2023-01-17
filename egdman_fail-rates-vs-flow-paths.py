# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def station_number(col_name):

    return int(col_name.split('_')[1][1:])





def get_combination(row):

#    feats = [(feat_name, feat_exists) for feat_name, feat_exists in row.items()\

#             if feat_name not in ['Id', 'Response']]

    combin = [0 for _ in range(52)]

    for feat_name, feat_exists in row.items():

        if feat_exists:

            st_num = station_number(feat_name)

            combin[st_num] = 1



    # make into a string

    combin_str = ''

    for val in combin: combin_str += str(val)

    return combin_str





def write_csv(streamobj, data):

    for row in data:

        line = str(row[0])

        for word in row[1:]: line += ',' + str(word)

        streamobj.write(line + '\n')
num_path = '../input/train_numeric.csv'

cat_path = '../input/train_categorical.csv'



NROWS = 550000 # total number of rows to read

CHUNK = 10000

uniq_combinations = set()



all_ids = []

all_responses = []

all_combinations = []



lines_read = 0

for num_df, cat_df in zip(

    pd.read_csv(num_path, dtype=str, chunksize=CHUNK),

    pd.read_csv(cat_path, dtype=str, chunksize=CHUNK)

):

    all_ids.extend(num_df['Id'].values)

    all_responses.extend(num_df['Response'].values)

    

    num_df = num_df.notnull().drop(['Id', 'Response'], axis=1)

    cat_df = cat_df.notnull().drop(['Id'], axis=1)

    for numrow, catrow in zip(num_df.iterrows(), cat_df.iterrows()):       

        full_row = numrow[1].to_dict()

        full_row.update(catrow[1].to_dict())

        combination = get_combination(full_row)

        uniq_combinations.add(combination)

        all_combinations.append(combination)

        

    lines_read += CHUNK

    

    if lines_read % 100000 == 0: print("progress: {0}".format(lines_read))

    if lines_read >= NROWS: break



uniq_combinations = list(enumerate(sorted(uniq_combinations)))

print("found {0} combinations".format(len(uniq_combinations)))



        

with open("enumerated_combinations.csv", 'w') as resfile:

    resfile.write("combid,combination\n")

    write_csv(resfile, uniq_combinations)

    

with open("id-comb-res.csv", 'w') as resfile:

    resfile.write("Id,combination,Response\n")

    write_csv(resfile, zip(all_ids, all_combinations, all_responses))
test_df = pd.read_csv("enumerated_combinations.csv")

print(test_df.head(5))



test_df = pd.read_csv("id-comb-res.csv")

print(test_df.head(5))
enum_comb_path = "enumerated_combinations.csv"

id_comb_res_path = "id-comb-res.csv"



SIGNIF = 100 # significance threshold



# create dictionary {combination: combid} e.g. {'1110100110010000000000000000011101110100000000000000': 6779}

combid_dic = pd.read_csv(enum_comb_path, index_col=1).to_dict()['combid']

data_df = pd.read_csv(id_comb_res_path, dtype=str)



data_df['combid'] = data_df['combination'].apply(lambda comb_str: combid_dic[comb_str])



responses = data_df['Response'].astype(float).values

combids = data_df['combid'].values





# PROCESSING DATA

#############################################################################

from operator import itemgetter

from itertools import groupby



fail_combids = [combid for combid, resp in zip(combids, responses) if resp > 0]



all_combids = sorted(combids)

fail_combids = sorted(fail_combids)



all_counts		= [(combid, len(list(group))) for combid, group in groupby(all_combids)]

failure_counts	= [(combid, len(list(group))) for combid, group in groupby(fail_combids)]



failure_counts = dict(failure_counts)



# throw away rare insignificant combids:

significant_counts = list(filter(lambda combid_count: combid_count[1] >= SIGNIF, all_counts))



failure_rates_sig = [(combid, failure_counts.get(combid, 0) / (1.*num_samples))\

                     for combid, num_samples in significant_counts]



failure_rates_all = [(combid, failure_counts.get(combid, 0) / (1.*num_samples))\

                     for combid, num_samples in all_counts]

#############################################################################







# let's write down failure rates for all significant combids sorted from high to low

failure_rates_sig = sorted(failure_rates_sig, key=itemgetter(1), reverse=True)

with open('failure_rates.csv', 'w') as resfile:

    resfile.write("combid,failrate\n")

    write_csv(resfile, failure_rates_sig)

    

# let's write down sample counts for all combids

all_counts = sorted(all_counts, key=itemgetter(1), reverse=True)

with open('sample_counts.csv', 'w') as resfile:

    resfile.write("combid,count\n")

    write_csv(resfile, all_counts)

    

print("{0} points total".format(len(all_counts)))

print("{0} significant points".format(len(significant_counts)))
test_df = pd.read_csv("failure_rates.csv")

print(test_df.head(5))



test_df = pd.read_csv("sample_counts.csv")

print(test_df.head(5))
# PLOTTING

#############################################################################

from matplotlib import pyplot as plt

import math



failure_rates_all	= np.array(failure_rates_all)

failure_rates_sig	= np.array(failure_rates_sig)

all_counts			= np.array(all_counts)



# plot failrates only for significant combids

fig2, ax2 = plt.subplots(figsize=(16,8))

ax2.bar(failure_rates_sig[:,0], failure_rates_sig[:,1], facecolor='red',   edgecolor='red')

ax2.set_ylabel("failure rate")

ax2.set_xlabel("combid")

fig2.savefig("failures-vs-combids.png")





# plot numbers of samples for each combid

veclog10 = np.vectorize(lambda x: math.log(x, 10))



fig3, ax3 = plt.subplots(figsize=(16,8))

ax3.bar(all_counts[:,0], veclog10(all_counts[:,1]), facecolor='black', edgecolor='black')

ax3.set_xlabel('combid')

ax3.set_ylabel('$log_{10}$(number of samples)')

fig3.savefig("logcounts-vs-combids.png")

#############################################################################