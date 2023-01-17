import numpy as np

import pandas as pd 

import os 



def MinMaxBestBaseStacking(input_folder, best_base, output_path):

    sub_base = pd.read_csv(best_base)

    all_files = os.listdir(input_folder)



    # Read and concatenate submissions

    outs = [pd.read_csv(os.path.join(input_folder, f), index_col=0) for f in all_files]

    concat_sub = pd.concat(outs, axis=1)

    cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))

    concat_sub.columns = cols

    concat_sub.reset_index(inplace=True)



    # get the data fields ready for stacking

    concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)

    concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)

    concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)

    concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)



    # set up cutoff threshold for lower and upper bounds

    cutoff_lo = 0.85

    cutoff_hi = 0.17



    concat_sub['is_iceberg_base'] = sub_base['target']

    concat_sub['target'] = np.where(np.all(concat_sub.iloc[:, 1:6] > cutoff_lo, axis=1),

                                        concat_sub['is_iceberg_max'],

                                        np.where(np.all(concat_sub.iloc[:, 1:6] < cutoff_hi, axis=1),

                                                 concat_sub['is_iceberg_min'],

                                                 concat_sub['is_iceberg_base']))

    concat_sub[['image_name', 'target']].to_csv(output_path,

                                            index=False, float_format='%.12f')

MinMaxBestBaseStacking('../input/melanoma-ensemble-files/', '../input/melanoma-ensemble-files/blend_sub.csv', 'submission.csv')