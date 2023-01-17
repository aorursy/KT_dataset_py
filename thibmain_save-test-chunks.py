# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import pandas as pd


def save_test_chunks():
    """
    Load all the test set chunk by chunk, and dump to individual pickled files
    This requires an extra 20 GB of storage and allows for easier manipulation of the test set:
    - Faster load time
    - Parallelizable over chunks
    """

    # Create a new directory for the test chunks
    os.makedirs("test_chunks", exist_ok=True)

    # There are 453653105 lines in the test_set.csv
    # We will create ~ 200 smaller, pickled files for easy access
    chunksize = 453653105 // 200

    reader = pd.read_table("../input/test_set.csv", sep=",", chunksize=chunksize)

    # Get the first chunk
    df_temp = next(reader)
    for chunk_idx, df_chunk in enumerate(reader):
        # Find out if df_chunk has an overlap with df_temp

        last_id = df_temp.object_id.values[-1]
        first_id = df_chunk.object_id.values[0]

        if last_id == first_id:
            df_temp = pd.concat(
                [df_temp, df_chunk[df_chunk.object_id == last_id]]
            ).reset_index(drop=True)

        df_temp.to_pickle(f"test_chunks/test_set_chunk{chunk_idx}.pickle")

        df_temp = df_chunk[df_chunk.object_id != last_id].reset_index(drop=True)

    # Dump the last chunk
    df_temp.to_pickle(f"test_chunks/test_set_chunk{chunk_idx + 1}.pickle")