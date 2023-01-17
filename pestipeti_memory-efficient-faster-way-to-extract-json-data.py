import pandas as pd

import numpy as np

import json

import csv

import gc



from collections import OrderedDict

from tqdm import tqdm_notebook as tqdm



# dtypes for pd.read_csv

# These are help to reduce the memory usage.

DTYPES_RAW = {

    'event_id': 'object',

    'game_session': 'object',

    'installation_id': 'object',

    'event_count': np.uint16,

    'event_code': np.uint16,

    'game_time': np.uint32,

    'type': 'category',

    'world': 'category',

    'title': 'category',  

}
# Extract these arguments from JSON.

# There is not enough memory to extract everything with this method.

# You should try it whether it can process the private test set too

# with your selected arguments

FIELDS = {

    # Extras from JSON

    # If you add more data, do not forget

    # to add default values below.

    'level': np.uint8,

    'round': np.uint8,

    'correct': np.int8,

    'misses': np.int8,

    

    # Nested object separated by '_'

    # for example: {'coordinates': {'x': 12, 'y': 12}}

    # 'coordinates_x': np.uint16

    # 'coordinates_y': np.uint16

}



DTYPES = OrderedDict( (dt[0], (dt[1], i)) for i, dt in enumerate(FIELDS.items()))
# This only needs if you want to show a TQDM progress bar.

import subprocess



def file_len(fname):

    """Returns the number of lines in a file.

       @see: https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows

    """

    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 

                                              stderr=subprocess.PIPE)

    result, err = p.communicate()

    if p.returncode != 0:

        raise IOError(err)

    return int(result.strip().split()[0])+1
def flatten(dct, res, separator='_'):

    """Flatten a dictionary.

       @see: https://stackoverflow.com/a/34094630/4158850

    """

    queue = [('', dct)]



    while queue:

        prefix, d = queue.pop()

        for k, v in d.items():

            key = prefix + k

            if not isinstance(v, dict):

                if key in FIELDS.keys():

                    res[0][DTYPES[key][1]] = v

            else:

                queue.append((key + separator, v))



    return res



def records_from_json(fh, n_rows, event_ids_to_drop):

    """Yields the records from a file object."""

    rows = csv.reader(fh, delimiter=',')

    skip_header = next(rows)

    

    # define dtype for more memory-efficiency.

    dtype = dict(names=list(FIELDS.keys()), formats=list(FIELDS.values()))

    defrow = np.zeros((1,), dtype=dtype)



    for event_id, game_session, timestamp, event_data, installation_id, event_count, event_code, game_time, title, typ, world in tqdm(rows, total=n_rows):

        

        # It is more memory-efficient if we don't use the the train df's columns yet.

        row = defrow.copy()



        # Default (required because of the copy above) values for the extracted data

        # you can use np.nan too (in this case the dtype should be np.float64)

        row[0][DTYPES['level'][1]] = 0

        row[0][DTYPES['round'][1]] = 0

        row[0][DTYPES['correct'][1]] = -1

        row[0][DTYPES['misses'][1]] = -1



        if event_id not in event_ids_to_drop:

            row = flatten(json.loads(event_data), row)



        yield row[0]



def from_records(path, event_ids_to_drop):

    n_rows = file_len(path)

    with open(path) as fh:

        return pd.DataFrame.from_records(records_from_json(fh, n_rows, event_ids_to_drop))

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

specs.args = specs.args.apply(lambda x: json.loads(str(x)))

eventIdsToDrop = []



for _, spec in specs.iterrows():

    j = pd.io.json.json_normalize(spec.args)

    vals = j.loc[(j.name.isin(FIELDS.keys()))].name.values



    if len(vals) == 0:

        eventIdsToDrop += [spec.event_id]



set(eventIdsToDrop)

print(len(eventIdsToDrop))
extras_df = from_records('/kaggle/input/data-science-bowl-2019/train.csv', eventIdsToDrop)
train_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', parse_dates=['timestamp'], dtype=DTYPES_RAW, usecols=['timestamp'] + list(DTYPES_RAW.keys()))
train_df = train_df.merge(extras_df, left_index=True, right_index=True)

train_df.info()
train_df.to_csv('train_extras.csv', index=False)
del extras_df

del train_df

gc.collect()



%reset -f Out
extras_df = from_records('/kaggle/input/data-science-bowl-2019/test.csv', eventIdsToDrop)
test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', parse_dates=['timestamp'], dtype=DTYPES_RAW, usecols=['timestamp'] + list(DTYPES_RAW.keys()))

test_df = test_df.merge(extras_df, left_index=True, right_index=True)



test_df.to_csv('test_extras.csv', index=False)
del extras_df

del test_df

gc.collect()



%reset -f Out