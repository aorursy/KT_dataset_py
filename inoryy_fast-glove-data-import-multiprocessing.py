import numpy as np

from multiprocessing import Pool



num_cpu = 4

embed_size = 300

glove_file_path = '../input/glove840b300dtxt/glove.840B.300d.txt'
def get_coefs(row):

    row = row.strip().split()

    # can't use row[0], row[1:] split because 840B contains multi-part words 

    word, arr = " ".join(row[:-embed_size]), row[-embed_size:]

    return word, np.asarray(arr, dtype='float32')
def get_glove():

    return dict(get_coefs(row) for row in open(glove_file_path))
def get_glove_fast():

    pool = Pool(num_cpu)

    with open(glove_file_path) as glove_file:

        return dict(pool.map(get_coefs, glove_file, num_cpu))
# Time for sequential data import

%time glove1 = get_glove()
# Time for multiprocessing data import

%time glove2 = get_glove_fast()
assert len(glove1) == len(glove2)