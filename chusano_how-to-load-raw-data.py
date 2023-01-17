import pickle

import os
def get_pickle(path, name):

    with open(path+name, 'rb') as handle:

        data = pickle.load(handle)

    return data
city = "gijon"

reviews = get_pickle("/kaggle/input/tripadvisor-image-restaurant/tripadimgrest_raw_"+city+"/",city+".pkl")