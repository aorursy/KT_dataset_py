import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import operator


samples = pd.read_csv('../input/sample_submission.csv')
samples.sample(10)




#parameters
block_size = 2
k_size = 3
k =0.04
def featurize(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)
    dst = cv2.cornerHarris(img,block_size,k_size,k)
    return -1 * dst.mean()
    
def make_filename(set_id,day_id):
    return "../input/test_sm/set{0}_{1}.jpeg".format(set_id,day_id)

def reorder(set_id):
    order_day = { d : d / 10 for d in range(1,6)}
    for d in order_day:
        order_day[d] = featurize(make_filename(set_id, d ))
    ordered_day = sorted(order_day.items(), key=operator.itemgetter(1))
    return "{0} {1} {2} {3} {4}".format(*[d[0] for d in ordered_day])
    


        
samples['day'] = samples['setId'].map(reorder)
samples.sample(10)
samples.to_csv('naive_submit.csv', index=False)
