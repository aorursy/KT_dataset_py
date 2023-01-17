%matplotlib inline
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
try:
    from skimage.util.montage import montage2d
except ImportError as e:
    print('scikit-image is too new',e)
    from skimage.util import montage as montage2d
def read_many_aff(in_paths):
    img_out, label_out = [], []
    for c_path in in_paths:
        a, b = read_affdata(c_path, verbose=False)
        img_out += [a]
        label_out += [b]
    return np.concatenate(img_out, 0), np.concatenate(label_out, 0)
def read_affdata(in_path, verbose=True):
    v = loadmat(in_path)['affNISTdata'][0][0]
    if verbose:
        for k in v:
            print(k.shape)
    img = v[2].reshape((40, 40, -1)).swapaxes(0, 2).swapaxes(1, 2)
    label = v[5][0]
    if verbose:
        plt.imshow(montage2d(img[:81]), cmap='bone')
    return img, label
valid_img_data, valid_img_label = read_affdata('../input/validation.mat')
test_img_data, test_img_label = read_affdata('../input/test.mat')
c_id = np.random.choice(list(range(valid_img_data.shape[0])))
plt.matshow(valid_img_data[c_id])
print(valid_img_label[c_id])
train_img_data, train_img_label = read_many_aff(glob('../input/training_batches/training_batches/*.mat'))
plt.hist(train_img_label)
print(train_img_data.shape, train_img_label.shape)
def rebalance_data(in_img, in_label, new_size = None, favor_classes=None, demote_classes=None):
    base_p = np.ones_like(in_label).astype('float32')
    if favor_classes is None:
        favor_classes=[]
    if demote_classes is None:
        demote_classes=[]
    if new_size is None:
        new_size = in_label.shape[0]
    
    for k in favor_classes:
        base_p[in_label==k] *= 2
    for k in demote_classes:
        base_p[in_label==k] /= 2
        
    base_p /= base_p.sum()
    new_idx = np.random.choice(np.arange(base_p.shape[0]), size=new_size, replace=True, p=base_p) 
    return in_img[new_idx], in_label[new_idx]
trn_img, trn_lab = rebalance_data(train_img_data, 
                                  train_img_label, 
                                  favor_classes=[7], 
                                  demote_classes=[1])
plt.hist(trn_lab, np.arange(11))
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage2d(trn_img[:400]), cmap='bone')
fig.savefig('full_res.png', dpi=200)
tst_img, tst_lab = rebalance_data(test_img_data, 
                                  test_img_label, 
                                  favor_classes=[1], 
                                  demote_classes=[7])
plt.hist(tst_lab, np.arange(11))
def write_to_disk(img_vec, lab_vec, base_name):
    assert img_vec.shape[0]==lab_vec.shape[0], "Shapes should match"
    idx = np.random.permutation(np.arange(img_vec.shape[0]))
    np.savez_compressed('{}.npz'.format(base_name),
                        img = img_vec,
                        idx = idx
                       )
    pd.DataFrame({'idx': idx,
                  'label': lab_vec
                 }).to_csv('{}_labels.csv'.format(base_name), index=False)
write_to_disk(trn_img, trn_lab, 'train')
write_to_disk(valid_img_data, valid_img_label, 'valid')
write_to_disk(tst_img, tst_lab, 'test')
!ls -lh *.npz
!ls -lh *.csv
test_df = pd.read_csv('test_labels.csv')
test_df['label'] = 5 # make every guess 5 
test_df.to_csv('sample_submission.csv', index=False)