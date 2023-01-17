%matplotlib inline
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import mark_boundaries
import cv2
DATA_DIR = os.path.join('..', 'input')
class_str = """car, 33
motorbicycle, 34
bicycle, 35
person, 36
rider, 37
truck, 38
bus, 39
tricycle, 40
others, 0
rover, 1
sky, 17
car_groups, 161
motorbicycle_group, 162
bicycle_group, 163
person_group, 164
rider_group, 165
truck_group, 166
bus_group, 167
tricycle_group, 168
road, 49
siderwalk, 50
traffic_cone, 65
road_pile, 66
fence, 67
traffic_light, 81
pole, 82
traffic_sign, 83
wall, 84
dustbin, 85
billboard, 86
building, 97
bridge, 98
tunnel, 99
overpass, 100
vegatation, 113
unlabeled, 255"""
class_dict = {v.split(', ')[0]: int(v.split(', ')[-1]) for v in class_str.split('\n')}
all_paths = pd.DataFrame(dict(path = glob(os.path.join(DATA_DIR, '*', '*.*p*g'))))
classdict = {0:'others', 1:'rover', 17:'sky', 33:'car', 34:'motorbicycle', 35:'bicycle', 36:'person', 37:'rider', 38:'truck', 39:'bus', 40:'tricycle', 49:'road', 50:'siderwalk', 65:'traffic_cone'}
all_paths['split'] = all_paths['path'].map(lambda x: x.split('/')[-2].split('_')[0])
all_paths['group'] = all_paths['path'].map(lambda x: x.split('/')[-2].split('_')[-1])
all_paths['group'] = all_paths['group'].map(lambda x: 'color' if x == 'test' else x)
all_paths['id'] = all_paths['path'].map(lambda x: '_'.join(os.path.splitext(os.path.basename(x))[0].split('_')[:4]))
all_paths.sample(5)
group_df = all_paths.pivot_table(values = 'path', columns = 'group', aggfunc = 'first', index = ['id', 'split']).reset_index()
group_df.sample(5)
classifiers = {}
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

class PersonDetector():
    def __init__(self):
        # make it picklable
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    def detect(self, gray_img, run_filter):
        c_found, _ = self.hog.detectMultiScale(gray_img, winStride=(8,8), padding=(32,32), scale=1.05)
        if run_filter:
            for ri, r in enumerate(c_found):
                for qi, q in enumerate(c_found):
                    if ri != qi and inside(r, q):
                        break
                else:
                    found_filtered.append(r)
            return found_filtered.tolist()
        else:
            try:
                return c_found.tolist()
            except AttributeError:
                # sometimes opencv returns empty tuples
                return []
        
classifiers['person'] = PersonDetector()
# ensure empty case works well
PersonDetector().detect(np.zeros((128, 128), dtype = np.uint8), False)
def apply_classifier(in_path, all_classes, run_filter = False, debug_mode = False):
    im = cv2.imread(in_path)
    gr_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    out_segs = np.zeros(gr_im.shape, dtype = np.uint16)
    found = []
    for c_label, c_class in all_classes.items():
        c_found = c_class.detect(gr_im, run_filter = run_filter)
        found += c_found
        thickness = 4
        for i, (x, y, w, h) in enumerate(c_found):
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            if debug_mode:
                cv2.rectangle(im, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
            out_segs[(y+pad_h):(y+h-pad_h), (x+pad_w):(x+w-pad_w)] = class_dict[c_label]*1000+i
    if debug_mode:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), out_segs
    else:
        return out_segs
test_path = '../input/train_color/171206_033232166_Camera_5.jpg'
nrm_img, _ = apply_classifier(test_path, {}, debug_mode = True)
%timeit _ = apply_classifier(test_path, classifiers, debug_mode = False)
cls_img, seg_img = apply_classifier(test_path, classifiers, debug_mode = True)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 8))
ax1.imshow(nrm_img)
ax2.imshow(cls_img[1501:1963, 844:1936])
ax2.set_title('Person Detection ({})'.format(len(np.unique(seg_img))-1));
ax3.imshow(seg_img[1501:1963, 844:1936])
ax3.set_title('Segment Output');
train_df = group_df.query('split=="train"')
print(train_df.shape[0], 'rows')
sample_rows = 10
fig, m_axs = plt.subplots(sample_rows, 5, figsize = (20, 6*sample_rows))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
out_rows = []
for (ax1, ax2, ax4, ax3, ax_c_crop), (_, c_row) in zip(m_axs, train_df.sample(sample_rows, random_state = 2018).iterrows()):
    c_img = imread(c_row['color'])
    l_img = imread(c_row['label'])//1000
    seg_img = apply_classifier(c_row['color'], classifiers)
    ax1.imshow(c_img)
    ax1.set_title('Color')
    ax2.imshow(l_img, cmap = 'nipy_spectral')
    ax2.set_title('Segments')
    xd, yd = np.where(l_img>0)
    bound_img = mark_boundaries(image = c_img, label_img = l_img, color = (1,0,0), background_label = 255, mode = 'thick')
    ax3.imshow(bound_img[xd.min():xd.max(), yd.min():yd.max(),:])
    ax3.set_title('Cropped Overlay')
    ax4.imshow(cls_img)
    ax4.set_title('HOG Image %d objects\n Accuracy: %2.3f%%\n Non-zero Accuracy %2.2f%%' % (np.max(seg_img) % 1000, 
                                                                                            100*np.mean((l_img>0)==(seg_img>0)),
                                                                                           100*np.mean((seg_img[l_img>0]>0))))
    ax_c_crop.imshow(seg_img[xd.min():xd.max(), yd.min():yd.max()])
    ax_c_crop.set_title('Cropped HOG')
fig.savefig('sample_overview.png')
test_df = group_df.query('split=="test"').drop(['label'], axis = 1)
print(test_df.shape[0], 'rows')
test_df.sample(3)
def rle_encoding(x):
    """ Run-length encoding based on
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    Modified by Konstantin, https://www.kaggle.com/lopuhin
    """
    assert x.dtype == np.bool
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.append([b, 0])
        run_lengths[-1][1] += 1
        prev = b
    return '|'.join('{} {}'.format(*pair) for pair in run_lengths)

def segs_to_rle_rows(lab_img, **kwargs):
    out_rows = []
    for i in np.unique(lab_img[lab_img>0]):
        c_dict = dict(**kwargs)
        c_dict['LabelId'] = i//1000
        c_dict['PixelCount'] = np.sum(lab_img==i)
        c_dict['Confidence'] = 0.5 # our classifier isnt very good so lets not put the confidence too high
        c_dict['EncodedPixels'] = rle_encoding(lab_img==i)
        out_rows += [c_dict]
    return out_rows
# make sure it works on a simple case
pd.DataFrame(segs_to_rle_rows(seg_img, ImageId = -1))
def read_row(in_row):
    c_segs = apply_classifier(in_row['color'], classifiers)
    return segs_to_rle_rows(c_segs, ImageId = in_row['id'])
%%time
from tqdm import tqdm_notebook
all_rows = []
for _, c_row in tqdm_notebook(list(test_df.sample(10).iterrows())):
    all_rows += read_row(c_row.to_dict())
%%time
all_rows = []
for _, c_row in tqdm_notebook(list(test_df.iterrows())):
    all_rows += read_row(c_row.to_dict())
all_rows_df = pd.DataFrame(all_rows)
all_rows_df = all_rows_df[['ImageId', 'LabelId', 'PixelCount', 'Confidence', 'EncodedPixels']]
all_rows_df.to_csv('opencv_full_submission.csv', index = False)
all_rows_df.sample(5)
