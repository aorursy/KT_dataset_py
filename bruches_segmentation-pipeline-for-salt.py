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
!pip install segmentation_pipeline
!pip uninstall -y imgaug
!pip install git+https://github.com/aleju/imgaug
!pip install shapely
from segmentation_pipeline.impl.datasets import PredictionItem
import os
from segmentation_pipeline.impl import rle
import imageio
import pandas as pd

class SegmentationRLE:

    def __init__(self,path,imgPath):
        self.data=pd.read_csv(path);
        self.values=self.data.values;
        self.imgPath=imgPath;
        self.ship_groups=self.data.groupby('id');
        self.masks=self.ship_groups['id'];
        self.ids=list(self.ship_groups.groups.keys())
        pass
    
    def __len__(self):
        return len(self.masks)


    def __getitem__(self, item):
        pixels=self.ship_groups.get_group(self.ids[item])["rle_mask"]
        return PredictionItem(self.ids[item] + '.png', imageio.imread(os.path.join(self.imgPath,self.ids[item]+'.png')),
                              rle.masks_as_image(pixels, shape=(101,101)) > 0.5)
    
    def isPositive(self, item):
        pixels=self.ship_groups.get_group(self.ids[item])["rle_mask"]
        for mask in pixels:
            if isinstance(mask, str):
                return True;
        return False
CSV_PATH = '../input/train.csv'
IMG_PATH = '../input/train/images/'
dataset = SegmentationRLE(CSV_PATH, IMG_PATH)
print(dataset[0].x.shape)
print(dataset[1].y.shape)
!wget https://raw.githubusercontent.com/BruchesLena/DataSets/master/salt.yaml
from segmentation_pipeline.impl.datasets import PredictionItem
import segmentation_pipeline.impl.datasets
from segmentation_pipeline import segmentation
from segmentation_pipeline.impl.datasets import  SimplePNGMaskDataSet

segmentation_pipeline.impl.datasets.AUGMENTER_QUEUE_LIMIT=1
cfg = segmentation.parse("salt.yaml")
cfg.verbose = 2
cfg.fit(dataset)
import numpy as np

val_accs = []
for i in range(5):
  metric_file = 'metrics/metrics-'+str(i)+'.0.csv'
  metrics = pd.read_csv(metric_file)
  acc = list(metrics['val_binary_accuracy'])
  val_accs.append(acc[-5])
best_fold = np.argmax(np.array(val_accs))
from segmentation_pipeline import  segmentation
from segmentation_pipeline.impl.rle import rle_encode
from skimage.morphology import remove_small_objects, remove_small_holes
import pandas as pd

#this is our callback which is called for every image
def onPredict(file_name, img, data):
    threshold = 0.25
    predictions = data["pred"]
    imgs = data["images"]
    post_img = remove_small_holes(remove_small_objects(img.arr > threshold))
    rle = rle_encode(post_img)
    predictions.append(rle)
    imgs.append(file_name[:file_name.index(".")])
    pass

predictions = []
images = []
cfg.predict_in_directory("../input/test/images/", best_fold, 0, onPredict, {"pred": predictions, "images": images})

df = pd.DataFrame.from_dict({'id': images, 'rle_mask': predictions})
df.to_csv('submission_best_fold.csv', index=False)
from segmentation_pipeline import  segmentation
from segmentation_pipeline.impl.rle import rle_encode
from skimage.morphology import remove_small_objects, remove_small_holes
import pandas as pd

def onPredict(file_name, img, data):
    threshold = 0.25
    predictions = data["pred"]
    imgs = data["images"]
    post_img = remove_small_holes(remove_small_objects(img.arr > threshold))
    rle = rle_encode(post_img)
    predictions.append(rle)
    imgs.append(file_name[:file_name.index(".")])
    pass

predictions = []
images = []
cfg.predict_in_directory("../input/test/images/", [0,1,2,3,4], 0, onPredict, {"pred": predictions, "images": images})

df = pd.DataFrame.from_dict({'id': images, 'rle_mask': predictions})
df.to_csv('submission_all.csv', index=False)