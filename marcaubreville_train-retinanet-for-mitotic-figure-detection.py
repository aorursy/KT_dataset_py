%reload_ext autoreload

%autoreload 2

%matplotlib inline

from pathlib import Path

from fastai import *

from fastai.vision import *

from fastai.callbacks import *



"""

         Own libraries

"""

from sliderunnerdatabase import Database

from objectdetectiontools import get_slides, PascalVOCMetric, create_anchors,ObjectItemListSlide, SlideObjectCategoryList, bb_pad_collate_min, show_anchors_on_images, slide_object_result 

from retinanet import RetinaNet,RetinaNetFocalLoss
path = Path('/kaggle/input/mitosis-wsi-ccmct-training-set/')



database = Database()

database.open(str(path/'MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite'))





getslides = """SELECT filename FROM Slides"""

all_slides = database.execute(getslides).fetchall()



lbl_bbox, train_slides,val_slides,files = get_slides(slidelist_test=[1,2,3,4,5,6,7,8,9,10,11,12,13,14], size=512, positive_class=2, negative_class=7, database=database,basepath=str(path))

            
bs = 16

train_images = 5000

val_images = 5000

size=512



img2bbox = dict(zip(files, np.array(lbl_bbox)))

get_y_func = lambda o:img2bbox[o]



tfms = get_transforms(do_flip=True,

                      flip_vert=True,

                      max_rotate=90,

                      max_lighting=0.0,

                      max_zoom=1.,

                      max_warp=0.0,

                      p_affine=0.5,

                      p_lighting=0.0,

                      #xtra_tfms=xtra_tfms,

                     )

train_files = list(np.random.choice([files[x] for x in train_slides], train_images))

valid_files = list(np.random.choice([files[x] for x in val_slides], val_images))





train =  ObjectItemListSlide(train_files, path=path)

valid = ObjectItemListSlide(valid_files, path=path)

valid = ObjectItemListSlide(valid_files, path=path)

item_list = ItemLists(path, train, valid)

lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryList) #

lls = lls.transform(tfms, tfm_y=True, size=size)

data = lls.databunch(bs=bs, collate_fn=bb_pad_collate_min, num_workers=4).normalize()
data.show_batch(rows=2, ds_type=DatasetType.Train, figsize=(15,15))
anchors = create_anchors(sizes=[(32,32)], ratios=[1], scales=[0.6, 0.7,0.8])

not_found = show_anchors_on_images(data, anchors)
crit = RetinaNetFocalLoss(anchors)

encoder = create_body(models.resnet18, True, -2)

model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=3, sizes=[32], chs=128, final_bias=-4., n_conv=3)





voc = PascalVOCMetric(anchors, size, [str(i-1) for i in data.train_ds.y.classes[1:]])

learn = Learner(data, model, loss_func=crit, callback_fns=[ShowGraph], #BBMetrics, ShowGraph

                metrics=[voc]

               )
learn.split([model.encoder[6], model.c5top5])

learn.freeze_to(-2)

learn.model_dir='/kaggle/working/'



learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-4)
slide_object_result(learn, anchors, detect_thresh=0.3, nms_thresh=0.2, image_count=10)
learn.unfreeze()
learn.fit_one_cycle(10,1e-4)
slide_object_result(learn, anchors, detect_thresh=0.3, nms_thresh=0.2, image_count=10)
learn.export('/kaggle/working/RetinaNetMitoticFigures')