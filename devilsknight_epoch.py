%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
!mkdir ../files
!cp -r ../input ../files/
# !cp -r ../input/train\ data/Train\ data/leukocoria/* ../files/input/train\ data/Train\ data/leukocoria/
path = Path('../files/input/train data/Train data')

test = Path('../files/input/evaluation data/Evaluation data/')
import os

import shutil

src=path/'leukocoria'

src_files = os.listdir(src)

for file_name in src_files:

    full_file_name = os.path.join(src, file_name)

    copy_file_name = os.path.join(src, 'copy'+file_name)

    if (os.path.isfile(full_file_name)):

        shutil.copy(full_file_name, copy_file_name)
print(len([name for name in os.listdir(path/'leukocoria')]))
import os

import shutil

src=path/'leukocoria'

src_files = os.listdir(src)

for file_name in src_files:

    full_file_name = os.path.join(src, file_name)

    copy_file_name = os.path.join(src, 'copy1'+file_name)

    if (os.path.isfile(full_file_name)):

        shutil.copy(full_file_name, copy_file_name)
print(len([name for name in os.listdir(path/'leukocoria')]))
path.ls()
np.random.seed(42)

data = (ImageList.from_folder(path) 

        .split_by_rand_pct(.2)             

        .label_from_folder()            

        .add_test(ImageList.from_folder(test))

        .transform(get_transforms(),size=224)

        .databunch(bs=bs, num_workers=0)

        .normalize(imagenet_stats)) 
# data.test_ds.x.items
# data.classes
# data.show_batch(rows=3, figsize=(7,8))
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
with open('submission0.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(5e-5,3e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
with open('submission1.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])
learn.save('stage-2')
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)