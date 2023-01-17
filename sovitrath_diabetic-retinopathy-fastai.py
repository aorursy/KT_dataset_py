from fastai.vision import *

import os
os.makedirs('/root/.cache/torch/checkpoints')
!cp ../input/resnet34fastai/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth
model_path = 'models'

plot_path = 'plots'



if not os.path.exists(model_path):

    os.makedirs(model_path)

    os.makedirs(os.path.join(model_path, plot_path))
'''

Severity Levels



0 - 'No_DR',

1 - 'Mild',

2 - 'Moderate',

3 - 'Severe',

4 - 'Proliferate_DR'

'''



classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
path = Path('../input/diabetic-retinopathy-224x224-2019-data/colored_images')

path.ls()
# remove the images that we cannot open

# for c in classes:

#     print(c)

#     verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, 

                                  ds_tfms=get_transforms(), size=224, 

                                  num_workers=4, bs=16).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(10, 7))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/kaggle/working/models')

# learner = load('../input/resnet34-fastai/resnet34.pth')

# learn = cnn_learner(data, learner, metrics=error_rate)
learn.fit_one_cycle(20)    
learn.recorder.plot_losses()
!pwd

print(os.listdir('../../'))
learn.save('colored_stage1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-5))
learn.save('colored_stage2')
learn.load('colored_stage2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
!pwd
learn.export('/kaggle/working/models/colored_export.pkl')