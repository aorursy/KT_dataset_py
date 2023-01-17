from fastai.vision import *

import os
# this is where we will copy our pretrained models

import os

os.makedirs('/root/.cache/torch/checkpoints')
!cp ../input/resnet34/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth
# to save the models

model_path = 'models'

# to save the plots

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
path = Path('../input/diabetic-retinopathy-2015-data-colored-resized/colored_images/colored_images/')

path.ls()
'''

Remove the images that we cannot open. 

Execute this only once per kernel run.

'''

for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, 

                                  ds_tfms=get_transforms(), size=224, 

                                  num_workers=4, bs=32).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(10, 7))
learn = cnn_learner(data, models.resnet34, 

                    metrics=error_rate, 

                    model_dir='/kaggle/working/models')
learn.fit_one_cycle(20)    
learn.recorder.plot_losses()

plt.savefig('models/plots/interp1.png')
learn.save('colored_stage1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

plt.savefig('models/plots/interp2.png')
learn.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4))
learn.save('colored_stage2')
learn.load('colored_stage2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

plt.savefig('models/plots/interp.png')

plt.show()