from fastai.vision import *
import fastai; fastai.__version__
import numpy as np
# path = untar_data(URLs.MNIST_SAMPLE)
# data = ImageDataBunch.from_folder(path)
# learn = cnn_learner(data, models.resnet18, metrics=accuracy)
# learn.fit(1)
# def verify_images():
#     for folder in ('cheetahs', 'jaguars', 'leopards','panthers','tigers'):
#         print(folder)
#         verify_images(os.path.join(path, folder), delete=True, max_size=700)
    
# We can verify that we don’t have any corrupt images using the verify_images method.
import os
# model_dir= ("/tmp/model/")
path = Path('../input/1243565/cats')
for folder in ('cheetahs', 'jaguars', 'leopards','panthers','tigers'):
        print(folder)
        verify_images(os.path.join(path, folder), delete=True, max_size=2900)
# # We can verify that we don’t have any corrupt images using the verify_images method.
# import os
# # model_dir= ("/tmp/model/")
# path = Path('../input/cats-232/cats')
# try:
#     for folder in ('cheetahs', 'jaguars', 'leopards','panthers','tigers'):
#         print(folder)
#         verify_images(os.path.join(path, folder), delete=True, max_size=700)
# except OSError:
#     print("Ima korumpiranih slika")
# else:
#     print('Uspesno smo prosli kroz sve')

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=path, valid_pct=0.33,
                                  ds_tfms=get_transforms(),bs=15 ,size=500, num_workers=0).normalize(imagenet_stats)
# Setting num_workers=0 prevents crushing!
data.classes
data.show_batch(rows=5, figsize=(7, 8))
# from fastai.metrics import error_rate # 1 - accuracy
# learn = create_cnn(data, models.resnet34, metrics=accuracy)
# # The created model uses the resnet34 architecture, with weights pretrained on the imagenet dataset.
# # By default, only the fully connected layers at the top are unfrozen (can be trained), 
# # which if you are familiar with transfer learning makes perfect sense.
# here we are using resnet 50 which is using 50 layers insted of 34
from fastai.metrics import error_rate # 1 - accuracy
learn = create_cnn(data, models.resnet50, metrics=accuracy)
# The created model uses the resnet34 architecture, with weights pretrained on the imagenet dataset.
# By default, only the fully connected layers at the top are unfrozen (can be trained), 
# which if you are familiar with transfer learning makes perfect sense.
# defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(10)
# Will take too long without GPU
learn.model_dir='/kaggle/working/'
learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-4))
learn.save('..animal-detection-stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(6, figsize=(15,15))
# Show images in `top_losses` along with their prediction, actual, loss, and probability of actual class.
from fastai.widgets import *

losses, idxs = interp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]

# this one is not working. Should try ImageCleaner isnted. 
# fd = FileDeleter(file_paths=top_loss_paths)

learn.summary()
learn.load('..animal-detection-stage-1') # loading the weights
# learn.data = db # replacing the data

# learn.freeze()
# learn.fit_one_cycle(4)

# learn.unfreeze()

# learn.lr_find()
# learn.recorder.plot()

# learn.fit_one_cycle(4, max_lr=slice(3e-5, 3e-4))
# learn.save('animal-detection-stage-2')
# learn.freeze()
# learn.fit_one_cycle(4)
# learn.lr_find()
# learn.recorder.plot()
# learn.fit_one_cycle(4, max_lr=slice(1e-5, 1e-6))
# learn.save('animal-detection-stage-2')
