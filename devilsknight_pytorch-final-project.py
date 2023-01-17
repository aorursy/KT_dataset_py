!mkdir ../input/new
!cp -r ../input/flower-data/flower_data/flower_data/train ../input/new
!cp -r ../input/flower-data/flower_data/flower_data/valid ../input/new
!cp -r ../input/flowers-missing-test-set/test_flowers_detection_pytorch_challenge/test ../input/new
from fastai.vision import *
path = Path('../input/new')
np.random.seed(42)
data = ImageDataBunch.from_folder(path,
        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
data.classes
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8, max_lr=slice(1e-4,1e-3))
