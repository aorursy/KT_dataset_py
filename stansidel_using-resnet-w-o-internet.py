%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
input_path = Path(".").absolute().parent / "input"

resnet34_path = input_path / "resnet34fastai/resnet34.pth"

data_path = input_path / "bears-fastai-course"
np.random.seed(42)

data = ImageDataBunch.from_folder(data_path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, bs=16).normalize(imagenet_stats)
models_path = Path('./models')

models_path.mkdir(exist_ok=True)



import shutil



my_file = pathlib.Path('/etc/hosts')

to_file = pathlib.Path('/tmp/foo')

shutil.copy(str(resnet34_path), str(models_path))
learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False, model_dir=str(models_path.absolute()))
learn.load('resnet34');
learn.load('resnet34', strict=False);
learn.fit_one_cycle(1)
learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir=str(models_path.absolute()))
learn.fit_one_cycle(1)
learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False, model_dir=str(models_path.absolute()))
learn.fit_one_cycle(1)
!cp -r /kaggle/input/resnet34fastai/ /kaggle/working/models/
model = create_cnn(data, models.resnet34, metrics=error_rate, path=".", pretrained=False, model_dir='/kaggle/working/models/resnet34fastai')
model.fit_one_cycle(1)
!ls -lah /kaggle/input/resnet34fastai/

!ls -lah /kaggle/working/models/resnet34fastai

!ls -lah /kaggle/working/models/resnet34fastai/resnet34