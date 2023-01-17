# importamos librerias
import zipfile
from fastai.vision import *
warnings.simplefilter("ignore")
path = Path('../input/dogs-vs-cats-redux-kernels-edition/')
path.ls()
TRAIN_ZIP = path / 'train.zip'
with zipfile.ZipFile(TRAIN_ZIP, 'r') as z: z.extractall('data')
data = (ImageList.from_folder('data')
                 .split_by_rand_pct(0.2)
                 .label_from_func(lambda x: x.stem.split('.')[0])
                 .transform(get_transforms(), size=224)
                 .databunch(bs=32)
                 .normalize())
data
data.show_batch(3, figsize=(8,8))
# Examinemos la data
xb,yb = data.one_batch()
xb.shape
# [batch_size, channels (RGB), ancho, alto]
yb
# Tenemos 1 label por cada imagen
data.c2i
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(3, 1e-3)
learn.show_results(DatasetType.Train, 4)
interp = learn.interpret()
interp.plot_confusion_matrix()
interp.plot_top_losses(9)
interp.plot_top_losses(9, largest=False, heatmap=True)
interp.plot_top_losses(9, heatmap=True)

