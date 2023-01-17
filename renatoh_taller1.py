# importamos librerias
from fastai.vision import *
warnings.simplefilter("ignore")
path = Path('../input/medical-mnist/')
path.ls()
data = (ImageList.from_folder(path)
                 .split_by_rand_pct()
                 .label_from_folder()
                 .transform(get_transforms(), size=64)
                 .databunch(bs=64)
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
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(512,data.c) # data.c = numero de clases en el dataset
)

model
learn = Learner(data, model, metrics=accuracy)
learn.fit_one_cycle(3, 1e-3)
learn.show_results(DatasetType.Train, 4)
interp = learn.interpret()
interp.plot_confusion_matrix()
interp.plot_top_losses(9)


