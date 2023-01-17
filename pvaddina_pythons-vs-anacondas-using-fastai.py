from fastai import *

from fastai.vision import *
path = Path('../input/anacondas_pythons')

path.ls()
predict_python = get_image_files(path/"valid/python")

predict_anaconda = get_image_files(path/"valid/anaconda")
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224, bs = 8)

data.normalize(imagenet_stats)
DatasetType.Train
data.show_batch(rows=3)
print(data.classes)
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/models")
learn.fit_one_cycle(10)
learn.save("stage-1")
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9)
interp.plot_confusion_matrix()
interp.most_confused()
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
def do_prediction(files, expected):

    p = []

    for f in files:

        p_img = open_image(f)

        pred_class,pred_idx,outputs = learn.predict(p_img)

        if str(pred_class) != expected:

            p.append(p_img)

    return p
wrong_anaconda_pred = do_prediction(predict_anaconda, "anaconda")

wrong_python_pred = do_prediction(predict_python, "python")
for f in wrong_anaconda_pred:

    show_image(f)
for f in wrong_python_pred:

    show_image(f)