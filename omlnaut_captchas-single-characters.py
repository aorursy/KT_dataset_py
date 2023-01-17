from fastai.vision import *

import os



path = Path("../input/samples/samples")

print(os.listdir(path)[:10])
def plot_lr(learn):

    lr_find(learn)

    learn.recorder.plot()
def char_from_path(path, position):

    return path.name[position]
data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders

        .split_by_rand_pct(0.2)              #How to split in train/valid? -> use the folders

        .label_from_func(partial(char_from_path, position=0))            #How to label? -> depending on the folder of the filenames

        .transform(get_transforms(do_flip=False))       #Data augmentation? -> use tfms with a size of 64

        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch
data.show_batch(3, figsize=(10,10))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/tmp', ps=0.)
plot_lr(learn)
lr = 5e-2

learn.fit_one_cycle(5, lr)
learn.save('pretrained')
learn.load('pretrained')

learn.unfreeze()
plot_lr(learn)
learn.fit_one_cycle(15, slice(5e-4, lr/5))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(7,7))
interp.plot_top_losses(4, heatmap_thresh=14, largest=False)
def data_from_position(position):

    data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders

        .split_by_rand_pct(0.2)              #How to split in train/valid? -> use the folders

        .label_from_func(partial(char_from_path, position=position))            #How to label? -> depending on the folder of the filenames

        .transform(get_transforms(do_flip=False))       #Data augmentation? -> use tfms with a size of 64

        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch

    return data
learners = []

for i in range(5):

    data = data_from_position(i)

    

    learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/tmp', ps=0.)

    

    lr = 5e-2

    learn.fit_one_cycle(5, lr)

    

    learn.unfreeze()

    learn.fit_one_cycle(15, slice(5e-4, lr/5))

    

    learners.append(learn)
figures = []

for learner in learners:

    figures.append(learner.interpret().plot_top_losses(4, heatmap_thresh=14, figsize=(8,8), largest=False, return_fig=True))
for e,f in enumerate(figures):

    f.suptitle('')

    for a in f.axes: a.set_title(f'Position: {e+1}')

    f.savefig(f'{e}_heatmap.png', bbox_inches='tight')
def predict_captcha(img, learners):

    return ''.join([str(learner.predict(img)[0]) for learner in learners])
fig, ax = plt.subplots(ncols=5, figsize=(20,10))

for a, (img, lbl) in zip(ax.flatten(), learners[0].data.valid_ds):

    show_image(img, a)

    a.set_title(f'predicted: {predict_captcha(img, learners)}')

plt.show()
img_paths = learners[0].data.valid_ds.items

count = 0

correct = 0



for img_path in img_paths:

    lbl = img_path.name[:-4]

    img = open_image(img_path)

    predicted = predict_captcha(img, learners)

    if lbl==predicted: correct +=1

    count += 1

correct/count