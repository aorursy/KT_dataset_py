from fastai.vision import *
path = Path('../input/flowers/flowers')
path.ls()
get_image_files(path/'daisy')[:5]
bs = 64

size = 224

num_workers = 0  # set this to 0 to prevent kernel from crashing
tfms = get_transforms()                               #Do standard data augmentation

data = (ImageItemList.from_folder(path)               #Get data from path

        .random_split_by_pct()                        #Randomly separate 20% of data for validation set

        .label_from_folder()                          #Label based on dir names

        .transform(tfms, size=size)                   #Pass in data augmentation

        .databunch(bs=bs, num_workers=num_workers)    #Create ImageDataBunch

        .normalize(imagenet_stats))                   #Normalize using imagenet stats
print(len(data.train_ds))

print(len(data.valid_ds))
data.classes
data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet50, metrics=accuracy, model_dir='/tmp/models')
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(9,9), dpi=60)
interp.most_confused(min_val=2)