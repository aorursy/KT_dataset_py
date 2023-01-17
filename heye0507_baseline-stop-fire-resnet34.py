!ls ../input/fire-test/challenge1/
from fastai.vision import *

from fastai import *
path = Path('../input/fire-test/challenge1')
smoke_path = path/'smoke'

no_smoke_path = path/'no_smoke'
smoke_filenames = get_files(smoke_path)

no_smoke_filenames = get_files(no_smoke_path)

len(smoke_filenames),len(no_smoke_filenames)
smoke_img = open_image(smoke_filenames[0])

smoke_img.size
smoke_img
no_smoke_img = open_image(no_smoke_filenames[0])

no_smoke_img.size
no_smoke_img
tfms = get_transforms()
data = (ImageList

        .from_folder(path,include=['smoke','no_smoke'])

        .split_by_rand_pct()

        .label_from_folder()

        .transform(tfms,size=(128,128))

        .databunch(bs=64)

        .normalize(imagenet_stats)

)
data.show_batch(rows=3,figsize=(12,10))
# loading imagenet pre-trained model, also using mix precision for training 

learn = cnn_learner(data,models.resnet34,metrics=[accuracy],model_dir='/kaggle/working').to_fp16() 
learn.model[1] # customized Head
# Finding best learning rate

learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = 3e-3

learn.fit_one_cycle(3,slice(lr))
learn.unfreeze() # Unfreeze the body, fine tune the whole model
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.save('resnet34-stage-1-128')
learn.fit_one_cycle(5,slice(1e-5,lr/5))
data_256 = (ImageList

        .from_folder(path,include=['smoke','no_smoke'])

        .split_by_rand_pct()

        .label_from_folder()

        .transform(tfms,size=(256,256))

        .databunch(bs=64)

        .normalize(imagenet_stats)

)
learn.save('res34-stage-2-128')
learn.data = data_256

learn.freeze_to(-1)
learn.to_fp16(); # using mix precision 
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-3

learn.fit_one_cycle(3,slice(lr))
learn.unfreeze()
learn.save('res34-stage-1-256')
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,slice(1e-5,lr/5))
learn.save('res34-stage-2-256')
y_p, y, loss = learn.get_preds(with_loss=True)
y_p.shape,y.shape
pred = y_p.argmax(dim=-1).float()
pred.shape
from sklearn.metrics import f1_score
f1_score(y,pred)
learn.show_results()
interp = ClassificationInterpretation(learn, y_p, y, loss)
interp.plot_confusion_matrix()
interp.plot_top_losses(9,figsize=(12,12))