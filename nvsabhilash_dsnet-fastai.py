from fastai.vision import *

from fastai.metrics import accuracy, fbeta
path = Path("../input")
src = (ImageList.from_folder(path/'train')

                .split_by_rand_pct()

                .label_from_folder()

                .add_test(Path('../input/test/test').ls()))
data = (src.transform(get_transforms(), size=128)

           .databunch(path=Path("../"))

           .normalize(imagenet_stats))
data.show_batch(3)
data.batch_stats()
arch = models.resnet34
learn = cnn_learner(data, arch, metrics=accuracy).to_fp16()
learn.lr_find()

learn.recorder.plot()
lr = 5e-2
learn.fit_one_cycle(6, max_lr=slice(lr))
learn.save('stage-1')
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('submission-1.csv', index=False)
learn.load('stage-1');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(5e-6, lr/10))
learn.save('stage-2')
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('submission-2.csv', index=False)
learn.load('stage-2');
data = (src.transform(get_transforms(), size=256)

           .databunch(path=Path("../"))

           .normalize(imagenet_stats))
learn.data = data

learn = learn.to_fp16()
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr = 5e-3
learn.fit_one_cycle(5, max_lr=slice(lr))
learn.save('stage-256-1')
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('submission-3.csv', index=False)
learn.load('stage-256-1');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-4))
learn.save('stage-256-2')
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('submission-4.csv', index=False)
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn.to_fp32(), preds, y, losses)
interp.plot_top_losses(6, figsize=(15, 15))
interp.most_confused(min_val=2)
test_probs, _ = learn.TTA(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('submission-5.csv', index=False)
interp.plot_confusion_matrix(figsize=(20,20), normalize=True, )