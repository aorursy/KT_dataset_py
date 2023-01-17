import fastai.vision as fv
mnist = fv.untar_data(fv.URLs.MNIST_TINY); mnist
mnist.ls()
tfms = fv.get_transforms(do_flip=False); tfms
data = (fv.ImageItemList.from_folder(mnist)

        .split_by_folder()          

        .label_from_folder()

        .add_test_folder('test')

        .transform(tfms, size=32)

        .databunch()

        .normalize(fv.imagenet_stats)) 
learn = fv.create_cnn(data, fv.models.resnet18, metrics=fv.accuracy)

learn.fit_one_cycle(1,1e-2)
learn.save('mini_train')
learn = fv.create_cnn(data, fv.models.resnet18).load('mini_train')
learn.export()
learn = fv.load_learner(mnist)
img = data.train_ds[0][0]

learn.predict(img)
learn = fv.load_learner(mnist, test=fv.ImageItemList.from_folder(mnist/'test'))
preds,y = learn.get_preds(ds_type=fv.DatasetType.Test)

preds[:5]