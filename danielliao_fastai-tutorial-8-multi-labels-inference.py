import fastai.vision as fv
fv.__version__
planet = fv.untar_data(fv.URLs.PLANET_TINY); planet
planet.ls()
(planet/"train").ls()[:5]
fnames = fv.get_image_files(planet/"train"); fnames[:2]

fnames.__len__()
planet_tfms = fv.get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

planet_tfms
# fv.ImageList is only available in 1.0.46, we are at 1.0.45 so have to use fv.ImageItemList

data = (fv.ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')

        .random_split_by_pct()

        .label_from_df(label_delim=' ')

        .transform(planet_tfms, size=128)

        .databunch()

        .normalize(fv.imagenet_stats))
data.show_batch(rows=2, figsize=(9,7))
learn = fv.create_cnn(data, fv.models.resnet18)

learn.fit_one_cycle(5,1e-2)

learn.save('mini_train')
learn.show_results(rows=3, figsize=(12,15))
learn.export()
learn = fv.load_learner(planet)
img = data.train_ds[0][0]

learn.predict(img)
learn.predict(img, thresh=0.3)