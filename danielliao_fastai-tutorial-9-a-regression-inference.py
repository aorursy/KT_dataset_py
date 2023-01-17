import fastai.vision as fv
biwi = fv.untar_data(fv.URLs.BIWI_SAMPLE)
fn2ctr = fv.pickle.load(open(biwi/'centers.pkl', 'rb'))
data = (fv.PointsItemList.from_folder(biwi)

        .random_split_by_pct(seed=42)

        .label_from_func(lambda o:fn2ctr[o.name])

        .transform(fv.get_transforms(), tfm_y=True, size=(120,160))

        .databunch()

        .normalize(fv.imagenet_stats))
data.show_batch(rows=3, figsize=(9,6))
learn = fv.create_cnn(data, fv.models.resnet18, lin_ftrs=[100], ps=0.05)

learn.fit_one_cycle(5, 5e-2)

learn.save('mini_train')
learn.show_results(rows=3)
learn.export()
learn = fv.load_learner(biwi)
img = data.valid_ds[0][0]

learn.predict(img)
img.show(y=learn.predict(img)[0])