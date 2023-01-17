import fastai.vision as fv
camvid = fv.untar_data(fv.URLs.CAMVID_TINY)

path_lbl = camvid/'labels'

path_img = camvid/'images'
codes = fv.np.loadtxt(camvid/'codes.txt', dtype=str)

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
data = (fv.SegmentationItemList.from_folder(path_img)

        .random_split_by_pct()

        .label_from_func(get_y_fn, classes=codes)

        .transform(fv.get_transforms(), tfm_y=True, size=128)

        .databunch(bs=16, path=camvid)

        .normalize(fv.imagenet_stats))
data.show_batch(rows=2, figsize=(7,5))
learn = fv.unet_learner(data, fv.models.resnet18)

learn.fit_one_cycle(3,1e-2)

learn.save('mini_train')
learn.show_results()