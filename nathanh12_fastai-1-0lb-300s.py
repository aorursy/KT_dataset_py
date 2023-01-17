import time

start = time.time()



from pathlib import Path

from fastai import *

from fastai.vision import *
data_folder = Path("../input")
train_df = pd.read_csv(data_folder/'train.csv')

test_df = pd.read_csv(data_folder/'sample_submission.csv')
test_data = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
train_imgs = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')

                 .split_by_rand_pct(0.1)  

                 .label_from_df()

                 .add_test(test_data)

                 .transform(get_transforms(flip_vert=True), size=128)

                 .databunch(path='.', bs=96)

                 .normalize(imagenet_stats)

       )

                    
learner = cnn_learner(train_imgs, models.densenet161)
lr = 3e-2
learner.fit_one_cycle(6, max_lr = slice(lr))
preds,_ = learner.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)
end = time.time()

print(end - start)