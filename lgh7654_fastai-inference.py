from fastai.gen_doc.nbdoc import *
from fastai.vision import *
URLs.MNIST_TINY
mnist = untar_data(URLs.MNIST_TINY)     # untar_data은 url로 부터 tgz 파일을 다운받는다.

tfms = get_transforms(do_flip=False)    # transform을 위해 설정

                                        # do_flip: if True, a random flip is applied with probability 0.5

                                        # 50% 확률로 이미지 flip을 수행(뒤집기)
mnist
mnist.ls()
(mnist/'train').ls()
from PIL import Image



Image.open('/tmp/.fastai/data/mnist_tiny/train/3/7422.png') # 3 폴더내의 이미지를 살펴보자.
!ls '/tmp/.fastai/data/mnist_tiny/test'
import pandas as pd



df = pd.read_csv(mnist/'labels.csv')



df[df['name'].str.contains("test")]
df[df['name'].str.contains("train")].head()
data = (ImageList.from_folder(mnist)    # ImageList.from_folder를 사용해서 PosixPath('/tmp/.fastai/data/mnist_tiny')접근

        .split_by_folder()              # mnist_tiny 내 폴더명(train, valid)을 기준으로 나눈다. (default로 train, valid 폴더)

        .label_from_folder()            # 폴더명을 label로 사용. (3->3, 7->7)

        .add_test_folder('test')        # *테스트셋은 직접 추가해주어야 한다.*

        .transform(tfms, size=32)       # 트렌스폼 적용 (3, 28, 28) -> (3, 32, 32)

        .databunch()                    # fastai.data_block.LabelLists -> fastai.vision.data.ImageDataBunch 으로 변경

        .normalize(imagenet_stats))     # ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data
data.show_batch(rows=3, figsize=(4,4))
# 위 튜토리얼에서 만든 모델은 다음과 같다.

learn = cnn_learner(data, models.resnet18, metrics=accuracy)

learn.fit_one_cycle(1,1e-2)

learn.save('mini_train')  # 현재 모델 저장
# learn.get_preds(ds_type=DatasetType.Train)
learn.path
(learn.path/'models').ls()
learn = cnn_learner(data, models.resnet18).load('mini_train')  # 저장한 모델 불러오기
learn.export()
learn.path
!ls '/tmp/.fastai/data/mnist_tiny'
# mnist = PosixPath('/tmp/.fastai/data/mnist_tiny')

learn = load_learner(mnist)               # export.pkl 파일이있는 경로만 저징해주면 된다.

learn = load_learner(mnist, 'export.pkl') # 이렇게 특정 pkl 파일을 불러올수도 있음
img = data.train_ds[0][0]

print(type(img))

print(img.shape)

img
learn.predict(img)
ImageList.from_folder(mnist/'test') # ImageList (ItemList를 상속함)
for img in ImageList.from_folder(mnist/'test'):

    img.show()
learn = load_learner(mnist, test=ImageList.from_folder(mnist/'test'))
learn # Train, Valid 는 아무것도 들어있지 않고, Test 셋만 가지고 있다.
learn.data.test_ds
DatasetType.Train, DatasetType.Valid, DatasetType.Test # enum 순서대로 1, 2, 3
preds,y = learn.get_preds(ds_type=DatasetType.Test) # ds_type으로 Train, Valid, Test 중 무엇인지 지정
preds, y # 예측값(proba)와 label을 출력한다. 

         # test 셋의 label이 제공되지않아 모두 0으로 표시되었다.
preds.argmax(1) # 위의 test set의 이미지와 비교해보면 모두 정답으로 예측하였다.
planet = untar_data(URLs.PLANET_TINY)

planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

planet.ls()
(planet/'train').ls()
pd.read_csv(planet/'labels.csv').head()
data = (ImageList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')

        .split_by_rand_pct()

        .label_from_df(label_delim=' ')   # labels.csv 파일을 label로 적용

        .transform(planet_tfms, size=128) 

        .databunch()

        .normalize(imagenet_stats))
data
learn = cnn_learner(data, models.resnet18)

learn.fit_one_cycle(5,1e-2)

learn.save('mini_train') # 모델 저장
learn = cnn_learner(data, models.resnet18).load('mini_train') # 모델 불러오기
(planet/'models').ls()
learn.export() # learner 저장하기
planet.ls() # export.pkl
learn = load_learner(planet) # learner를 불러오자 (export.pkl)
img, mlabel = data.train_ds[0] # 이미지와 레이블을 저장
img
mlabel
learn.predict(img)
learn.predict(img, thresh=0.8) # threshold를 임의로 지정할 수 있다. (기본 0.5)