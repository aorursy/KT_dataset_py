%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
import requests, zipfile, io

zip_file_url = "https://download.pytorch.org/tutorial/faces.zip"

r = requests.get(zip_file_url)

z = zipfile.ZipFile(io.BytesIO(r.content))

z.extractall()
import os

os.listdir()
path = "faces"
images_and_coordinates = pd.read_csv("faces/face_landmarks.csv")
images_and_coordinates.head()
images_and_coordinates.describe()
def getPointCoordinates(filename):

    singleDFEntry = images_and_coordinates[images_and_coordinates.image_name == filename[len(path +'/'):]]

    onlyDFCoordinates = singleDFEntry.drop(columns='image_name')

    numpyMatrix = onlyDFCoordinates.values.reshape(-1,2)

    numpyMatrix[:, 0], numpyMatrix[:, 1] = numpyMatrix[:, 1], numpyMatrix[:, 0].copy()

    ##print(numpyMatrix.shape)

    return torch.from_numpy(numpyMatrix).float()
fileName = path + '/' + images_and_coordinates.image_name[65]

samplePoints = getPointCoordinates(fileName)

img = open_image(fileName)

samplePoints = ImagePoints(FlowField(img.size, samplePoints), scale=True)

img.show(y=samplePoints)
src = (PointsItemList.from_csv(path, 'face_landmarks.csv')

        .split_by_rand_pct(0.2)

        .label_from_func(getPointCoordinates))
src
data = (src.transform(tfm_y=True, size=(160,160))

        .databunch(bs=16).normalize(imagenet_stats))   
data.show_batch(3, figsize=(9,6))
learn = cnn_learner(data, models.resnet34)
learn.lr_find()

learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(5, slice(lr))
learn.show_results()