%reload_ext autoreload

%autoreload 2

%matplotlib inline
#!pip install git+https://github.com/fastai/fastai.git

!pip install fastai==1.0.55
from fastai.vision import *
import requests, zipfile, io

zip_file_url = "https://download.pytorch.org/tutorial/faces.zip"

r = requests.get(zip_file_url)

z = zipfile.ZipFile(io.BytesIO(r.content))

z.extractall()
import os

os.listdir()

# os.listdir('faces')
path = "faces"
images_and_coordinates = pd.read_csv("faces/face_landmarks.csv")

images_and_coordinates
images_and_coordinates.head()
images_and_coordinates.describe()
def getPointCoordinates(filename):

    singleDFEntry = images_and_coordinates[images_and_coordinates.image_name == filename[len(path +'/'):]]

    onlyDFCoordinates = singleDFEntry.drop(columns='image_name')

#     print(onlyDFCoordinates)

    numpyMatrix = onlyDFCoordinates.values.reshape(-1,2)

#     print(numpyMatrix)

    numpyMatrix[:, 0], numpyMatrix[:, 1] = numpyMatrix[:, 1], numpyMatrix[:, 0].copy()

#     print(numpyMatrix)

#     print(numpyMatrix.shape)

    return torch.from_numpy(numpyMatrix).float()
fileName = path + '/' + images_and_coordinates.image_name[65]

samplePoints = getPointCoordinates(fileName)

img = open_image(fileName)

samplePoints = ImagePoints(FlowField(img.size, samplePoints), scale=True)

img.show(y=samplePoints)
# src = (PointsItemList.from_df(path, 'face_landmarks.csv')

#                .filter_by_rand(0.5)

#         .split_by_rand_pct(0.2)

#         .label_from_func(getPointCoordinates)

# )
import cv2

import PIL



def after_open(img):

    print(type(img))

    x = np.asarray(img)

#     print(x)

#     npim = image2np(img)*255 # convert to numpy array in range 0-255

#     npim = npim.astype(np.uint8) # convert to int

    

    # If your image is a regular colour image, remember it will be in RGB so convert to BGR if needed

    img_bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    

    print(img_bgr.shape)

    x, y, c = img_bgr.shape

    

#     center = (y/2, x/2)

    center = (100,100)

    scale = 0.5

    

    xmin = int(center[1] - (scale * center[1]))

    xmax = int(center[1] + (scale * center[1]))

    ymin = int(center[0] - (scale * center[0]))

    ymax = int(center[0] + (scale * center[0]))

    

    img_bgr = img_bgr[xmin:xmax, ymin:ymax, :]



    

    

    

# #     img_bgr = cv2.resize(img_bgr, (100,100))

#     img_bgr = img_bgr[50:,50:,:]





    

    # transform using opencv

    # ...

    # don't forget to convert back to RGB if needed

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    

    return PIL.Image.fromarray(img_rgb)



    # assuming result image is stored in result_img

#     return pil2tensor(img_rgb, dtype=np.float32)/255 # convert back to tensor and return

#     return img

#     print(img_rgb.shape)

#     return Image(torch.as_tensor(img_rgb))





def getPointCoordinates_test(filename):

    singleDFEntry = images_and_coordinates[images_and_coordinates.image_name == filename[len(path +'/'):]]

    onlyDFCoordinates = singleDFEntry.drop(columns='image_name')

#     print(onlyDFCoordinates)

    numpyMatrix = onlyDFCoordinates.values.reshape(-1,2)

#     print(numpyMatrix)

    numpyMatrix[:, 0], numpyMatrix[:, 1] = numpyMatrix[:, 1], numpyMatrix[:, 0].copy()

    

    # Translation

#     x = 100

#     y = 100

#     print(numpyMatrix.shape)

#     h,w = numpyMatrix.shape









    center = (100,100)

    scale = 0.5

    

    xmin = int(center[1] - (scale * center[1]))

    xmax = int(center[1] + (scale * center[1]))

    ymin = int(center[0] - (scale * center[0]))

    ymax = int(center[0] + (scale * center[0]))

    

#     img_bgr = img_bgr[xmin:xmax, ymin:ymax, :]



    numpyMatrix[:, 0] = numpyMatrix[:, 0] - (xmin)

    numpyMatrix[:, 1] = numpyMatrix[:, 1] - (ymin)

    

#     w, h = numpyMatrix.size()

    

    

#     print(numpyMatrix)

#     print(numpyMatrix.shape)

    return torch.from_numpy(numpyMatrix).float()



src = (PointsItemList.from_df(images_and_coordinates, path, 

                              after_open=after_open

                             )

               .filter_by_rand(0.5)

        .split_by_rand_pct(0.2)

        .label_from_func(getPointCoordinates_test)

)
src
import cv2



def custom_transform(x): # you get a PyTorch tensor as input, not a fastai Image object

#     npim = image2np(x)*255 # convert to numpy array in range 0-255

#     npim = npim.astype(np.uint8) # convert to int

    

    # If your image is a regular colour image, remember it will be in RGB so convert to BGR if needed

#     img_bgr = cv2.cvtColor(npim, cv2.COLOR_RGB2BGR)

    

    # transform using opencv

    # ...

    # don't forget to convert back to RGB if needed



    # assuming result image is stored in result_img

#     return pil2tensor(img_bgr, dtype=np.float32)/255 # convert back to tensor and return

    return x



# custom_trans = TfmPixel(custom_transform) # wrap in TfmPixel to create a valid transform

# tfms = [[custom_trans()],[custom_trans()]] # set your custom transform for training dataset only

# final_tfms = get_transforms(tfms, size=(160,160))





# data = (src.transform(tfms,

#                       tfm_y=True, 

#                       size=(160,160)

#                      )

#         .databunch(bs=16).normalize(imagenet_stats))



data = (src.transform(

    tfm_y=True, 

    size=(160,160))

        .databunch(bs=16).normalize(imagenet_stats))   
data.show_batch(3, figsize=(9,6))
data = (src.transform(tfm_y=True, size=(160,160))

        .databunch(bs=16).normalize(imagenet_stats))   
data.show_batch(3, figsize=(9,6))
learn = cnn_learner(data, models.resnet34)
learn.lr_find()

learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(5, slice(lr))
learn.show_results()