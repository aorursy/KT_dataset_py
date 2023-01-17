#import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



%matplotlib inline

import matplotlib.pyplot as plt



from tqdm import tqdm



from PIL import Image, ImageDraw
!pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.3/imageai-2.0.3-py3-none-any.whl
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
!wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
s_sub = pd.read_csv('../input/sample_submission.csv')

s_sub.head()
len(s_sub), s_sub.iloc[-1]
s_sub['PredictionString'][0]
test_filename = os.listdir('../input/test')

test_filename[:5]
labelMap = pd.read_csv('class-descriptions-boxable.csv', header=None, names=['labelName', 'Label'])

labelMap.head()
labelMap.loc[labelMap['labelName'].isin(['/m/05s2s','/m/0c9ph5'])]
# Show one image

def show_image_by_index(i):

    sample_image = plt.imread(f'../input/test/{test_filename[i]}')

    plt.imshow(sample_image)



def show_image_by_filename(filename):

    sample_image = plt.imread(filename)

    plt.imshow(sample_image)
show_image_by_index(2)
show_image_by_filename('../input/test/209b3b3b4102fba5.jpg')
from imageai.Detection import ObjectDetection
#Get the path to the working directory

execution_path = os.getcwd()
# load model

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3() #Retina도 있고 tinyYOLO도 있음

detector.setModelPath(os.path.join(execution_path, "yolo.h5"))

detector.loadModel()
# test detection on one image 

detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test', '209b3b3b4102fba5.jpg'),

                                             output_image_path=os.path.join(execution_path , "result.jpg"),

#                                            output_type = 'array',

                                             extract_detected_objects = False)

for eachObject in detections:

    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )



# show the result

show_image_by_filename('./result.jpg') #( x_min,y_min, x_max, y_max)으로 그려지네
def format_prediction_string(image_id, result, labelMap, xSize, ySize):

    prediction_strings = []

    #print(xSize, ySize)

    for i in range(len(result)):

        class_name = result[i]['name'].capitalize()

        class_name = pd.DataFrame(labelMap.loc[labelMap['Label'].isin([class_name])]['labelName'])

        #print(result[i]['box_points'])

        xMin = result[i]['box_points'][0] / xSize

        xMax = result[i]['box_points'][2] / xSize

        yMin = result[i]['box_points'][1] / ySize

        yMax = result[i]['box_points'][3] / ySize

        

        if len(class_name) > 0:

            class_name = class_name.iloc[0]['labelName']

            boxes = [xMin, yMin, xMax, yMax]#result[i]['box_points']

            score = result[i]['percentage_probability']



            prediction_strings.append(

                f"{class_name} {score} " + " ".join(map(str, boxes))

            )

        

    prediction_string = " ".join(prediction_strings)



    return {

            "ImageID": image_id,

            "PredictionString": prediction_string

            }
# Test prediction on input images

res = []

for i in tqdm(os.listdir('../input/test')[0:3]):

    detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test', i),

                                                 output_image_path=os.path.join(execution_path , "result.jpg"),

                                                 #output_type = 'array',

                                                 extract_detected_objects = False)

    currentImg = Image.open(os.path.join('../input/test', i))

    print(currentImg.size) #사이즈가 다 다르구나

    xSize = currentImg.size[0]

    ySize = currentImg.size[1]

    print(detections)

    p = format_prediction_string(i, detections, labelMap, xSize, ySize)

    res.append(p)
res
detections[0]['name'].capitalize()

class_name=detections[0]['name'].capitalize()
class_name=pd.DataFrame(labelMap.loc[labelMap['Label'].isin([class_name])]['labelName'])

class_name
# Convert res variable to DataFrame

pred_df = pd.DataFrame(res)

pred_df.head()
pd.DataFrame(res)
# Get the file name without extension

pred_df['ImageID'] = pred_df['ImageID'].map(lambda x: x.split(".")[0])
pred_df.head()
# Run detection on test images

# 여기가 오래 걸리네

sample_submission_df = pd.read_csv('../input/sample_submission.csv')

image_ids = sample_submission_df['ImageId']

predictions = []

res = []

for image_id in tqdm(image_ids):

    detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test', image_id + '.jpg'),

                                                 output_image_path=os.path.join(execution_path , "result.jpg"),

                                                 #output_type = 'array',

                                                 extract_detected_objects = False)

    currentImg = Image.open(os.path.join('../input/test', image_id + '.jpg'))

    xSize = currentImg.size[0]

    ySize = currentImg.size[1]

    p = format_prediction_string(image_id, detections, labelMap, xSize, ySize)

    res.append(p)
# Save submission file

pred_df = pd.DataFrame(res)

pred_df['ImageID'] = pred_df['ImageID'].map(lambda x: x.split(".")[0])

pred_df.to_csv('result.csv', index=False)