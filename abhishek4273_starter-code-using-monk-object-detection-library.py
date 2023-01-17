import os

import sys
os.listdir("/kaggle/input/kaggle-monk-pytorch-efficientdet/kaggle_monk_pytorch_efficientdet")
os.listdir("/kaggle/input/global-wheat-detection")
! cp -r /kaggle/input/kaggle-monk-pytorch-efficientdet/kaggle_monk_pytorch_efficientdet .
! cd kaggle_monk_pytorch_efficientdet/cocoapi/PythonAPI/ && pip install -e .
sys.path.append("kaggle_monk_pytorch_efficientdet/cocoapi/PythonAPI/")
! pip install kaggle_monk_pytorch_efficientdet/webcolors-1.11.1-py3-none-any.whl
## Import



import os

import sys

sys.path.append("kaggle_monk_pytorch_efficientdet/Monk_Object_Detection/10_pytorch_efficientdet/lib/");



from infer_detector import Infer



gtf = Infer();
## Load model



model_path = "kaggle_monk_pytorch_efficientdet/trained_weights/custom/efficientdet-d4_trained.pth";

classes_list = ["wheat"];

gtf.load_model(model_path, classes_list, use_gpu=True);
img_list = os.listdir("/kaggle/input/global-wheat-detection/test/")
img_path = "/kaggle/input/global-wheat-detection/test/" + img_list[0];

scores, labels, boxes = gtf.predict(img_path, threshold=0.3);

from IPython.display import Image

Image(filename='output.jpg') 
# Run in loop for for submission



from tqdm import tqdm

combined = [];

for i in tqdm(range(len(img_list))):

    img_id = img_list[i].split(".")[0];

    img_path = "/kaggle/input/global-wheat-detection/test/" + img_list[i];

    scores, labels, boxes = gtf.predict(img_path, threshold=0.3);

    

    wr = "";

    

    for j in range(len(scores)):

        score = float(scores[j])

        x1 = int(boxes[j][0]);

        y1 = int(boxes[j][1]);

        x2 = int(boxes[j][2]);

        y2 = int(boxes[j][3]);

        w = x2-x1;

        h = y2-y1;

        #print(type(score), type(x1), type(y1), type(w), type(h))

        wr += str(score) + " " + str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + " ";

    wr = wr[:len(wr)-1];

    combined.append([img_id, wr]);
import pandas as pd



df = pd.DataFrame(combined, columns = ['image_id', 'PredictionString']);

df.to_csv("submission.csv", index=False)
! rm -r kaggle_monk_pytorch_efficientdet
df.iloc[0]
os.listdir("./")