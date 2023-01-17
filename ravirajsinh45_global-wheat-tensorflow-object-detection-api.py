import numpy as np

import tensorflow as tf

import cv2 

import matplotlib.pyplot as plt

import os

import pandas as pd

from tqdm import tqdm
model_path = '../input/global-wheat-tf-object-detection-api-inception-v2/inceptinV2_frozen_inference_graph.pb'

test_images_path = '../input/global-wheat-detection/test/'

sample_csv = '../input/global-wheat-detection/sample_submission.csv'
sub = pd.read_csv(sample_csv)

sub.head()
with tf.compat.v1.gfile.FastGFile(model_path, 'rb') as f:

    graph_def = tf.compat.v1.GraphDef()

    graph_def.ParseFromString(f.read())
submission = pd.DataFrame(columns=list(sub.columns))

plt.figure(figsize=(20,10))



with tf.compat.v1.Session() as sess:

        # Restore session

        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')



        for ind,image_name in tqdm(enumerate(os.listdir(test_images_path))) :



            img_name = image_name



    





            # Read and preprocess an image.

            img = cv2.imread(test_images_path + img_name,1)

            rows = img.shape[0]

            cols = img.shape[1]

            inp = cv2.resize(img, (1024, 1024))

            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB



            # Run the model

            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),

                            sess.graph.get_tensor_by_name('detection_scores:0'),

                            sess.graph.get_tensor_by_name('detection_boxes:0'),

                            sess.graph.get_tensor_by_name('detection_classes:0')],

                            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})



            # Saving detected bounding boxes.

            num_detections = int(out[0][0])

            pred_str = ''

            for i in range(num_detections):

                classId = int(out[3][0][i])

                score = float(out[1][0][i])

                bbox = [float(v) for v in out[2][0][i]]

                if score > 0.3:

                    x = int(bbox[1] * cols)

                    y = int(bbox[0] * rows)

                    xmax = int(bbox[3] * cols)

                    ymax = int(bbox[2] * rows)



                    cv2.rectangle(img, (x, y), (xmax, ymax), (0,0,255), 2)



                    pred = '{} {} {} {} {} '.format(np.round(score,2),x,y,xmax-x,ymax-y)

                    pred_str = pred_str + pred

            if ind<10:

                plt.subplot(2,5,ind+1)

                plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

                plt.title(image_name[:-4])

                plt.axis('off')



            submission.loc[ind,'image_id'] = image_name[:-4]

            submission.loc[ind,'PredictionString'] = pred_str



plt.show()
submission
submission.to_csv('submission.csv',index=False)