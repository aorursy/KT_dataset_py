# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('/kaggle/input/faster-submission')

# cascade-submission-result

# faster-submission
cascade = '/kaggle/input/cascade-submission-result/submission.csv'

faster = '/kaggle/input/faster-submission/submission .csv'
def string2bbox(result_string):

    results = result_string.split(' ')

    assert len(results) % 5 == 0

    results = np.array(results, dtype=str)

    results = results.reshape(-1, 5)

    return results
def show_csv_result(csv_file):

    df = pd.read_csv(csv_file)

    results = df['PredictionString']

    for idx in range(len(df)):

        item = df.iloc[idx]

        image_name, bbox = item

        image_path = f'/kaggle/input/global-wheat-detection/test/{image_name}.jpg'

        image = cv.imread(image_path)

        assert image is not None

        bbox = string2bbox(bbox)

        for single_box in bbox:

            score = float(single_box[0])

            _box = list(map(float, single_box[1:]))

            _box = list(map(round, _box))

            x, y, w, h = _box

            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        submission_info = csv_file.split('/')[-2]

        save_path = f'/kaggle/working/vis_results/{submission_info}'

        if not os.path.exists(save_path):

            os.makedirs(save_path)

        cv.imwrite(f'{save_path}/{image_name}.jpg', image)

            
!rm -rf /kaggle/working/vis_results

show_csv_result(cascade)

show_csv_result(faster)
!zip -r vis_results.zip /kaggle/working/vis_results