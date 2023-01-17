# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import time

# Read the test.csv and get the unique StudyInstanceUID

# root = '/kaggle/input/rsna-str-pulmonary-embolism-detection'

start_time = time.time()

def elapsed_time():
    t = str(round((time.time() - start_time) / 60, 2))
    print(f'{t} minutes')




test_csv_path = '/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv'

test = pd.read_csv(test_csv_path)

print(test.shape)
test.head()



suffix_list = ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe',
                     'chronic_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']


study_exam_id_list = []
for v in test.StudyInstanceUID:
    if v not in study_exam_id_list:
        study_exam_id_list.append(v)

count = 1

image_id_list = []
for v in test.SOPInstanceUID:
    if v not in image_id_list:
        elapsed_time()
        print(f'{count} {v}')
        count = count + 1
        image_id_list.append(v)

print(f'len(image_id_list) = {len(image_id_list)}')


str_temp =''

rows_list = []


for suffix in suffix_list:
    for study_exam_id in study_exam_id_list:
        str_temp = study_exam_id +'_'+ suffix
        rows_list.append(str_temp)
        elapsed_time()


for i in image_id_list:
    rows_list.append(i)


score = []
for i in range(len(rows_list)):
    score.append(0.095) # Hard coded value 


print(len(score))
elapsed_time()

sub_temp = pd.DataFrame({'id':rows_list, 'label':score})

sub_temp.to_csv('submission.csv', index=False)



# SOPInstanceUID

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
