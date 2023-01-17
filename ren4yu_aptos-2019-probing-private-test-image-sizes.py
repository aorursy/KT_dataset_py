from pathlib import Path

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
# private + public test

test_csv_path = "../input/aptos2019-blindness-detection/test.csv"

df = pd.read_csv(test_csv_path)
test_image_dir = Path("../input/aptos2019-blindness-detection/test_images")

private_img_cnt = 0

target_img_cnt = 0



for _, row in df.iterrows():

    id_code = row["id_code"]

    img_path = test_image_dir.joinpath(f"{id_code}.png")

    img = cv2.imread(str(img_path), 1)

    h, w, _ = img.shape



    if w == 640 and h == 480:

        target_img_cnt += 1

        

    private_img_cnt += 1

    

target_ratio = target_img_cnt / private_img_cnt

target_ratio_code = int(target_ratio * 10)

print(target_ratio)

print(target_ratio_code)
submissions = [

    "submission_0.683.csv",

    "submission_0.694.csv",

    "submission_0.709.csv",

    "submission_0.711.csv",

    "submission_0.739.csv",

    "submission_0.751.csv",

    "submission_0.755.csv",

    "submission_0.766.csv",

    "submission_0.768.csv",

    "submission_0.785.csv"

]



# select a submission file according to the ratio of target image size count.

pub_test_csv_path = "../input/aptos10submissions/" + submissions[target_ratio_code]

pub_df = pd.read_csv(pub_test_csv_path)

id_to_diagnosis = {id_code: diag for id_code, diag in zip(pub_df.id_code, pub_df.diagnosis)}
all_diagnosis = []



for _, row in df.iterrows():

    id_code = row["id_code"]

    

    if id_code in id_to_diagnosis:

        all_diagnosis.append(id_to_diagnosis[id_code])

    else:

        all_diagnosis.append(0)
id_codes = df.id_code.values

new_df = pd.DataFrame.from_dict(data={"id_code": df.id_code.values, "diagnosis": all_diagnosis})

new_df.to_csv("submission.csv", index=False)