%reset -sf

!ls ../input/shopee-product-detection-student/
import pandas as pd

import numpy as np

submission_df = pd.read_csv("../input/shopee-product-detection-student/test.csv")
submission_df["category"] = submission_df["category"].apply(lambda x: np.random.randint(10))  # 00 to 41 inclusive

submission_df["category"] = submission_df["category"].apply(lambda x: "{:02}".format(x))  # pad zeroes

submission_df.to_csv("submission.csv", index=False)
!head -10 submission.csv
# count number of files in directory

!find "../input/shopee-product-detection-student/test/test/test/" -type f | wc -l
# count the number of lines in a submission

!wc -l submission.csv
import glob

dir_files = [file.split("/")[-1][:-4] for file in glob.glob("../input/shopee-product-detection-student/test/test/test/*.jpg")]

test_files = [file[:-4] for file in submission_df["filename"]]
set(dir_files) - set(test_files)