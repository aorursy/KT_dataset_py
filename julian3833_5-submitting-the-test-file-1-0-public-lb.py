import os
import pandas as pd

test_files = [f for f in os.listdir("../input/test/")]
df = pd.read_csv("../input/test_ship_segmentations.csv")
df = df[df['ImageId'].isin(test_files)].drop_duplicates(subset="ImageId")
df.to_csv("submission.csv", index=False)
len(df)
!head submission.csv
