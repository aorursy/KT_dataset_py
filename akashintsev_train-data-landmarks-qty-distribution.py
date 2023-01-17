import pandas as pd

import numpy as np
train_data = pd.read_csv("https://s3.amazonaws.com/google-landmark/metadata/train.csv")
landmarks = train_data['landmark_id'].value_counts().sort_values(ascending=False).to_frame(name="Landmark's samples qty")
percentiles = list(range(10, 110, 10))
hist = landmarks.hist()
hist = landmarks[landmarks<100].hist()
hist = landmarks[landmarks<50].hist()
pd.DataFrame(np.percentile(landmarks, percentiles), index=percentiles, columns = ["Landmark's samples qty"])
landmarks_qty = train_data['landmark_id'].map(landmarks["Landmark's samples qty"]).to_frame(name="Landmark's samples qty")
hist = landmarks_qty.hist()
hist = landmarks_qty[landmarks_qty<100].hist()
hist = landmarks_qty[landmarks_qty<50].hist()
pd.DataFrame(np.percentile(landmarks_qty, percentiles), index=percentiles, columns = ['Landmark images qty'])