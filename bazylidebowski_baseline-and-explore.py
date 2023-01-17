import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from kaggle.competitions import twosigmanews

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
env = twosigmanews.make_env()
# We will use these to show the format
day1_captured = False
market_day1 = None
news_day1 = None
pred_template_day1 = None

days = env.get_prediction_days()
for (market_obs_df, news_obs_df, predictions_template_df) in days:

    # Store only the day 1 information
    if not day1_captured:
        market_day1 = market_obs_df.copy(True)
        news_day1 = news_obs_df.copy(True)
        pred_template_day1 = predictions_template_df.copy(True)
        day1_captured = True

    # Set all predictions for this day to 1.0
    predictions_template_df['confidenceValue'] = 1.0
    env.predict(predictions_template_df)

# Create submission
env.write_submission_file()
market_day1.head(10)
news_day1.head(10)
pred_template_day1.head(10)
