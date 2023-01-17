import pandas as pd
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sub_df['selected_text'] = test_df['text'].values
sub_df
sub_df.to_csv("/kaggle/working/submission.csv", index=False)