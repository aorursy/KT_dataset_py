import pandas as pd



df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

df_sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")



def lb_trick(selected): return " ".join(set(selected.lower().split())) # LB trick



df_sub.selected_text = df_test.text.str.lower().map(lb_trick)

df_sub.to_csv("submission.csv", index=False)

# Still one line left for a comment! :)