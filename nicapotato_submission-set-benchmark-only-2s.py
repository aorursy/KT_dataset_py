import pandas as pd
sub = pd.read_csv("../input/sampleSubmission.csv")
print(sub.Sentiment.value_counts())
sub.to_csv("sampleSubmission.csv",index=False)
sub.head()