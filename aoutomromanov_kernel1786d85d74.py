import pandas as pd

data = pd.read_csv("../input/result2/sample_submission.csv")

print (data)

data.to_csv('submission.csv',index=False)





   