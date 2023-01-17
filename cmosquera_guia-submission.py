#Reemplazar 'submission1' por el nombre de tu versión, y 'sample_submission' por el nombre de tu CSV 

import pandas as pd

mysubmission = pd.read_csv("../input/submission1/sample_submission.csv")

mysubmission.to_csv("submission.csv")