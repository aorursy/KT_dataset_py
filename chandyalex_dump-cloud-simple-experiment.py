import pandas as pd



submissionCSV = pd.read_csv(

    '../input/understanding_cloud_organization/sample_submission.csv',

    converters={'EncodedPixels': lambda e: ''})

print(submissionCSV.head())

submissionCSV.to_csv('submission.csv', index=False)