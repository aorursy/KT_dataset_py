import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.csv'):

            print(os.path.join(dirname, filename))
# Path to uploaded submission file

submission = pd.read_csv("/kaggle/input/first-submissioncsv/submission.csv")



# Save submission.csv for grading

submission.sort_values('filename').to_csv('submission.csv', index=False)



# Print submission head

submission.head()
