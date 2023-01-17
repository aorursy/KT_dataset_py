# Current CWD
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Installing and Updating Dependencies
!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!pip install mindsdb --no-cache-dir --force-reinstall
# Importing Libraries
import pandas as pd
from mindsdb import *

# Downloading Files from Kernel
from IPython.display import FileLink
# Initiating MindsDB Object
mdb = MindsDB(name = "MindsDB Modelling")
# Instructing what to Learn
mdb.learn(
    from_data = "/kaggle/input/bad-customer-or-good-customer/x_train.csv", 
    to_predict = 'target'
)
# Importing Test to Generate Results
test = pd.read_csv('/kaggle/input/bad-customer-or-good-customer/x_test.csv')
test.head(5)
# Taking the Result in a List
output = []
for i in range(0, 223):
  output.append(int(result[i]['target']))
# Defining Submission DataFrame
submission = pd.DataFrame()
submission['ID'] = test.ID
submission['Expected'] = output

submission.head(5)
# Converting Dataframe into CSV
submission.to_csv('MindsDB Output.csv', index = False)
# Download CSV
FileLink(r'MindsDB Output.csv')