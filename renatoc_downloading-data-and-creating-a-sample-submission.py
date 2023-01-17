import os
import sys
import getpass
# Now you have to go to https://www.kaggle.com/<username>/account
# and download an API token. The API token is a JSON file and
# you can paste the file's contents in this cell

path = os.path.join(os.path.expanduser('~'), '.kaggle')
try:
    os.makedirs(path)
except FileExistsError:
    pass
with open(os.path.join(path, 'kaggle.json'), 'w') as fp:
    fp.write(getpass.getpass())
    os.chmod(fp.name, 0o600)
from kaggle import cli
sys.argv = 'kaggle competitions download -c kddbr-2018'.split()
cli.main()
# Alright! We should have some data! Let's delete our kaggle token just in case
os.unlink(os.path.join(path, 'kaggle.json'))
# Now, let's change to the data dir
os.chdir('/tmp/.kaggle/competitions/kddbr-2018')
# Let's create a zero submission file now...
import pandas as pd
import numpy as np

testdf = pd.read_csv('test.csv')

# We will create an array of zeros, then we will use the ID's in the
# test set as our index.

submissiondf = pd.DataFrame(
    columns=['production'],
    data=np.zeros(len(testdf)),
    index=testdf['Id']
)

submissiondf
# Cool! To save our file:
submissiondf.to_csv('zeros.csv', index='Id')
!cat zeros.csv