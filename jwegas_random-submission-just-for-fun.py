import numpy as np

import pandas as pd

import zipfile
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
def getShuffledList(ad_id_string):

    tmp = np.array(ad_id_string.split())

    np.random.shuffle(tmp)

    return ' '.join(tmp)
submission['ad_id'] = submission['ad_id'].apply(lambda x: getShuffledList(x))
submission.head()
submission.to_csv('submission_shuffled.csv', index=False)
with zipfile.ZipFile('submission_shuffled.csv.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:

    myzip.write('submission_shuffled.csv')