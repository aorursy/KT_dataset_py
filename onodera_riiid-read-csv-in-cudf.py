import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
from time import time

from contextlib import contextmanager



import cudf
@contextmanager

def timer(name):

    t0 = time()

    yield

    print(f'[{name}] done in {time() - t0:.2f} s')
with timer('cuDF'):

    df = cudf.read_csv('../input/riiid-test-answer-prediction/train.csv')
df