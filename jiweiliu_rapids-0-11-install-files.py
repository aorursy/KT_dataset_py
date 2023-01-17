import time
%%time

# INSTALL RAPIDS WITH CONDA. TAKES 6 MINUTES :-(

import sys

!conda create -n rapids -c rapidsai/label/xgboost -c rapidsai -c nvidia -c conda-forge rapids=0.11 python=3.6 cudatoolkit=10.1 --yes

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf, cuml

print('cuML version',cuml.__version__)

print('cuDF version',cudf.__version__)



df = cudf.DataFrame({'a':[1,2,3]})

df
%%time

!7z a rapids.zip /opt/conda/envs/rapids/lib/*
%%time

!ls ./