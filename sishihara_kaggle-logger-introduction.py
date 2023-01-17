import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG





def create_logger(exp_version):

    log_file = ("{}.log".format(exp_version))



    # logger

    logger_ = getLogger(exp_version)

    logger_.setLevel(DEBUG)



    # formatter

    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")



    # file handler

    fh = FileHandler(log_file)

    fh.setLevel(DEBUG)

    fh.setFormatter(fmr)



    # stream handler

    ch = StreamHandler()

    ch.setLevel(INFO)

    ch.setFormatter(fmr)



    logger_.addHandler(fh)

    logger_.addHandler(ch)





def get_logger(exp_version):

    return getLogger(exp_version)
VERSION = "001" # 実験番号

create_logger(VERSION)

get_logger(VERSION).info("This is a message")
df = pd.read_csv('../input/train.csv')
df.head()
df.columns
df.shape
get_logger(VERSION).info(df.columns)

get_logger(VERSION).info(df.shape)