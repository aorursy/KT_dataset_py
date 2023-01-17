# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
clicks_test = pd.load_csv('../input/clicks_test.csv')

pd.load_csv('../input/clicks_train.csv')

pd.load_csv('../input/documents_categories.csv')

pd.load_csv('../input/documents_entities.csv')

pd.load_csv('../input/documents_meta.csv')

pd.load_csv('../input/documents_topics.csv')

pd.load_csv('../input/events.csv')

pd.load_csv('../input/page_views_sample.csv')

pd.load_csv('../input/promoted_content.csv')

pd.load_csv('../input/sample_submission.csv')