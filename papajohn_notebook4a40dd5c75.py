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
# Run only once

scorecard_df = pd.read_csv('../input/Scorecard.csv')
# Print feature names

scorecard_df.columns
# Select only the public institutions

public_institutions = scorecard_df.loc[scorecard_df.CONTROL == 'Public']

private_institutions = scorecard_df.loc[scorecard_df.CONTROL == 'Private for-profit']
private_institutions.COSTT4_A