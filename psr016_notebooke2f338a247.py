# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#Read input data and preview

debate_lines = pd.read_csv("../input/debate.csv", encoding= "iso-8859-1")

debate_lines.head(10)
ClintonLines = debate_lines.loc[debate_lines['Speaker'] == 'Clinton']

ClintonLines.head(10)