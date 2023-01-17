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

BenefitsCostSharing = pd.read_csv("../input/BenefitsCostSharing.csv")
#BusinessRules = pd.read_csv("../input/BusinessRules.csv")
#Crosswalk2015 = pd.read_csv("../input/Crosswalk2015.csv")
#Crosswalk2016 = pd.read_csv("../input/Crosswalk2016.csv")
#Network = pd.read_csv("../input/Network.csv")
#PlanAttributes = pd.read_csv("../input/PlanAttributes.csv")
#Rate = pd.read_csv("../input/Rate.csv")
#ServiceArea = pd.read_csv("../input/ServiceArea.csv")
BenefitsCostSharing.head()
BenefitsCostSharing['BusinessYear'].unique()
