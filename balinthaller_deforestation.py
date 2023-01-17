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
indicators = pd.read_csv("../input/Indicators.csv")
indicators.drop(['IndicatorName'], 1, inplace=True)
forest = indicators[indicators.IndicatorCode == 'AG.LND.FRST.ZS'].copy()
forest.drop(['IndicatorCode'], 1, inplace=True)
latest = forest.groupby(['CountryCode'])['Year'].transform(max) == forest['Year']
previous = forest.groupby('CountryCode')['Year'].transform(max)-5 == forest['Year']
deforest = forest[previous].loc[:,'Value'].as_matrix() - forest[latest].loc[:,'Value'].as_matrix()
deforestation = forest[latest].copy()
deforestation.loc[:,'Deforestation'] = deforest
deforestation.drop(['Value', 'Year'], 1, inplace=True)
deforestation.sort_values('Deforestation', ascending=False).head(10)
deforestation.sort_values('Deforestation', ascending=True).head(10)