# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import IFrame

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
IFrame('https://public.tableau.com/views/SpeakerStatistics/SpeakerStatistics?:embed=y&:showVizHome=no', width=1000, height=800)
# embed=y&:display_count=yes&publish=yes
IFrame('https://public.tableau.com/views/TedxEvents/TEDXEventsView?:embed=y&:showVizHome=no', width=1000, height=850)
# https://public.tableau.com/views/TedxEvents/TEDXEventsView?:embed=y&:display_count=yes&publish=yes
IFrame('https://public.tableau.com/views/MostCommentbyTEDTalks/MostCommentbyTEDTalks?:embed=y&:showVizHome=no', width=1000, height=850)
# https://public.tableau.com/views/MostCommentbyTEDTalks/MostCommentbyTEDTalks?:embed=y&:display_count=yes&publish=yes
IFrame('https://public.tableau.com/views/AuthorbyComment/AuthorbyComment?:embed=y&:showVizHome=no', width=1000, height=850)
# https://public.tableau.com/views/AuthorbyComment/AuthorbyComment?:embed=y&:display_count=yes&publish=yes
IFrame('https://public.tableau.com/views/TedxSeasonbyViewCount/TEDxSeasonByViewCount?:embed=y&:showVizHome=no', width=1000, height=850)
# https://public.tableau.com/views/TedxSeasonbyViewCount/TEDxSeasonByViewCount?:embed=y&:display_count=yes&publish=yes
