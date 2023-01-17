# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image
import os
Image("../input/1.png")
Image("../input/2 .png")
Image("../input/tags.png")
Image("../input/stemming.png")
Image("../input/remove_degits.png")
Image("../input/tokenize.png")
Image("../input/freqDist.png")
Image("../input/stopwords.png")
Image("../input/Polarity.png")
Image("../input/histogramme of polarities.png")
Image("../input/histogramme of polarities 2.png")
Image("../input/CM Bayes.png")
Image("../input/SVM.png")