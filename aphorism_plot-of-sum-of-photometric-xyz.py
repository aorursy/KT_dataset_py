# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install colour-science
import colour
import numpy
import matplotlib
CMFS = colour.CMFS["CIE 2012 2 Degree Standard Observer"]
CMFS_sums = numpy.sum(CMFS.values, axis=1)
matplotlib.pyplot.style.use({'figure.figsize': (20, 6)})
matplotlib.pyplot.plot(CMFS.wavelengths, CMFS_sums)