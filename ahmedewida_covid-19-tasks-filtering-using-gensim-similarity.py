# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top Ten authors.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top ten Journals.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top Ten license.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top Ten publish time.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Top sources.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/text_corpus.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/Titles_Words.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/abstract_words.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((5000,4000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
  
from PIL import Image, ImageOps

from matplotlib import image
from matplotlib import pyplot
# load the image
image = Image.open('/kaggle/input/summarydata/elbow_plot.png')
# report the size of the image
# resize image and ignore original aspect ratio
img_resized = image.resize((1000,1000))

# report the size of the thumbnail
#print(img_resized.size) 


# display the array of pixels as an image
pyplot.imshow(img_resized)
pyplot.show()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks1 = pd.read_csv('/kaggle/input/summarydata/Task1.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks1_table=HTML(Tasks1.to_html(escape=False,index=False))
display(Tasks1_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks2 = pd.read_csv('/kaggle/input/summarydata/Task2.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks2_table=HTML(Tasks2.to_html(escape=False,index=False))
display(Tasks2_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks3 = pd.read_csv('/kaggle/input/summarydata/Task3.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks3_table=HTML(Tasks3.to_html(escape=False,index=False))
display(Tasks3_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks4 = pd.read_csv('/kaggle/input/summarydata/Task4.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks4_table=HTML(Tasks4.to_html(escape=False,index=False))
display(Tasks4_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks5 = pd.read_csv('/kaggle/input/summarydata/Task5.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks5_table=HTML(Tasks5.to_html(escape=False,index=False))
display(Tasks5_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks6 = pd.read_csv('/kaggle/input/summarydata/Task6.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks6_table=HTML(Tasks6.to_html(escape=False,index=False))
display(Tasks6_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks7 = pd.read_csv('/kaggle/input/summarydata/Task7.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks7_table=HTML(Tasks7.to_html(escape=False,index=False))
display(Tasks7_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks8 = pd.read_csv('/kaggle/input/summarydata/Task8.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks8_table=HTML(Tasks8.to_html(escape=False,index=False))
display(Tasks8_table)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import functools
from IPython.core.display import display, HTML
Tasks9 = pd.read_csv('/kaggle/input/summarydata/Task9.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks9_table=HTML(Tasks9.to_html(escape=False,index=False))
display(Tasks9_table)