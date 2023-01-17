# Loading necessary libraries



import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# For plotting

import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})



#For standardising the dat

from sklearn.preprocessing import StandardScaler



#Ignore warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

#Setting the label and the feature columns

y = train.loc[:,'label'].values

x = train.loc[:,'pixel0':].values



print(x.shape)

print(y)
# Subsetting a data for faster execution



x_subset = x[0:10000]

y_subset = y[0:10000]



print(np.unique(y_subset))
import umap

reducer = umap.UMAP(random_state=42)

embedding = reducer.fit_transform(x_subset)

embedding.shape
plt.scatter(reducer.embedding_[:, 0], reducer.embedding_[:, 1], s= 5, c=y_subset, cmap='Spectral')

plt.gca().set_aspect('equal', 'datalim')

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))

plt.title('Visualizing Kannada MNIST with UMAP', fontsize=24);
# Encoding all the images for inclusion in a dataframe.

from io import BytesIO

from PIL import Image

import base64





def embeddable_image(data):

    img_data = 255 - 15 * data.astype(np.uint8)

    image = Image.fromarray(img_data, mode='L').resize((28,28), Image.BICUBIC)

    buffer = BytesIO()

    image.save(buffer, format='png')

    for_encoding = buffer.getvalue()

    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()



# loading up bokeh and other tools to generate a suitable interactive plot.



from bokeh.plotting import figure, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper

from bokeh.palettes import Spectral10



output_notebook()
# Generating the plot itself with a custom hover tooltip 



x_subset_reshape = x_subset.reshape(10000,28,28)



digits_df = pd.DataFrame(embedding, columns=('x', 'y'))

digits_df['digit'] = [str(x) for x in y_subset]

digits_df['image'] = list(map(embeddable_image, x_subset_reshape))





datasource = ColumnDataSource(digits_df)

color_mapping = CategoricalColorMapper(factors=[str(9 - x) for x in y_subset],

                                       palette=Spectral10)



plot_figure = figure(

    title='UMAP projection of the Kannada MNIST dataset',

    plot_width=600,

    plot_height=600,

    tools=('pan, wheel_zoom, reset')

)



plot_figure.add_tools(HoverTool(tooltips="""

<div>

    <div>

        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>

    </div>

    <div>

        <span style='font-size: 16px; color: #224499'>Digit:</span>

        <span style='font-size: 18px'>@digit</span>

    </div>

</div>

"""))



plot_figure.circle(

    'x',

    'y',

    source=datasource,

    color=dict(field='digit', transform=color_mapping),

    line_alpha=0.6,

    fill_alpha=0.6,

    size=4

)

show(plot_figure)