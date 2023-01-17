# Import necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





from os import listdir

from os.path import join, isfile, isdir

from glob import glob





from PIL import Image

sns.set()

from tqdm import tqdm

%matplotlib inline









from keras.preprocessing.image import ImageDataGenerator

data_dir1 = '../input/data/'

data_dir2 = '../input/chestxray8-dataframe/'

train_df = pd.read_csv(data_dir1 + 'Data_Entry_2017.csv')

image_label_map = pd.read_csv(data_dir2 + 'train_df.csv')

bad_labels = pd.read_csv(data_dir2 + 'cxr14_bad_labels.csv')



# Listing all the .jpg filepaths

image_paths = glob(data_dir1+'images_*/images/*.png')

print(f'Total image files found : {len(image_paths)}')

print(f'Total number of image labels: {image_label_map.shape[0]}')

print(f'Unique patients: {len(train_df["Patient ID"].unique())}')



image_label_map.drop(['No Finding'], axis = 1, inplace = True)

labels = image_label_map.columns[2:-1]

labels





train_df.rename(columns={"Image Index": "Index"}, inplace = True)

image_label_map.rename(columns={"Image Index": "Index"}, inplace = True)

train_df = train_df[~train_df.Index.isin(bad_labels.Index)]

train_df.shape



Index =[]

for path in image_paths:

    Index.append(path.split('/')[5])

index_path_map = pd.DataFrame({'Index':Index, 'FilePath': image_paths})

index_path_map.head()



# Merge the absolute path of the images to the main dataframe

pd.merge(train_df, index_path_map, on='Index', how='left')
pd.merge(train_df, index_path_map, on='Index', how='left')
IMAGE_SIZE=[256, 256]

EPOCHS = 20

# BATCH_SIZE = 8 * strategy.num_replicas_in_sync

BATCH_SIZE = 64
def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):

    """

    Return generator for training set, normalizing using batch

    statistics.



    Args:

      train_df (dataframe): dataframe specifying training data.

      image_dir (str): directory where image files are held.

      x_col (str): name of column in df that holds filenames.

      y_cols (list): list of strings that hold y labels for images.

      batch_size (int): images per batch to be fed into model during training.

      seed (int): random seed.

      target_w (int): final width of input images.

      target_h (int): final height of input images.

    

    Returns:

        train_generator (DataFrameIterator): iterator over training set

    """        

    print("getting train generator...")

    # normalize images

    image_generator = ImageDataGenerator(

        samplewise_center=True,

        samplewise_std_normalization= True, 

        shear_range=0.1,

        zoom_range=0.15,

        rotation_range=5,

        width_shift_range=0.1,

        height_shift_range=0.05,

        horizontal_flip=True, 

        vertical_flip = False, 

        fill_mode = 'reflect')

    

    

    # flow from directory with specified batch size

    # and target image size

    generator = image_generator.flow_from_dataframe(

            dataframe=df,

            directory=image_dir,

            x_col=x_col,

            y_col=y_cols,

            class_mode="raw",

            batch_size=batch_size,

            shuffle=shuffle,

            seed=seed,

            target_size=(target_w,target_h))

    

    return generator



train_generator = get_train_generator(df = image_label_map,

                                      image_dir = None, 

                                      x_col = 'FilePath',

                                      y_cols = labels, 

                                      batch_size=BATCH_SIZE,

                                      target_w = IMAGE_SIZE[0], 

                                      target_h = IMAGE_SIZE[1] 

                                      )
X, Y = train_generator.next()



def get_label(y):



    ret_labels = []

    for idx in range(len(y)):

        if y[idx]: ret_labels.append(labels[idx])

    if len(ret_labels):  return '|'.join(ret_labels)

    else: return 'No Label'



rows = int(np.floor(np.sqrt(X.shape[0])))

cols = int(X.shape[0]//rows)

fig = plt.figure(figsize=(20,15))

for i in range(1, rows*cols+1):

    fig.add_subplot(rows, cols, i)

    plt.imshow(X[i-1], cmap='gray')

    plt.title(get_label(Y[i-1]))

    plt.axis(False)

    fig.add_subplot
import bokeh

import IPython.display as ipd

from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, LinearAxis, Range1d

from bokeh.models.tools import HoverTool

from bokeh.palettes import BuGn4, cividis

from bokeh.plotting import figure, output_notebook, show, output_file

from bokeh.transform import cumsum

from bokeh.palettes import Category20b



output_notebook()

diagnosis = ['Normal', 'Sick' ]

counts = [(train_df['Finding Labels'] == 'No Finding').sum(), train_df.shape[0]- (train_df['Finding Labels'] == 'No Finding').sum()]

source = ColumnDataSource(pd.DataFrame({'Type':diagnosis,'Counts':counts, 'color':['#054000', '#e22d00']}))



tooltips = [

    ("Category", "@Type"),

    ("No of Samples", "@Counts")

]



normal_vs_sick = figure(x_range=diagnosis, y_range=(0,70000), plot_height=400, plot_width = 400, title="Normal vs Sick Distribution", tooltips = tooltips)

normal_vs_sick.vbar(x='Type', top='Counts', width=0.75, legend_field="Type", color = 'color', source=source)

normal_vs_sick.xgrid.grid_line_color = None

normal_vs_sick.legend.orientation = "vertical"

normal_vs_sick.legend.location = "top_right"

show(normal_vs_sick)





data = image_label_map[labels].sum(axis=0).sort_values(ascending = True)



# bokeh packages



diagnosis = data.index.tolist()

source = ColumnDataSource(data=dict(diagnosis=data.index.tolist(), counts=data.tolist(), color = Category20b[len(data)]))



tooltips = [("Diagnosis", "@diagnosis"), ("Count", "@counts") ]

diag_dist = figure(x_range=diagnosis, y_range=(0,15000), plot_height=400, plot_width = 700, title="Diagnosis Distributions", tooltips = tooltips)

diag_dist.vbar(x='diagnosis', top='counts', width=0.65, color='color', legend_field="diagnosis", source=source)



diag_dist.xgrid.grid_line_color = None

diag_dist.legend.orientation = "vertical"

diag_dist.legend.location = "top_left"



# show(diag_dist)









def plot_pie_bokeh(data = None):

    from math import pi

    from bokeh.palettes import Category20c

    x = data.to_dict()



    data = pd.Series(x).reset_index(name='value').rename(columns={'index':'category'})

    data['angle'] = data['value']/data['value'].sum() * 2*pi

    data['color'] = Category20b[len(x)]

    p = figure(plot_height=400, plot_width = 700, title="Pie Chart", tooltips="@category: @value%", x_range=(-0.5, 1.0))

    p.wedge(x=0.38, y=1, radius=0.4, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

            line_color="black", fill_color='color', legend_field='category', source=data)



    p.axis.axis_label=None

    p.axis.visible=False

    p.grid.grid_line_color = None



    p.legend.orientation = "vertical"

    p.legend.location = "top_left"

    

    return p





dist_diag_percent = plot_pie_bokeh(data/data.sum()*100)



show(column(diag_dist, dist_diag_percent))
show(plot_pie_bokeh(data/data.sum()*100))
train_df.rename(columns={"Patient Age": "PatientAge"}, inplace = True)

train_df[train_df['PatientAge'] > 100]
average_age = int(train_df[train_df['PatientAge'] < 100]['PatientAge'].mean())

for idx in range(train_df.shape[0]):

    if train_df.iloc[idx, 4] > 100:

        print(f'{train_df.iloc[idx, 0]} : age {train_df.iloc[idx, 4]} is changed to ->> {average_age}')

        train_df.iloc[idx, 4] = average_age



train_df[train_df['PatientAge'] > 100]
def hist_hover(data, column=None,  title = 'Histogram',  colors=["SteelBlue", "Tan"], bins=30, log_scale=False, show_plot=True):



    # build histogram data with Numpy

    hist, edges = np.histogram(data, bins = bins)



    hist_df = pd.DataFrame({column: hist, "left": edges[:-1], "right": edges[1:]})

    hist_df["interval"] = ["%d to %d" % (left, right) for left, 

                           right in zip(hist_df["left"], hist_df["right"])]



    # bokeh histogram with hover tool

    if log_scale == True:

        hist_df["log"] = np.log(hist_df[column])

        src = ColumnDataSource(hist_df)

        plot = figure(plot_height = 300, plot_width = 600,

              title = title,

              x_axis_label = column.capitalize(),

              y_axis_label = "Log Count")    

        plot.quad(bottom = 0, top = "log",left = "left", 

            right = "right", source = src, fill_color = colors[0], 

            line_color = "black", fill_alpha = 0.7,

            hover_fill_alpha = 1.0, hover_fill_color = colors[1])

    else:

        src = ColumnDataSource(hist_df)

        plot = figure(plot_height = 300, plot_width = 600,

            title = title,

              x_axis_label = column.capitalize(),

              y_axis_label = "Count")    

        plot.quad(bottom = 0, top = column,left = "left", 

            right = "right", source = src, fill_color = colors[0], 

            line_color = "black", fill_alpha = 0.7,

            hover_fill_alpha = 1.0, hover_fill_color = colors[1])

    # hover tool

    hover = HoverTool(tooltips = [(' Age Interval', '@interval'),

                              ('Sample Count', str("@" +str(column)))])

    plot.add_tools(hover)

    # output

    if show_plot == True:

        show(plot)

    else:

        return plot
hist_hover(train_df['PatientAge'], column = 'PatientAge', bins = 100)
train_df[train_df['Patient Age'] > 100 ]
ages_male = train_df.loc[(train_df["Patient Gender"] == 'M'), "PatientAge"].tolist()

ages_female = train_df.loc[(train_df["Patient Gender"] == 'F'), "PatientAge"].tolist()
show(column(hist_hover(ages_male, column = 'MaleAges', title = 'Male Patients Age Histogram', bins = 95, show_plot=False),

            hist_hover(ages_female, column = 'FemaleAges', title = 'Female Patients Age Histogram',  bins = 95, show_plot=False)))
train_df.PatientAge.max() - train_df.PatientAge.min()