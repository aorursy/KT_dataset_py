import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.layouts

import ipywidgets

input_path = '../input/'
#input_path = './'
bokeh.io.output_notebook()
import sys
for name, module in sorted(sys.modules.items()):
    if name in ['numpy', 'matplotlib', 'pandas', 'ipywidgets']:
        if hasattr(module, '__version__'): 
            print(name, module.__version__)
!python --version
assert(input_path[-1] == '/')
train_labels = pd.read_csv(input_path + "train.csv", index_col=0)
train_labels.head()
def to_rgba(img):
    """Convert a microscopy four channel image 0 to 1 encoded to Bokeh's RGBA encoded
    
    The protein of interest is visualized in green,
    while reference markers for microtubules (red),
    endoplasmic reticulum (yellow) and nucleus (blue) outline the cell.
    We changed the order putting the red first to match the RGB order.
    
    For more information about the images of the Protein Atlas dataset see
    https://www.proteinatlas.org/humancell/organelle
    For more information on microscopy image see:
    https://en.wikipedia.org/wiki/Fluorescence_microscope
    
         Parameters
    ----------
    img : array of shape (4, x_resolution, y_resolution)
        If the image has the size 512x512, then the img parameter must have the shape (4, 512, 512).
        Each of the (512,512) elements of the image is the the overlay image encoded with numbers
        between 0 and 1 (including). Each number is the intensity of the point where 0 means no intensity
        and 1 means max intesity. The input channels are in the following order:
            img[0] is the red channel
            img[1] is the green channel
            img[2] is the blue channel
            img[3] is the yellow channel
        
    Returns
    -------
    numpy.ndarray of shape (4, x_resolution, y_resolution) and dtype('uint32')
        The returned array have the four channels encoded as uint32 numbers in the (tricky) byte order
        expected by Bokeh. The channel 0 takes the color red; 1 takes green; 2 takes blue, and 3 takes
        the color yellow. The intensity is used as an alpha transparency.
    """
    assert np.min(img) >= 0
    assert np.max(img) <= 1
    new_img = np.left_shift((np.array(img) * 255).astype(np.uint32), 24)
    view = new_img.view(dtype=np.uint8).reshape((*new_img.shape, 4))
    view[0, :, :, 0] = 255                         # red
    view[1, :, :, 1] = 255                         # green
    view[2, :, :, 2] = 255                         # blue
    view[3, :, :, 0] = 255; view[3, :, :, 1] = 255 # red + green = yellow

    return new_img
locations = {0:  "Nucleoplasm",
             1:  "Nuclear membrane",
             2:  "Nucleoli",
             3:  "Nucleoli fibrillar center",
             4:  "Nuclear speckles",
             5:  "Nuclear bodies",
             6:  "Endoplasmic reticulum",
             7:  "Golgi apparatus",
             8:  "Peroxisomes",
             9:  "Endosomes",
             10: "Lysosomes",
             11: "Intermediate filaments",
             12: "Actin filaments",
             13: "Focal adhesion sites",
             14: "Microtubules",
             15: "Microtubule ends",
             16: "Cytokinetic bridge",
             17: "Mitotic spindle",
             18: "Microtubule organizing center",
             19: "Centrosome",
             20: "Lipid droplets",
             21: "Plasma membrane",
             22: "Cell junctions",
             23: "Mitochondria",
             24: "Aggresome",
             25: "Cytosol",
             26: "Cytoplasmic bodies",
             27: "Rods & rings"}
%%time
samples = train_labels.Target.str.split(expand=True)
samples.index = train_labels.index
samples.index.name = 'id'
samples = pd.DataFrame(samples.stack().reset_index(drop=True, level=1).astype(int), columns=['target'])
samples.replace({'target': locations}, inplace=True)
samples.head(20)
samples[samples.target == 'Nucleoplasm'].head()
samples.reset_index().groupby('target').count()
def load_img(uuid, path=input_path+'train/'):
    a = []
    assert(path[-1] == '/')
    a.append(plt.imread(path + uuid + '_red.png'))
    a.append(plt.imread(path + uuid + '_green.png'))
    a.append(plt.imread(path + uuid + '_blue.png'))
    a.append(plt.imread(path + uuid + '_yellow.png'))
    return a

%time r = to_rgba(load_img('02ce0bfa-bbc9-11e8-b2bc-ac1f6b6435d0'))
%timeit r = to_rgba(load_img('02ce0bfa-bbc9-11e8-b2bc-ac1f6b6435d0'))
labels = samples.target.unique()
ids = samples[samples.target == labels[0]].reset_index().id.unique()

current_image = ids[0]
source = bokeh.models.ColumnDataSource(data=samples)

r = to_rgba(load_img(current_image))
_, xres, yres = r.shape
picture = bokeh.plotting.figure(x_range=(0, xres), y_range=(0, yres))
picture.background_fill_color = 'black'
ticker = bokeh.models.tickers.FixedTicker(ticks=np.arange(0, 512+128, 128))
picture.xaxis.ticker = ticker
picture.yaxis.ticker = ticker
picture.grid.ticker = ticker
source = bokeh.models.ColumnDataSource(data={'red':    [r[0]],
                                             'green':  [r[1]],
                                             'blue':   [r[2]],
                                             'yellow': [r[3]]})
images = []
images.append(picture.image_rgba(image='red', x=0, y=0, dw=xres, dh=yres, source=source))
images.append(picture.image_rgba(image='green', x=0, y=0, dw=xres, dh=yres, source=source))
images.append(picture.image_rgba(image='blue', x=0, y=0, dw=xres, dh=yres, source=source))
images.append(picture.image_rgba(image='yellow', x=0, y=0, dw=xres, dh=yres, source=source))

text1 = bokeh.models.widgets.Div(text="Current image: <br />" + current_image)
tags = "; ".join(samples.loc[current_image].target.values)
text2 = bokeh.models.widgets.Div(text="Labels: <br />" + tags)

callback = bokeh.models.CustomJS(args={'picture':picture}, code='''\
    if (picture.background_fill_color == 'white') {
        picture.background_fill_color = 'black'
    } else {
        picture.background_fill_color = 'white'
    }
''')
checkbox_background = bokeh.models.widgets.CheckboxGroup(labels=["Black background"],
                                                         active=[0],
                                                         callback=callback)

button_types = ['danger', 'success', 'primary', 'warning']
callbacks = []
toggles = []
for i, label in enumerate(["Microtubules", "Antibody", "Nucleus", "Endoplasmic Reticulum"]):
    callbacks.append(bokeh.models.CustomJS(args={'object': images[i]},
                                           code='object.visible = !object.visible'))
    toggles.append(bokeh.models.widgets.Toggle(label=label,
                                               button_type=button_types[i],
                                               callback=callbacks[i]))

w = bokeh.layouts.widgetbox(checkbox_background, *toggles, text1, text2)

bokeh.plotting.show(bokeh.layouts.row([picture, w]), notebook_handle=True)

def update_img(change):
    if change['type'] != 'change' or change['name'] != 'value':
        return
    global images, picture
    current_image = ids[change.new-1]
    text1.text = "Current image: \n" + current_image
    tags = samples.loc[current_image]
    if tags.size > 1:
        text2.text ="Labels: <br />" + ("; ".join(tags.target))
    else:
        text2.text ="Labels: <br />" + tags.target
    r = to_rgba(load_img(current_image))
    source.update(data={'red':    [r[0]],
                        'green':  [r[1]],
                        'blue':   [r[2]],
                        'yellow': [r[3]]})
    bokeh.io.push_notebook()

slider = ipywidgets.IntSlider(
    value=1,
    min=1,
    max=len(ids),
    step=1,
    description='Image:',
    continuous_update=False,
    orientation='horizontal',
)

slider.observe(update_img)

def update_slider(change):
    if change['type'] != 'change' or change['name'] != 'value':
        return
    global ids, slider
    ids = samples[samples.target == change.new].reset_index().id.unique()
    slider.max = len(ids)
    slider.value = 1
    slider.notify_change({'name': 'value', 'new': 1, 'type': 'change'})

select = ipywidgets.Select(
    options=labels,
    value=labels[0],
    rows=10,
    description='Filter'
)

select.observe(update_slider)

box = ipywidgets.Box([slider, select])
box