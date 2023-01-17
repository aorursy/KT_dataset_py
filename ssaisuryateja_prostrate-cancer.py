# Importing all the needed libraries
import os
import openslide
import skimage.io
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display
import plotly.graph_objs as go
# Train and Testing data
import pandas as pd
train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
train.columns.to_list()
test = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')
test.columns.to_list()
import matplotlib.pyplot as plt
# Plot histogram
train.data_provider.hist(bins=30, alpha=0.5)
plt.show()
# Location of the training images
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'
def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):
    """Print some basic information about a slide"""
    # Generate a small image thumbnail
    if show_thumbnail:
        display(slide.get_thumbnail(size=max_size))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")

# Open the image (does not yet read the image into memory)
image = openslide.OpenSlide(os.path.join(data_dir, '005e66f06bce9c2e49142536caf2f6ee.tiff'))

# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.
# At this point image data is read from the file and loaded into memory.
patch = image.read_region((17800,19500), 0, (256, 256))

# Display the image
display(patch)

# Close the opened slide after use
image.close()
example_slides = [
    '005e66f06bce9c2e49142536caf2f6ee',
    '00928370e2dfeb8a507667ef1d4efcbb',
    '007433133235efc27a39f11df6940829',
    '024ed1244a6d817358cedaea3783bbde',
]

for case_id in example_slides:
    biopsy = openslide.OpenSlide(os.path.join(data_dir, f'{case_id}.tiff'))
    print_slide_details(biopsy)
    biopsy.close()
    
    # Print the case-level label
    print(f"ISUP grade: {train_labels.loc[case_id, 'isup_grade']}")
    print(f"Gleason score: {train_labels.loc[case_id, 'gleason_score']}\n\n")
biopsy = openslide.OpenSlide(os.path.join(data_dir, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))

x = 5150
y = 21000
level = 0
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)

x = 5140
y = 21000
level = 1
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)
biopsy.close()
def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):
    """Print some basic information about a slide"""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Generate a small image thumbnail
    if show_thumbnail:
        # Read in the mask data from the highest level
        # We cannot use thumbnail() here because we need to load the raw label data.
        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        # Mask data is present in the R channel
        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')
        mask_data.thumbnail(size=max_size, resample=0)
        display(mask_data)

    # Compute microns per pixel (openslide gives resolution in centimeters)
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")
mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
print_mask_details(mask, center='radboud')
mask.close()
mask = openslide.OpenSlide(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
print_mask_details(mask, center='karolinska')
mask.close()
mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

plt.figure()
plt.title("Mask with default cmap")
plt.imshow(np.asarray(mask_data)[:,:,0], interpolation='nearest')
plt.show()

plt.figure()
plt.title("Mask with custom cmap")
# Optional: create a custom color map
cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
plt.imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
plt.show()

mask.close()
def overlay_mask_on_slide(slide, mask, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Load data from the highest level
    slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

    # Mask data is present in the R channel
    mask_data = mask_data.split()[0]

    # Create alpha mask
    alpha_int = int(round(255*alpha))
    if center == 'radboud':
        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
    elif center == 'karolinska':
        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)
    
    alpha_content = PIL.Image.fromarray(alpha_content)
    preview_palette = np.zeros(shape=768, dtype=int)
    
    if center == 'radboud':
        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
    elif center == 'karolinska':
        # Mapping: {0: background, 1: benign, 2: cancer}
        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)
    
    mask_data.putpalette(data=preview_palette.tolist())
    mask_rgb = mask_data.convert(mode='RGB')

    overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
    overlayed_image.thumbnail(size=max_size, resample=0)

    display(overlayed_image)
slide = openslide.OpenSlide(os.path.join(data_dir, '08ab45297bfe652cc0397f4b37719ba1.tiff'))
mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
overlay_mask_on_slide(slide, mask, center='radboud')
slide.close()
mask.close()
slide = openslide.OpenSlide(os.path.join(data_dir, '090a77c517a7a2caa23e443a77a78bc7.tiff'))
mask = openslide.OpenSlide(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
overlay_mask_on_slide(slide, mask, center='karolinska', alpha=0.6)
slide.close()
mask.close()
biopsy = skimage.io.MultiImage(os.path.join(data_dir, '0b373388b189bee3ef6e320b841264dd.tiff'))
for i,level in enumerate(biopsy):
    print(f"Biopsy level {i} dimensions: {level.shape}")
    print(f"Biopsy level {i} memory size: {level.nbytes / 1024**2:.1f}mb")
display(PIL.Image.fromarray(biopsy[-1]))
# Deleting the object frees up memory
del biopsy
biopsy_level_0 = skimage.io.imread(os.path.join(data_dir, '0b373388b189bee3ef6e320b841264dd.tiff'))
print(biopsy_level_0.shape)
del biopsy_level_0
biopsy = skimage.io.MultiImage(os.path.join(data_dir, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))

x = 5150
y = 21000
level = 0
width = 512
height = 512

patch = biopsy[0][y:y+width, x:x+height]

# You can also visualize patches with matplotlib
plt.figure()
plt.imshow(patch)
plt.show()

x = 5150 // 4
y = 21000 // 4
width = 512
height = 512

patch = biopsy[1][y:y+width, x:x+height]

plt.figure()
plt.imshow(patch)
plt.show()
x = 5150 // (4*4)
y = 21000 // (4*4)
width = 512
height = 512

patch = biopsy[2][y:y+width, x:x+height]

plt.figure()
plt.imshow(patch)
plt.show()
# Free up memory
del biopsy
maskfile = skimage.io.MultiImage(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
mask_level_2 = maskfile[-1][:,:,0]

plt.figure()
plt.imshow(mask_level_2)
plt.colorbar()
plt.show()
del maskfile
class WSIViewer(object):
    def __init__(self, plot_size = 1000):
        self._plot_size = plot_size
        
    def set_slide(self, slide_path):      
        self._slide = openslide.open_slide(slide_path)
        self._base_dims = self._slide.level_dimensions[-1]
        self._base_ds = self._slide.level_downsamples[-1]
        img_arr = self._slide.read_region((0,0), len(self._slide.level_dimensions[-1]), (self._base_dims[0], self._base_dims[1]))
        
        self._fig = go.FigureWidget(data=[{'x': [0, self._base_dims[0]], 
                                           'y': [0, self._base_dims[1]], 
                                           'mode': 'markers',
                                           'marker': {'opacity': 0}}], # invisible trace to init axes and to support autoresize
                                    layout={'width': self._plot_size, 'height': self._plot_size, 'yaxis' : dict(scaleanchor = "x", scaleratio = 1)})  
        # Set background image
        self._fig.layout.images = [go.layout.Image(
            source = img_arr,  # plotly now performs auto conversion of PIL image to png data URI
            xref = "x",
            yref = "y",
            x = 0,
            y = 0,
            sizex = self._base_dims[0],
            sizey = self._base_dims[1],
            sizing = "stretch",
            layer = "below")]
        self._fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False);        
        self._fig.layout.on_change(self._update_image, 'xaxis.range', 'yaxis.range', 'width', 'height')          

    def _gen_zoomed_image(self, x_range, y_range):
        # Below is a workaround which rounds image requests to multiples of 4, once the libpixman fix is in place these can be removed
        #xstart = x_range[0] * self._base_ds
        #ystart = (self._base_dims[1] - y_range[1]) * self._base_ds 
        xstart = 4 * round(x_range[0] * self._base_ds / 4)
        ystart = 4 * round((self._base_dims[1] - y_range[1]) * self._base_ds / 4)
        xsize0 = (x_range[1] - x_range[0]) * self._base_ds
        ysize0 = (y_range[1] - y_range[0]) * self._base_ds
        if (xsize0 > ysize0):
            req_downs = xsize0 / self._plot_size
        else:
            req_downs = ysize0 / self._plot_size
        req_level = self._slide.get_best_level_for_downsample(req_downs)
        level_downs = self._slide.level_downsamples[req_level]
        # Nasty workaround for buggy container
        level_size_x = int(xsize0 / level_downs)
        level_size_y = int(ysize0 / level_downs)
        new_img = self._slide.read_region((int(xstart), int(ystart)), req_level, (level_size_x, level_size_y)).resize((1000,1000)) # Letting PIL do the resize is faster than plotly
        return new_img
    
    def _update_image(self, layout, x_range, y_range, plot_width, plot_height):
        img = self._fig.layout.images[0]
        # Update with batch_update so all updates happen simultaneously
        with self._fig.batch_update():
            new_img = self._gen_zoomed_image(x_range, y_range)
            img.x = x_range[0]
            img.y = y_range[1]
            img.sizex = x_range[1] - x_range[0]
            img.sizey = y_range[1] - y_range[0]
            img.source = new_img

    def show(self):
        return self._fig
viewer = WSIViewer()
viewer.set_slide(os.path.join(data_dir, '08ab45297bfe652cc0397f4b37719ba1.tiff'))
viewer.show()
sample_sub = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')