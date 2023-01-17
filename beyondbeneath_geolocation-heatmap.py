import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
%matplotlib inline
# Load events
df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

# Sample it down to only the China region
lon_min, lon_max = 75, 135
lat_min, lat_max = 15, 55

idx_china = (df_events["longitude"]>lon_min) &\
            (df_events["longitude"]<lon_max) &\
            (df_events["latitude"]>lat_min) &\
            (df_events["latitude"]<lat_max)

df_events_china = df_events[idx_china]

# Sample the China events
df_events_sample = df_events_china.sample(n=1000)
# Gaussian function
# Modified from: https://gist.github.com/andrewgiessel/4635563
def gaussian_blob(size, fwhm, center=None):
    # Get max dimension
    max_dim = max(size[0], size[1])
    
    # Produce the axes
    x = np.arange(0, max_dim, 1)
    y = x[:, np.newaxis]

    # Define the center
    if center is None:
        x0 = size[0] // 2
        y0 = size[1] // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    # Calculate the Gaussian
    blob = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
    # Reshape the 'canvas' back to the right dimensions
    if (size[0]<size[1]):
        blob = blob[:,0:size[0]] # y > x
    else:
        blob = blob[0:size[1],:] # x >= y
    
    # Return the reshaped canvas
    return blob
    
# Test function
# Produce a blob of FWHM=50 at (x=200, y=50) on a (x=500 * y=200) canvas
plt.figure(0)
blob = gaussian_blob(size=[500,200], fwhm=50, center=[200,50])
plt.imshow(blob, cmap=plt.cm.viridis)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# Function to produce a "standard" map of China
# Will be reused a bunch of times
def create_china_map():
    # Define map limits
    lat_min, lat_max = 15, 55
    lon_min, lon_max = 75, 135
    
    # Setup the map
    m = Basemap(projection='merc',
                 llcrnrlat=lat_min,
                 urcrnrlat=lat_max,
                 llcrnrlon=lon_min,
                 urcrnrlon=lon_max,
                 lat_ts=35,
                 resolution='c')
    
    # Format it
    m.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
    m.drawmapboundary(fill_color='#000000')                # black background
    m.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
    
    # Return
    return m
m = create_china_map()

# Get the dimensions of the map in (x,y) coordinate system
x1, y1 = m(lon_min, lat_min)
x2, y2 = m(lon_max, lat_max)
w, h = (x2-x1)*1.0, (y2-y1)*1.0
max_dim = np.max([w, h])

# Produce the canvas and scaling factors
max_canvas = 500
scale_factor = max_canvas / max_dim
if (w>h): canvas_size = [max_canvas, int(max_canvas * (h/w))]
elif (w<h): canvas_size = [int(max_canvas * (w/h)), max_canvas]
else: canvas_size = (max_canvas, max_canvas)

# Cycle through all the coordinates and produce canvases with blobs of different sizes
alpha_total1 = np.zeros((canvas_size[1], canvas_size[0]))
alpha_total2 = np.zeros((canvas_size[1], canvas_size[0]))
for lon, lat in zip(df_events_sample["longitude"].tolist(), df_events_sample["latitude"].tolist()):
    cx, cy = m(lon, lat)
    cx, cy = cx*scale_factor, cy*scale_factor
    g1 = gaussian_blob(canvas_size, fwhm=10, center=(cx,cy))
    g2 = gaussian_blob(canvas_size, fwhm=30, center=(cx,cy))
    alpha_total1 += g1
    alpha_total2 += g2
    
# Normalise this
alpha_norm1 = alpha_total1 / alpha_total1.max()
alpha_norm2 = alpha_total2 / alpha_total2.max()

# See what the raw image looks like
plt.close()
plt.figure()
plt.subplot(121)
plt.imshow(alpha_norm1)
plt.title("alpha_norm1")

plt.subplot(122)
plt.imshow(alpha_norm2)
plt.title("alpha_norm2")
plt.show()
plt.figure(2, figsize=(12,8))
plt.clf()

# Plot the different versions
plt.subplot(231)

alpha_colours1 = plt.cm.rainbow(alpha_norm1)
alpha_colours1[:,:,3] = alpha_norm1
m2a = create_china_map()
m2a.imshow(alpha_colours1, zorder=3)
plt.title("Normal alpha")

plt.subplot(232)
alpha_colours1 = plt.cm.rainbow(alpha_norm1)
alpha_colours1[:,:,3] = np.sqrt(alpha_norm1)
m2b = create_china_map()
m2b.imshow(alpha_colours1, zorder=3)
plt.title("sqrt(alpha)")

plt.subplot(233)
alpha_colours1 = plt.cm.rainbow(alpha_norm1)
alpha_colours1[:,:,3] = np.sqrt(np.sqrt(np.sqrt(alpha_norm1)))
m2c = create_china_map()
m2c.imshow(alpha_colours1, zorder=3)
plt.title("sqrt(sqrt(sqrt(alpha)))")

plt.subplot(234)
alpha_colours2 = plt.cm.plasma(alpha_norm2)
alpha_colours2[:,:,3] = alpha_norm2
m2d = create_china_map()
m2d.imshow(alpha_colours2, zorder=3)

plt.subplot(235)
alpha_colours2 = plt.cm.plasma(alpha_norm2)
alpha_colours2[:,:,3] = np.sqrt(alpha_norm2)
m2e = create_china_map()
m2e.imshow(alpha_colours2, zorder=3)

plt.subplot(236)
alpha_colours2 = plt.cm.plasma(alpha_norm2)
alpha_colours2[:,:,3] = np.sqrt(np.sqrt(np.sqrt(alpha_norm2)))
m2f = create_china_map()
m2f.imshow(alpha_colours2, zorder=3)

plt.tight_layout()
plt.show()
# Show one at full size - note you can also tweak the colour mapping
plt.figure(3, figsize=(12,8))
alpha_colours1 = plt.cm.rainbow(alpha_norm1**0.5)
alpha_colours1[:,:,3] = alpha_norm1**0.5**3
m3 = create_china_map()
m3.imshow(alpha_colours1, zorder=3)
plt.show()
# Show another one, with larger blobs
plt.figure(4, figsize=(12,8))
alpha_colours1 = plt.cm.plasma(alpha_norm2)
alpha_colours1[:,:,3] = alpha_norm2**0.1
m4 = create_china_map()
m4.imshow(alpha_colours1, zorder=3, alpha=0.75)
plt.show()