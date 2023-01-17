import matplotlib.pyplot as plt

import cartopy.crs as ccrs
fig = plt.figure(figsize=(30, 5))

ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_title('stock_img (shaded relief)', fontsize = 20) 

ax.stock_img()

plt.show()
fig = plt.figure(figsize=(20, 5))

ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_title('bluemarble', fontsize = 20) 

img = plt.imread('../input/global-earth-map-images/bluemarble.png')

img_extent = (-180, 180, -90, 90)

ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())

plt.show()
fig = plt.figure(figsize=(20, 5))

ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_title('etopo', fontsize = 20) 

img = plt.imread('../input/global-earth-map-images/etopo.png')

img_extent = (-180, 180, -90, 90)

ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())

plt.show()