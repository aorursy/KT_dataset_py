# show mask hotmaps for depth in (0,400), (400,700),(700,959)
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

f = open('../input/depths.csv')
lines = f.readlines()
f.close()
del lines[0]

lines_train = lines[:4000]

hotmap1 = np.zeros((101,101))
hotmap2 = np.zeros((101,101))
hotmap3 = np.zeros((101,101))
count1 = 0
count2 = 0
count3 = 0

for l in lines_train:
    id = l.split(',')[0]
    z = float(l.split(',')[1])
    if z<400:
        hotmap1 += io.imread('../input/train/masks/'+id+'.png')/65535.
        count1 += 1
    elif 400<z<700:
        hotmap2 += io.imread('../input/train/masks/'+id+'.png')/65535.
        count2 += 1
    elif z>700:
        hotmap3 += io.imread('../input/train/masks/'+id+'.png')/65535.
        count3 += 1

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()
ax[0].imshow(hotmap1, cmap='seismic')
ax[0].set_title('z<400')
ax[1].imshow(hotmap2, cmap='seismic')
ax[1].set_title('400z<700')
ax[2].imshow(hotmap3, cmap='seismic')
ax[2].set_title('z>700')

fig.tight_layout()
plt.show()

bad_masks =('1eaf42beee','33887a0ae7','33dfce3a76','3975043a11','39cd06da7d','483b35d589','49336bb17b','4ef0559016','4fbda008c7','4fdc882e4b','50d3073821','53e17edd83','5b217529e7','5f98029612','608567ed23','62aad7556c','62d30854d7','6460ce2df7','6bc4c91c27','7845115d01','7deaf30c4a','80a458a2b6','81fa3d59b8','8367b54eac','849881c690','876e6423e6','90720e8172','916aff36ae','919bc0e2ba','a266a2a9df','a6625b8937','a9ee40cf0d','aeba5383e4','b63b23fdc9','baac3469ae','be7014887d','be90ab3e56','bfa7ee102e','bfbb9b9149','c387a012fc','c98dfd50ba','caccd6708f','cb4f7abe67','d0bbe4fd97','d4d2ed6bd2','de7202d286','f0c401b64b','f19b7d20bb','f641699848','f75842e215','00950d1627','0280deb8ae','06d21d76c4','09152018c4','09b9330300','0b45bde756','130229ec15','15d76f1672','182bfc6862','23afbccfb5','24522ec665','285f4b2e82','2bc179b78c','2f746f8726','3cb59a4fdc','403cb8f4b3','4f5df40ab2','50b3aef4c4','52667992f8','52ac7bb4c1','56f4bcc716','58de316918','640ceb328a','71f7425387','7c0b76979f','7f0825a2f0','834861f1b6','87afd4b1ca','88a5c49514','9067effd34','93a1541218','95f6e2b2d1','96216dae3b','96523f824a','99ee31b5bc','9a4b15919d','9b29ca561d','9eb4a10b98','ad2fa649f7','b1be1fa682','b24d3673e1','b35b1b412b','b525824dfc','b7b83447c4','b8a9602e21','ba1287cb48','be18a24c49','c27409a765','c2973c16f1','c83d9529bd','cef03959d8','d4d34af4f7','d9a52dc263','dd6a04d456','ddcb457a07','e12cd094a6','e6e3e58c43','e73ed6e7f2','f6e87c1458','f7380099f6','fb3392fee0','fb47e8e74e','febd1d2a67')

bad_mask_hotmap = np.zeros((101,101))

count = 0
for l in lines_train:
    id = l.split(',')[0]
    if id in bad_masks:
        bad_mask_hotmap += io.imread('../input/train/masks/'+id+'.png')/65535.
        count += 1

plt.imshow(bad_mask_hotmap, cmap='seismic')
plt.show()