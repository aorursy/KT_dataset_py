import pandas as pd

df = pd.read_csv("/kaggle/input/historical-income-tax-rates-brackets/tax_over_years.csv",sep=";")

df.drop(['Unnamed: 5','Unnamed: 6'],axis=1,inplace=True)

df.head()
import matplotlib.image as mpimg 

import matplotlib.pyplot as plt



img1 = mpimg.imread('/kaggle/input/viz-screenshots/1.jpg')

img2 = mpimg.imread('/kaggle/input/viz-screenshots/Annotation 2020-05-30 062108.jpg')

img3 = mpimg.imread('/kaggle/input/viz-screenshots/3.jpg')

img4 = mpimg.imread('/kaggle/input/viz-screenshots/4.jpg')

img5 = mpimg.imread('/kaggle/input/viz-screenshots/5.jpg')

img6 = mpimg.imread('/kaggle/input/viz-screenshots/6.jpg')
plt.figure(figsize=(15,10))

plt.imshow(img1)

plt.axis('off')
plt.figure(figsize=(15,10))

plt.imshow(img2)

plt.axis('off')
plt.figure(figsize=(15,10))

plt.imshow(img3)

plt.axis('off')
plt.figure(figsize=(15,10))

plt.imshow(img4)

plt.axis('off')
plt.figure(figsize=(15,10))

plt.imshow(img5)

plt.axis('off')
plt.figure(figsize=(15,10))

plt.imshow(img6)

plt.axis('off')