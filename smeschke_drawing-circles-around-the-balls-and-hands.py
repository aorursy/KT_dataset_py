# This notebook explores a data set about juggling!
# These tools are used to open the files and draw images
import matplotlib.pyplot as plt # plotting
import cv2
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
print('imports complete')

# Print some of the file names in the data set
print(os.listdir('../input/ballsandhands/data/annotations')[:5])
# Let's take a look at the first one
df = pd.read_csv('../input/ballsandhands/data/annotations/2-white-circus-sink.csv')
# Describe the data set
df.describe()
# Print the first few values, could they be coordinates?
for i in range(5):
    print(df.values[i])
# Hmmm...looks like the first value is an image. Let's look at that
# Read the image
img = cv2.imread('../input/ballsandhands/data/frames/2-white-circus-sink-002.png')
# Plot the image
imgplot = plt.imshow(img)
# Cool, so it's an image of juggling!
# Now let's try to draw some circles around the balls and hands
# First, need to get the data for that specific frame
frame_data = df.values[0]
# Next, draw circles using the coordinates in that frame data
cv2.circle(img, (frame_data[1], frame_data[2]), 5, 255, 2)
cv2.circle(img, (frame_data[3], frame_data[4]), 5, 255, 2)
cv2.circle(img, (frame_data[5], frame_data[6]), 5, 255, 2)
cv2.circle(img, (frame_data[7], frame_data[8]), 5, 255, 2)
# Show the image with circles
imgplot = plt.imshow(img)