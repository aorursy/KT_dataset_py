import os
import ast
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.DataFrame()
for file in os.listdir('../input/train_simplified/'):
    train = train.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', nrows=1))
# You may choose the shape you want.
def drawing_to_np(drawing, shape=(64, 64)):
    # evaluates the drawing array
    drawing = eval(drawing)
    fig, ax = plt.subplots()
    # Close figure so it won't get displayed while transforming the set
    plt.close(fig)
    for x,y in drawing:
        ax.plot(x, y, marker='.')
        ax.axis('off')        
    fig.canvas.draw()
    # Convert images to numpy array
    np_drawing = np.array(fig.canvas.renderer._renderer)
    # If you want to take only one channel, you can try somethin like:
    # np_drawing = np_drawing[:, :, 1]    
    return cv2.resize(np_drawing, shape) # Resize array
# One way you could apply the transformation to you dataset.
train['drawing_np'] = train['drawing'].map(drawing_to_np)
train['drawing_np2'] = train['drawing'].apply(drawing_to_np)
train.head(10)
drawings = [ast.literal_eval(pts) for pts in train['drawing'].head(1).values]

plt.figure(figsize=(10, 10))
for i, drawing in enumerate(drawings):
    plt.subplot(330 + (i+1))
    for x,y in drawing:
        plt.plot(x, y, marker='.')
        plt.axis('off')
# Function to plot images.
def plot_image(image_array):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, frameon=False)
    ax2.imshow(image_array)
    plt.axis('off')
    plt.show()
    print('Image shape:', image_array.shape)

sample_1 = train['drawing_np'].values[0]
plot_image(sample_1)
sample_2 = train['drawing_np2'].values[0]
plot_image(sample_2)