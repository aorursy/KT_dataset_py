!ls /kaggle/input/bacteria-detection-with-darkfield-microscopy
INPUT = "/kaggle/input/bacteria-detection-with-darkfield-microscopy"
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize(filename):
    image_path = f"{INPUT}/images/{filename}.png"
    mask_path = f"{INPUT}/masks/{filename}.png"
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask_applied = image.copy()
    mask_applied[mask == 1] = [255, 0, 0]
    mask_applied[mask == 2] = [255, 255, 0]
    
    out = image.copy()
    mask_applied = cv2.addWeighted(mask_applied, 0.5, out, 0.5, 0, out)
    
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(image)
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(mask, cmap="gray")
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(mask_applied)
visualize("025")
visualize("125")
visualize("225")