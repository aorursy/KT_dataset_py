import matplotlib.pyplot as plt

import os

os.listdir("../input/wheres-waldo/Hey-Waldo/")
os.listdir("../input/wheres-waldo/Hey-Waldo/64")
fig, axarr = plt.subplots(3, 3, figsize=(12, 12))



im = plt.imread("../input/wheres-waldo/Hey-Waldo/256/waldo/10_3_1.jpg")

axarr[0][0].imshow(im)

axarr[0][0].set_title("256")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/256-bw/waldo/10_3_1.jpg")

axarr[0][1].imshow(im)

axarr[0][1].set_title("256-bw")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/256-gray/waldo/10_3_1.jpg")

axarr[0][2].imshow(im)

axarr[0][2].set_title("256-gray")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/128/waldo/10_7_2.jpg")

axarr[1][0].imshow(im)

axarr[1][0].set_title("128")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/128-bw/waldo/10_7_2.jpg")

axarr[1][1].imshow(im)

axarr[1][1].set_title("128-bw")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/128-gray/waldo/10_7_2.jpg")

axarr[1][2].imshow(im)

axarr[1][2].set_title("128-gray")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/64/waldo/10_15_4.jpg")

axarr[2][0].imshow(im)

axarr[2][0].set_title("64")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/64-bw/waldo/10_15_4.jpg")

axarr[2][1].imshow(im)

axarr[2][1].set_title("64-bw")



im = plt.imread("../input/wheres-waldo/Hey-Waldo/64-gray/waldo/10_15_4.jpg")

axarr[2][2].imshow(im)

axarr[2][2].set_title("64-gray")
fig, axarr = plt.subplots(1, 3, figsize=(12, 12))



im = plt.imread("../input/wheres-waldo/Hey-Waldo/64/waldo/10_15_4.jpg")

axarr[0].imshow(im)



im = plt.imread("../input/wheres-waldo/Hey-Waldo/64/waldo/12_2_1.jpg")

axarr[1].imshow(im)



im = plt.imread("../input/wheres-waldo/Hey-Waldo/64/waldo/12_3_12.jpg")

axarr[2].imshow(im)