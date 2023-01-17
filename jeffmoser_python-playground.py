%matplotlib inline
1+2
import scipy.misc
face = scipy.misc.face()
face.shape
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(face)
plt.show()
