import numpy as np
a = np.arange(12).reshape(3,4)

a
a[2, 1]
a[:2, 1:3]
a[1:3, :]
a > 5
a[a>5]