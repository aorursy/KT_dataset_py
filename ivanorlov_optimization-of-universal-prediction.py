import numpy as np

C = np.array([168000,  190000])   # submited constants
S = np.array([0.41933, 0.44292])  # submissions score
K = np.log(C + 1)

X = np.stack(([1., 1.], -2*K), axis=-1)      # A,B coefficients
A,B = np.matmul(np.linalg.inv(X), S*S - K*K) # solve equations 
print("A = %.4f B = %.4f" % (A, B))
import math
C_opt = math.exp(B) - 1.
print("C_opt = %.6f" % C_opt)
S_final = math.sqrt(A - B*B)
print("S_final = %.6f" % S_final)