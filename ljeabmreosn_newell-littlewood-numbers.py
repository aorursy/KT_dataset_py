# We need CMake 3.11+ and Python3 (with dev libraries).
!pip3 install cmake==3.17.3
!cmake --version

# The current version of `nlnum` is 0.0.5. This may change with time.
# You can remove the `==0.0.5` to fetch the latest version.
!pip3 install nlnum==0.0.5
import numpy as np
from nlnum import lrcoef, nlcoef, nlcoef_slow
N = lambda mu, nu, lam: lambda t: nlcoef(t*mu, t*nu, t*lam)
# Here are the input partitions.
mu = nu = lam = np.array([2, 1, 1])

# This is the scaling function F(t).
F = N(mu, nu, lam)
%%time
np.array([ F(t) for t in range(13+1) ])
%%time
nlcoef_slow(6*mu, 6*nu, 6*lam)