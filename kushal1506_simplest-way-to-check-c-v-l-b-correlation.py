import numpy as np
cv=np.array([0.01469,0.01463,0.01457])

lb=np.array([0.01845,0.01839,0.01837])
np.corrcoef(cv,lb)[0,1]