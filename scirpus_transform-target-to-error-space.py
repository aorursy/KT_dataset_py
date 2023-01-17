import numpy as np # linear algebra

import pandas as pd
def transformtargets(originaltargets, weights):

    return np.log(np.power(originaltargets + 1.,np.sqrt(weights)))



def inversetransformtargets(transformedtargets, weights):

    return np.exp(transformedtargets/np.sqrt(weights)) - 1.
originaltargets =np.array([0,1,2,3])

weights = np.array([1,1.25,1,1])
transformtargets(originaltargets,weights)
inversetransformtargets(transformtargets(originaltargets,weights),weights)