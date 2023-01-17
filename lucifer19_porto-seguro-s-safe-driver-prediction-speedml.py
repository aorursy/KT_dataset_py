from speedml import Speedml

import warnings

warnings.filterwarnings("ignore")



print("import ready")



sml = Speedml('../input/train.csv', '../input/test.csv',

              target='target', uid='id')



print("read data")



sml.plot.correlate()