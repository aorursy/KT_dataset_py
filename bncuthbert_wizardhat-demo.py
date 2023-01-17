import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot, transform

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
device = muse2016
dummy_outlet = ble2lsl.Dummy(device)
receiver = acquire.Receiver()