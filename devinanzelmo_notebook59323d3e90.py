!jupyter nbextension enable --py --sys-prefix widgetsnbextension

import pandas as pd

import numpy as np

import ipywidgets
def f(x):

    return x
ipywidgets.interact(f, x=10);
ipywidgets.__version__