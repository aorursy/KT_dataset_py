!pip install bqplot

!jupyter nbextension enable --py --sys-prefix bqplot
import numpy as np

from bqplot import Scatter, LinearScale, Figure
sc_x, sc_y = LinearScale(), LinearScale()

scatter = Scatter(x=np.random.randn(10), y=np.random.randn(10), scales={'x': sc_x, 'y': sc_y})

Figure(marks=[scatter])