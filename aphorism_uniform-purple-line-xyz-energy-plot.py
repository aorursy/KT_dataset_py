# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install colour-science
import colour
import numpy
import matplotlib
CMFS = colour.CMFS["CIE 2012 10 Degree Standard Observer"]

samples = 20

wave_XYZ_blue = numpy.tile(colour.wavelength_to_XYZ(CMFS.shape.start, cmfs=CMFS), (samples, 1))
wave_XYZ_red = numpy.tile(colour.wavelength_to_XYZ(CMFS.shape.end, cmfs=CMFS), (samples, 1))

wave_XYZ_blue_sum = numpy.sum(wave_XYZ_blue)
wave_XYZ_red_sum = numpy.sum(wave_XYZ_red)

wave_XYZ_sum = wave_XYZ_blue_sum + wave_XYZ_red_sum
wave_XYZ_blue_ratio = (1.0 - (wave_XYZ_blue_sum / wave_XYZ_sum))
wave_XYZ_red_ratio = (1.0 - (wave_XYZ_red_sum / wave_XYZ_sum))

divisions = numpy.tile(numpy.linspace(0.0, 1.0, samples), (3, 1)).T

wave_XYZ_mix = ((wave_XYZ_blue * divisions * wave_XYZ_blue_ratio) + (wave_XYZ_red * (1.0 - divisions) * wave_XYZ_red_ratio))

sample_xy = colour.XYZ_to_xy(wave_XYZ_mix)
matplotlib.pyplot.style.use({'figure.figsize': (11, 11)})

plot_1931_fig, plot_1931_ax = \
    colour.plotting.plot_chromaticity_diagram_CIE1931(
        show_diagram_colours=False,
        standalone=False
    )

plot_1931_ax.set_xlim(-0.03, 0.75)
plot_1931_ax.set_ylim(-0.03, 0.85)

plot_1976_fig, plot_1976_ax = \
    colour.plotting.plot_chromaticity_diagram_CIE1976UCS(
        show_diagram_colours=False,
        standalone=False
    )
plot_1976_ax.set_xlim(-0.01, 0.64)
plot_1976_ax.set_ylim(-0.01, 0.60)

plot_1931_ax.plot(sample_xy[:, 0], sample_xy[:, 1], "o", c="k", markersize=4)

sample_uv = colour.xy_to_Luv_uv(sample_xy)
plot_1976_ax.plot(sample_uv[:, 0], sample_uv[:, 1], "o", c="k", markersize=4)